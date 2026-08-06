"""
Multi-terminal loading report — Kpler slate + gap charts + raw Copernicus scene.

Auto vessel detection is intentionally excluded from the client-facing page.
"""

from __future__ import annotations

import base64
import io
import html
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from loading_gaps import (
    LoadingGapConfig,
    LocationSpec,
    analyze_loading_gaps,
    fetch_export_flows,
    fetch_port_calls,
    kpler_config,
    parse_env,
    plot_loading_gaps_report,
)

HERE = Path(__file__).resolve().parent
DEFAULT_REGISTRY = HERE / "terminals.yaml"
DEFAULT_ENV = HERE.parent / ".env"
DEFAULT_CACHE = HERE / ".cache" / "sentinel"

EVALSCRIPT_RGB = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B03", "B02"] }],
    output: [{ id: "rgb", bands: 3, sampleType: "UINT8" }]
  };
}
function evaluatePixel(sample) {
  return {
    rgb: [
      Math.min(255, sample.B04 * 255 * 3.5),
      Math.min(255, sample.B03 * 255 * 3.5),
      Math.min(255, sample.B02 * 255 * 3.5)
    ]
  };
}
"""

EVALSCRIPT_RGB_NDWI = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B03", "B02", "B08"] }],
    output: [
      { id: "rgb", bands: 3, sampleType: "UINT8" },
      { id: "ndwi", bands: 1, sampleType: "FLOAT32" }
    ]
  };
}
function evaluatePixel(sample) {
  var ndwi = (sample.B03 - sample.B08) / (sample.B03 + sample.B08 + 1e-6);
  return {
    rgb: [
      Math.min(255, sample.B04 * 255 * 3.5),
      Math.min(255, sample.B03 * 255 * 3.5),
      Math.min(255, sample.B02 * 255 * 3.5)
    ],
    ndwi: [ndwi]
  };
}
"""

# Default NDWI vessel-mask knobs for CPC-style debug panels (analyst view only).
_DEBUG_NDWI_PARAMS = {
    "slot_water_only": False,
    "use_ndwi_water": False,
    "ndwi_vessel_pct": 15,
    "ndwi_vessel_delta": 0.05,
    "water_gray_hi": 100,
}


@dataclass
class SatelliteScene:
    aoi_key: str
    scene_date: date | None
    cloud_cover: float | None
    age_days: int | None
    rgb: np.ndarray | None = None
    ndwi: np.ndarray | None = None
    debug_fig: Any = None
    note: str = ""


@dataclass
class TerminalSection:
    terminal_id: str
    display_name: str
    asof: date
    blurb: str
    slate: pd.DataFrame
    chart_fig: Any
    satellite: SatelliteScene
    idle_hours_7d: float | None = None
    export_ma_latest: float | None = None
    events: list[tuple[date, str]] = field(default_factory=list)


def load_registry(path: Path | None = None) -> dict[str, Any]:
    path = path or DEFAULT_REGISTRY
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def _location_from_cfg(cfg: dict[str, Any]) -> LocationSpec:
    k = cfg["kpler"]
    return LocationSpec(name=k["name"], kind=k["kind"])


def _merged_events(
    registry: dict[str, Any], terminal_cfg: dict[str, Any]
) -> list[tuple[date, str]]:
    events: list[tuple[date, str]] = []
    for row in registry.get("events") or []:
        events.append((_parse_date(row["date"]), str(row["label"])))
    for row in terminal_cfg.get("events") or []:
        events.append((_parse_date(row["date"]), str(row["label"])))
    # Stable unique by date+label
    seen: set[tuple[date, str]] = set()
    out: list[tuple[date, str]] = []
    for item in events:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return sorted(out, key=lambda x: x[0])


def fetch_trades_slate(
    location: LocationSpec,
    *,
    product: str,
    start_date: date,
    end_date: date,
    email: str,
    password: str,
) -> pd.DataFrame:
    """Voyage/trade rows including Scheduled fixtures (Port Calls omits many of these)."""
    from kpler.sdk.resources.trades import Trades

    client = Trades(kpler_config(email, password))
    kwargs: dict[str, Any] = {
        "products": [product],
        "start_date": start_date,
        "end_date": end_date,
        "with_forecast": True,
    }
    if location.kind == "zone":
        kwargs["from_zones"] = [location.name]
    else:
        kwargs["from_installations"] = [location.name]

    df = client.get(**kwargs)
    if df is None or df.empty:
        return pd.DataFrame()
    return df.copy()


def _trade_load_ts(row: pd.Series) -> pd.Timestamp:
    """Prefer berthing start, else origin ETA, else trade start."""
    for col in ("origin_start", "origin_eta_date", "start"):
        val = row.get(col)
        if pd.notna(val):
            return pd.Timestamp(val).normalize()
    return pd.NaT


def _trade_status_label(row: pd.Series) -> str:
    status = str(row.get("status") or "").strip().lower()
    if status == "scheduled":
        return "fixture"
    if status in {"in transit", "delivered"}:
        return "realized"
    if pd.notna(row.get("origin_end")) or pd.notna(row.get("origin_start")):
        return "realized"
    return status or "fixture"


def _clean_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def _trade_grade(row: pd.Series) -> str:
    for col in ("closest_ancestor_grade", "closest_ancestor_product", "closest_ancestor_group"):
        text = _clean_str(row.get(col))
        if text:
            return text
    return ""


def _trade_destination(row: pd.Series) -> str:
    """Best-available destination; fixtures often only have forecast/zone names."""
    for col in (
        "destination_location_name",
        "installation_destination_name",
        "next_forecasted_destination_location_name",
        "zone_destination_name",
    ):
        text = _clean_str(row.get(col))
        if text:
            return text
    return ""


def build_slate(
    trades: pd.DataFrame,
    asof: date,
    *,
    n_back: int = 2,
    n_forward: int = 3,
) -> pd.DataFrame:
    """Last N realized + next N scheduled rows around asof (from Trades)."""
    cols = [
        "load_date",
        "vessel",
        "vessel_class",
        "installation",
        "grade",
        "destination",
        "kbbl",
        "status",
        "vs_asof",
    ]
    if trades is None or trades.empty:
        return pd.DataFrame(columns=cols)

    df = trades.copy()
    df["load_date"] = df.apply(_trade_load_ts, axis=1)
    df = df.dropna(subset=["load_date"])
    asof_ts = pd.Timestamp(asof)

    status = df.apply(_trade_status_label, axis=1)
    realized_mask = (status == "realized") & (df["load_date"] <= asof_ts)
    forward_mask = df["load_date"] > asof_ts
    today_mask = df["load_date"] == asof_ts

    back = df.loc[realized_mask].sort_values("load_date").tail(n_back)
    today = df.loc[today_mask].sort_values("load_date")
    forward = df.loc[forward_mask].sort_values("load_date").head(n_forward)

    slate = pd.concat([back, today, forward], ignore_index=True)
    if slate.empty:
        return pd.DataFrame(columns=cols)

    inst = slate.get("installation_origin_name")
    if inst is None:
        inst = slate.get("origin_location_name", pd.Series("", index=slate.index))
    slate = slate.assign(_installation=inst.astype(str).replace({"nan": ""}))
    slate = slate.drop_duplicates(
        subset=["vessel_name", "load_date", "_installation"], keep="first"
    )

    bbl = pd.to_numeric(
        slate.get("cargo_origin_barrels_split_by_product"), errors="coerce"
    )
    out = pd.DataFrame(
        {
            "load_date": pd.to_datetime(slate["load_date"]).dt.date,
            "vessel": slate["vessel_name"].astype(str),
            "vessel_class": slate.get("vessel_type", pd.Series("", index=slate.index))
            .astype(str)
            .replace({"nan": ""}),
            "installation": slate["_installation"],
            "grade": slate.apply(_trade_grade, axis=1),
            "destination": slate.apply(_trade_destination, axis=1),
            "kbbl": (bbl / 1000.0).round(0),
            "status": slate.apply(_trade_status_label, axis=1),
        }
    )
    out["vs_asof"] = out["load_date"].map(
        lambda d: "today"
        if d == asof
        else (f"+{(d - asof).days}d" if d > asof else f"-{(asof - d).days}d")
    )
    return out.sort_values(["load_date", "vessel"]).reset_index(drop=True)


def draft_blurb(
    display_name: str,
    slate: pd.DataFrame,
    asof: date,
    satellite: SatelliteScene,
) -> str:
    """Soft-language desk blurb — review before sending."""
    lines = [display_name, ""]

    if slate.empty:
        lines.append("No recent / forward loadings in the Kpler pull.")
    else:
        today = slate.loc[slate["vs_asof"] == "today"]
        tomorrow = slate.loc[slate["vs_asof"] == "+1d"]

        def _name_list(frame: pd.DataFrame) -> str:
            bits = []
            for row in frame.itertuples(index=False):
                vc = getattr(row, "vessel_class", "") or ""
                bits.append(f"{row.vessel} ({vc})" if vc else row.vessel)
            return ", ".join(bits)

        if len(today):
            lines.append(
                f"{len(today)} loading(s) dated today: {_name_list(today)} "
                f"[{', '.join(sorted(set(today['status'])))}]."
            )
        else:
            lines.append("No loadings dated today in the Kpler slate.")

        if len(tomorrow):
            lines.append(
                f"{len(tomorrow)} scheduled tomorrow: {_name_list(tomorrow)} "
                f"[{', '.join(sorted(set(tomorrow['status'])))}]."
            )

        later = slate.loc[~slate["vs_asof"].isin(["today", "+1d"]) & slate["vs_asof"].str.startswith("+")]
        if len(later):
            names = ", ".join(later["vessel"].astype(str).tolist())
            lines.append(f"Further forward: {names}.")

    lines.append("")
    if satellite.scene_date is None:
        lines.append(
            "Satellite: no clear Sentinel-2 scene in lookback "
            "(or credentials / sentinelhub unavailable)."
        )
    else:
        age = satellite.age_days if satellite.age_days is not None else "?"
        cloud = (
            f", cloud≈{satellite.cloud_cover:.0f}%"
            if satellite.cloud_cover is not None
            else ""
        )
        lines.append(
            f"Latest clear Copernicus scene: {satellite.scene_date:%d-%m-%Y} "
            f"({age}d stale{cloud}). Auto vessel count not used — "
            "human read: [add loadings / approach note]."
        )
    if satellite.note:
        lines.append(satellite.note)
    return "\n".join(lines)


def _fig_to_base64(fig: plt.Figure, *, dpi: int = 120) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _rgb_to_base64(rgb: np.ndarray, *, dpi: int = 110) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(rgb)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    b64 = _fig_to_base64(fig, dpi=dpi)
    plt.close(fig)
    return b64


def _slate_to_html(slate: pd.DataFrame) -> str:
    if slate.empty:
        return "<p><em>No slate rows.</em></p>"
    show = slate.copy()
    if "kbbl" in show.columns:
        show["kbbl"] = show["kbbl"].map(
            lambda x: "" if pd.isna(x) else f"{int(x):,}"
        )
    return show.to_html(index=False, classes="slate", border=0, escape=True)


# --- Copernicus (optional; report still works without it) -------------------


def _sh_config(env: dict[str, str]):
    from sentinelhub import SHConfig

    config = SHConfig()
    config.sh_client_id = env.get("SH_CLIENT_ID") or env.get("COPERNICUS_CLIENT_ID")
    config.sh_client_secret = (
        env.get("SH_CLIENT_SECRET") or env.get("COPERNICUS_CLIENT_SECRET")
    )
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    config.sh_token_url = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
        "protocol/openid-connect/token"
    )
    return config


def search_scene_catalog(
    geojson: dict,
    start: date,
    end: date,
    *,
    env: dict[str, str],
    max_cloud: float = 30,
) -> list[dict[str, Any]]:
    """Return catalog hits newest-first: {date, cloud_cover, id}."""
    from sentinelhub import CRS, DataCollection, Geometry, SentinelHubCatalog

    config = _sh_config(env)
    if not config.sh_client_id or not config.sh_client_secret:
        return []

    geom = Geometry(geojson, CRS.WGS84)
    catalog = SentinelHubCatalog(config=config)
    search_iterator = catalog.search(
        DataCollection.SENTINEL2_L2A,
        geometry=geom,
        time=(start.isoformat(), end.isoformat()),
        filter=f"eo:cloud_cover < {max_cloud}",
        fields={
            "include": ["id", "properties.datetime", "properties.eo:cloud_cover"],
            "exclude": [],
        },
    )
    by_date: dict[str, dict[str, Any]] = {}
    for r in search_iterator:
        d = r["properties"]["datetime"][:10]
        cloud = float(r["properties"].get("eo:cloud_cover", 100))
        prev = by_date.get(d)
        if prev is None or cloud < prev["cloud_cover"]:
            by_date[d] = {"date": d, "cloud_cover": cloud, "id": r["id"]}
    return sorted(by_date.values(), key=lambda x: x["date"], reverse=True)


def fetch_rgb_scene(
    geojson: dict,
    scene_date: str,
    *,
    env: dict[str, str],
    aoi_key: str,
    resolution_m: float = 10,
    cache_dir: Path | None = None,
) -> np.ndarray | None:
    rgb, _ = fetch_rgb_ndwi_scene(
        geojson,
        scene_date,
        env=env,
        aoi_key=aoi_key,
        resolution_m=resolution_m,
        cache_dir=cache_dir,
        want_ndwi=False,
    )
    return rgb


def fetch_rgb_ndwi_scene(
    geojson: dict,
    scene_date: str,
    *,
    env: dict[str, str],
    aoi_key: str,
    resolution_m: float = 10,
    cache_dir: Path | None = None,
    want_ndwi: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    from sentinelhub import (
        CRS,
        DataCollection,
        Geometry,
        MimeType,
        SentinelHubRequest,
        bbox_to_dimensions,
    )

    config = _sh_config(env)
    if not config.sh_client_id or not config.sh_client_secret:
        return None, None

    cache_dir = cache_dir or DEFAULT_CACHE
    rgb_path = cache_dir / "s2" / aoi_key / f"{scene_date}_report_rgb.npy"
    ndwi_path = cache_dir / "s2" / aoi_key / f"{scene_date}_report_ndwi.npy"

    if rgb_path.exists() and (not want_ndwi or ndwi_path.exists()):
        rgb = np.load(rgb_path)
        ndwi = np.load(ndwi_path) if want_ndwi and ndwi_path.exists() else None
        return rgb, ndwi

    geom = Geometry(geojson, CRS.WGS84)
    size = bbox_to_dimensions(geom.bbox, resolution=resolution_m)
    s2 = DataCollection.SENTINEL2_L2A.define_from(
        "s2l2a", service_url=config.sh_base_url
    )

    if want_ndwi:
        request = SentinelHubRequest(
            evalscript=EVALSCRIPT_RGB_NDWI,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=s2,
                    time_interval=(scene_date, scene_date),
                    mosaicking_order="leastCC",
                )
            ],
            responses=[
                SentinelHubRequest.output_response("rgb", MimeType.PNG),
                SentinelHubRequest.output_response("ndwi", MimeType.TIFF),
            ],
            size=size,
            config=config,
            geometry=geom,
        )
        data = request.get_data()
        if not data:
            return None, None
        payload = data[0]
        if isinstance(payload, dict):
            rgb = np.asarray(payload["rgb.png"])
            ndwi = np.asarray(payload["ndwi.tif"], dtype=np.float32)
        else:
            rgb = np.asarray(data[0])
            ndwi = np.asarray(data[1], dtype=np.float32)
        if ndwi.ndim == 3:
            ndwi = ndwi[..., 0]
        ndwi = np.where(ndwi <= -900, np.nan, ndwi)
    else:
        request = SentinelHubRequest(
            evalscript=EVALSCRIPT_RGB,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=s2,
                    time_interval=(scene_date, scene_date),
                    mosaicking_order="leastCC",
                )
            ],
            responses=[SentinelHubRequest.output_response("rgb", MimeType.PNG)],
            size=size,
            config=config,
            geometry=geom,
        )
        data = request.get_data()
        if not data:
            return None, None
        payload = data[0]
        rgb = np.asarray(payload["rgb.png"] if isinstance(payload, dict) else payload)
        ndwi = None

    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(rgb_path, rgb)
    if ndwi is not None:
        np.save(ndwi_path, ndwi)
    return rgb, ndwi


def build_ndwi_debug_figure(
    rgb: np.ndarray,
    ndwi: np.ndarray,
    slots_px: list[dict],
    *,
    title: str = "",
    params: dict[str, Any] | None = None,
) -> plt.Figure:
    """NDWI + low-NDWI vessel mask inside SPM/berth slots (analyst appendix)."""
    from vessel_detect import (
        _ndwi_vessel_mask,
        _slot_valid_region,
        build_berth_slot_masks,
    )

    params = {**_DEBUG_NDWI_PARAMS, **(params or {})}
    vessel_any = np.zeros(ndwi.shape[:2], dtype=np.uint8)
    for _, slot_mask in build_berth_slot_masks(rgb.shape, slots_px):
        valid = _slot_valid_region(rgb, slot_mask, params, ndwi)
        vmask, _ = _ndwi_vessel_mask(ndwi, valid, params)
        vessel_any = np.maximum(vessel_any, vmask)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im = axes[0].imshow(ndwi, vmin=-0.3, vmax=0.8, cmap="RdYlBu")
    axes[0].set_title(f"{title} (raw NDWI)" if title else "raw NDWI")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    axes[1].imshow(vessel_any, cmap="gray")
    axes[1].set_title("low-NDWI vessel mask (slots)")

    for ax in axes:
        for slot in slots_px:
            name = slot.get("name", "")
            x1, y1, x2, y2 = slot["box"]
            ax.plot(
                [x1, x2, x2, x1, x1],
                [y1, y1, y2, y2, y1],
                color="yellow",
                lw=1.2,
            )
            ax.text(
                (x1 + x2) / 2,
                y1 - 3,
                name,
                color="yellow",
                fontsize=8,
                ha="center",
            )
        ax.axis("off")
    fig.tight_layout()
    return fig


def latest_satellite_scene(
    imagery_cfg: dict[str, Any],
    asof: date,
    *,
    env: dict[str, str],
    lookback_days: int = 45,
    max_cloud: float = 30,
) -> SatelliteScene:
    aoi_key = imagery_cfg["aoi_key"]
    geojson = imagery_cfg["geojson"]
    resolution_m = float(imagery_cfg.get("resolution_m", 10))
    show_debug = bool(imagery_cfg.get("show_debug"))
    slots_px = imagery_cfg.get("berth_slots_px") or []
    start = asof - timedelta(days=lookback_days)

    try:
        hits = search_scene_catalog(
            geojson, start, asof, env=env, max_cloud=max_cloud
        )
    except Exception as exc:  # noqa: BLE001 — report should degrade gracefully
        return SatelliteScene(
            aoi_key=aoi_key,
            scene_date=None,
            cloud_cover=None,
            age_days=None,
            note=f"Catalog search failed: {exc}",
        )

    if not hits:
        return SatelliteScene(
            aoi_key=aoi_key,
            scene_date=None,
            cloud_cover=None,
            age_days=None,
            note="No Sentinel-2 scenes under cloud threshold in lookback.",
        )

    hit = hits[0]
    scene_date = _parse_date(hit["date"])
    try:
        rgb, ndwi = fetch_rgb_ndwi_scene(
            geojson,
            hit["date"],
            env=env,
            aoi_key=aoi_key,
            resolution_m=resolution_m,
            want_ndwi=show_debug,
        )
    except Exception as exc:  # noqa: BLE001
        return SatelliteScene(
            aoi_key=aoi_key,
            scene_date=scene_date,
            cloud_cover=hit.get("cloud_cover"),
            age_days=(asof - scene_date).days,
            note=f"Scene fetch failed: {exc}",
        )

    debug_fig = None
    note = ""
    if show_debug and rgb is not None and ndwi is not None and slots_px:
        try:
            debug_fig = build_ndwi_debug_figure(
                rgb,
                ndwi,
                slots_px,
                title=f"{aoi_key} {hit['date']}",
            )
        except Exception as exc:  # noqa: BLE001
            note = f"Debug NDWI panel failed: {exc}"
    elif show_debug and not slots_px:
        note = "show_debug set but berth_slots_px missing in terminals.yaml."

    return SatelliteScene(
        aoi_key=aoi_key,
        scene_date=scene_date,
        cloud_cover=hit.get("cloud_cover"),
        age_days=(asof - scene_date).days,
        rgb=rgb,
        ndwi=ndwi,
        debug_fig=debug_fig,
        note=note,
    )


def build_terminal_section(
    terminal_id: str,
    *,
    asof: date,
    email: str,
    password: str,
    env: dict[str, str] | None = None,
    registry: dict[str, Any] | None = None,
    fetch_satellite: bool = True,
) -> TerminalSection:
    registry = registry or load_registry()
    env = env if env is not None else parse_env(DEFAULT_ENV)
    terminals = registry["terminals"]
    if terminal_id not in terminals:
        raise KeyError(
            f"Unknown terminal {terminal_id!r}. "
            f"Known: {list(terminals)}"
        )
    cfg = terminals[terminal_id]
    display = cfg.get("display_name", terminal_id)
    location = _location_from_cfg(cfg)
    chart_start = _parse_date(registry.get("chart_start", date(2026, 2, 1)))
    product = registry.get("product", "crude")
    flow_unit = registry.get("flow_unit", "kbd")
    ma_days = int(registry.get("ma_days", 7))
    n_back = int(registry.get("slate_back", 2))
    n_forward = int(registry.get("slate_forward", 3))
    events = _merged_events(registry, cfg)

    # Trades slate: recent realized + scheduled fixtures (matches Kpler Voyages UI).
    slate_start = asof - timedelta(days=21)
    slate_end = asof + timedelta(days=14)
    gap_cfg = LoadingGapConfig(
        location=location,
        product=product,
        start_date=chart_start,
        end_date=asof,
        flow_unit=flow_unit,
        ma_days=ma_days,
        with_forecast=False,
        only_realized_flows=True,
    )

    realized = fetch_port_calls(
        gap_cfg, email=email, password=password, require_berthing=True
    )
    trades = fetch_trades_slate(
        location,
        product=product,
        start_date=slate_start,
        end_date=slate_end,
        email=email,
        password=password,
    )
    flows = fetch_export_flows(gap_cfg, email=email, password=password)
    analysis = analyze_loading_gaps(realized, gap_cfg)
    fig = plot_loading_gaps_report(gap_cfg, analysis, flows, events=events)

    slate = build_slate(trades, asof, n_back=n_back, n_forward=n_forward)

    if fetch_satellite and cfg.get("imagery"):
        satellite = latest_satellite_scene(
            cfg["imagery"],
            asof,
            env=env,
            lookback_days=int(registry.get("sat_lookback_days", 45)),
            max_cloud=float(registry.get("sat_max_cloud", 30)),
        )
    else:
        satellite = SatelliteScene(
            aoi_key=cfg.get("imagery", {}).get("aoi_key", ""),
            scene_date=None,
            cloud_cover=None,
            age_days=None,
            note="Satellite fetch skipped.",
        )

    idle_7d = None
    if not analysis["daily_idle_hours"].empty:
        idle_7d = float(analysis["daily_idle_hours"].tail(7).sum())
    export_latest = None
    if not flows.empty:
        export_latest = float(flows.rolling(ma_days, min_periods=1).mean().iloc[-1])

    blurb = draft_blurb(display, slate, asof, satellite)
    return TerminalSection(
        terminal_id=terminal_id,
        display_name=display,
        asof=asof,
        blurb=blurb,
        slate=slate,
        chart_fig=fig,
        satellite=satellite,
        idle_hours_7d=idle_7d,
        export_ma_latest=export_latest,
        events=events,
    )


def build_report(
    terminal_ids: list[str],
    *,
    asof: date | None = None,
    email: str,
    password: str,
    env: dict[str, str] | None = None,
    registry_path: Path | None = None,
    fetch_satellite: bool = True,
) -> list[TerminalSection]:
    asof = asof or date.today()
    registry = load_registry(registry_path)
    env = env if env is not None else parse_env(DEFAULT_ENV)
    return [
        build_terminal_section(
            tid,
            asof=asof,
            email=email,
            password=password,
            env=env,
            registry=registry,
            fetch_satellite=fetch_satellite,
        )
        for tid in terminal_ids
    ]


def render_html(
    sections: list[TerminalSection],
    *,
    title: str | None = None,
    asof: date | None = None,
) -> str:
    asof = asof or (sections[0].asof if sections else date.today())
    title = title or f"Terminal loading report — {asof:%Y-%m-%d}"
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8"/>',
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:Segoe UI,Helvetica,Arial,sans-serif;margin:24px;color:#222;}",
        "h1{font-size:1.4rem;} h2{margin-top:2.2rem;border-bottom:1px solid #ccc;padding-bottom:4px;}",
        "pre.blurb{background:#f6f8fa;padding:12px 14px;border-radius:6px;white-space:pre-wrap;}",
        "table.slate{border-collapse:collapse;width:100%;font-size:0.92rem;margin:8px 0 16px;}",
        "table.slate th,table.slate td{border:1px solid #ddd;padding:6px 8px;text-align:left;}",
        "table.slate th{background:#f0f0f0;}",
        "img.chart,img.sat{max-width:100%;height:auto;border:1px solid #e0e0e0;}",
        ".meta{color:#555;font-size:0.9rem;margin:4px 0 12px;}",
        ".note{color:#666;font-size:0.85rem;font-style:italic;}",
        "</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f'<p class="meta">Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. '
        "Satellite auto-counts are not used.</p>",
    ]

    for sec in sections:
        parts.append(f"<h2>{html.escape(sec.display_name)}</h2>")
        kpi = []
        if sec.idle_hours_7d is not None:
            kpi.append(f"idle hours (last 7d sum): {sec.idle_hours_7d:.0f}")
        if sec.export_ma_latest is not None:
            kpi.append(f"export 7d MA: {sec.export_ma_latest:.0f}")
        if kpi:
            parts.append(f'<p class="meta">{" · ".join(kpi)}</p>')
        parts.append(f'<pre class="blurb">{html.escape(sec.blurb)}</pre>')
        parts.append("<h3>Loading slate</h3>")
        parts.append(_slate_to_html(sec.slate))
        parts.append("<h3>Breaks / occupancy / exports</h3>")
        parts.append(
            f'<img class="chart" alt="chart" src="data:image/png;base64,{_fig_to_base64(sec.chart_fig)}"/>'
        )
        parts.append("<h3>Latest Copernicus scene (raw)</h3>")
        sat = sec.satellite
        if sat.scene_date and sat.rgb is not None:
            parts.append(
                f'<p class="meta">{sat.aoi_key} — {sat.scene_date:%Y-%m-%d}'
                f" · age {sat.age_days}d"
                + (
                    f" · cloud {sat.cloud_cover:.0f}%"
                    if sat.cloud_cover is not None
                    else ""
                )
                + "</p>"
            )
            parts.append(
                f'<img class="sat" alt="satellite" src="data:image/png;base64,{_rgb_to_base64(sat.rgb)}"/>'
            )
            parts.append(
                '<p class="note">Human eyeball only — approaching traffic outside '
                "berths/SPMs is not scored.</p>"
            )
            if sat.debug_fig is not None:
                parts.append("<h3>Analyst appendix — NDWI debug</h3>")
                parts.append(
                    '<p class="note">Low-NDWI mask inside SPM slots aids eyeballing; '
                    "not used for the cover blurb or auto counts.</p>"
                )
                parts.append(
                    f'<img class="sat" alt="ndwi-debug" '
                    f'src="data:image/png;base64,{_fig_to_base64(sat.debug_fig)}"/>'
                )
        else:
            parts.append(
                f"<p><em>{html.escape(sat.note or 'No satellite scene.')}</em></p>"
            )
        if sat.note and sat.rgb is not None:
            parts.append(f"<p class=\"note\">{html.escape(sat.note)}</p>")

    parts.append("</body></html>")
    return "\n".join(parts)


def write_html(
    sections: list[TerminalSection],
    out_path: Path,
    *,
    title: str | None = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    asof = sections[0].asof if sections else date.today()
    out_path.write_text(
        render_html(sections, title=title, asof=asof), encoding="utf-8"
    )
    return out_path
