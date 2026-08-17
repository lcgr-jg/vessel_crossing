"""Kayrros COI vs Kpler loadings vs SoH crossings — no Copernicus calls."""

from __future__ import annotations

import contextlib
import io
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
CACHE = HERE / "cache"
OUT = HERE / "output"


def _ensure_kayrros_on_path() -> Path | None:
    """Notebook kernels do not see kayros/kayrros_api unless we add it."""
    candidates = [
        BASE.parent / "kayros" / "kayrros_api",
        BASE.parent / "kayrros_api",
    ]
    for path in candidates:
        if (path / "kayrros_client").is_dir():
            resolved = str(path)
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
            return path
    return None


_ensure_kayrros_on_path()

# Port-call helpers live next to the terminal report; keep a single Kpler client.
_GAPS = BASE / "port_loading_gaps"
if str(_GAPS) not in sys.path:
    sys.path.insert(0, str(_GAPS))

from loading_gaps import (  # noqa: E402
    LoadingGapConfig,
    LocationSpec,
    fetch_export_flows,
    fetch_port_calls,
    occupancy_at,
    parse_env,
)


class SatelliteFetchBlocked(RuntimeError):
    """Raised if anyone tries to turn Copernicus on from this folder."""


def load_config(path: Path | None = None) -> dict[str, Any]:
    path = path or (HERE / "config.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _guard_no_satellite(cfg: dict[str, Any]) -> None:
    if cfg.get("fetch_satellite"):
        raise SatelliteFetchBlocked(
            "inventory_soh_residual must not call Copernicus / Sentinel Hub. "
            "Use cached port_loading_gaps/vessel_detections.csv only."
        )


def to_mbbl(value, unit: str) -> float | pd.Series:
    """Normalise Kayrros / Kpler units to million barrels."""
    if unit in ("bbl", "barrels"):
        return value / 1e6
    if unit in ("kbbl", "kb"):
        return value / 1e3
    if unit == "mbbl":
        return value
    raise ValueError(f"Unknown unit {unit!r}")


def kayrros_session():
    """Authenticate without printing Kayrros client chatter (includes secrets-adjacent notices)."""
    if _ensure_kayrros_on_path() is None:
        raise ModuleNotFoundError(
            "kayrros_client not found. Expected kayros/kayrros_api next to vessel_crossing."
        )
    from kayrros_client import KSession

    with contextlib.redirect_stdout(io.StringIO()):
        return KSession()


def _cache_path(kind: str, shape_id: str) -> Path:
    safe = shape_id.replace("/", "_")
    return CACHE / f"{kind}_{safe}.csv"


def fetch_onshore_series(
    session,
    shape_id: str,
    *,
    start: date,
    end: date,
    daily: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """Daily COI for one Kayrros shape. Cached so re-runs do not re-hit the API."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = _cache_path("onshore", shape_id)
    if path.exists() and not refresh:
        df = pd.read_csv(path)
        df["value_date"] = pd.to_datetime(df["value_date"])
        return df

    from kayrros_client.storage_onshore import stocks

    df = stocks.get_timeseries(
        session,
        _id=shape_id,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        daily=daily,
    )
    if df is None or df.empty:
        empty = pd.DataFrame(columns=["value_date", "stock", "capacity", "stock_change", "location"])
        empty.to_csv(path, index=False)
        return empty

    out = df.copy()
    out["value_date"] = pd.to_datetime(out["value_date"])
    out["shape_id"] = shape_id
    out.to_csv(path, index=False)
    return out


def fetch_tbt_series(
    session,
    shape_id: str,
    *,
    start: date,
    end: date,
    daily: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """Tank-by-tank COI. Recalculates stock_change on ascending dates (API order is newest-first)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = _cache_path("tbt", shape_id)
    if path.exists() and not refresh:
        df = pd.read_csv(path)
        df["value_date"] = pd.to_datetime(df["value_date"])
        if not df.empty:
            cmin, cmax = df["value_date"].min().date(), df["value_date"].max().date()
            # Re-pull if the cache does not cover the requested window.
            if cmin <= start and cmax >= end:
                return df.loc[
                    (df["value_date"].dt.date >= start) & (df["value_date"].dt.date <= end)
                ].copy()

    from kayrros_client.storage_onshore import stocks

    df = stocks.get_timeseries(
        session,
        _id=shape_id,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        daily=daily,
        tbt=True,
    )
    if df is None or df.empty or "tank_id" not in df.columns:
        empty = pd.DataFrame(columns=["value_date", "tank_id", "stock", "stock_change"])
        empty.to_csv(path, index=False)
        return empty

    out = df.copy()
    out["value_date"] = pd.to_datetime(out["value_date"]).dt.normalize()
    out = out.sort_values(["tank_id", "value_date"])
    out["stock_change"] = out.groupby("tank_id", sort=False)["stock"].diff().fillna(0.0)
    out["shape_id"] = shape_id
    out.to_csv(path, index=False)
    return out


def tank_change_wide(
    tbt: pd.DataFrame,
    *,
    start: date,
    end: date,
    display_unit: str = "kb",
) -> pd.DataFrame:
    """Date x tank_id stock_change, plus total_build (positives) and total_draw (negatives).

    display_unit: 'kb' matches the Kayrros TBT screenshot (bbl / 1000).
    """
    idx = pd.date_range(start, end, freq="D")
    if tbt is None or tbt.empty or "tank_id" not in tbt.columns:
        return pd.DataFrame(index=idx, columns=["total_build", "total_draw"])

    long = tbt.copy()
    long["value_date"] = pd.to_datetime(long["value_date"]).dt.normalize()
    wide = (
        long.pivot_table(
            index="value_date",
            columns="tank_id",
            values="stock_change",
            aggfunc="sum",
        )
        .reindex(idx)
        .fillna(0.0)
    )
    wide = wide.reindex(sorted(wide.columns), axis=1)
    # Convert bbl -> display unit after the pivot so sums stay consistent.
    scale = {"bbl": 1.0, "kb": 1e-3, "mbbl": 1e-6}[display_unit]
    wide = wide * scale
    tank_cols = list(wide.columns)
    wide["total_build"] = wide[tank_cols].clip(lower=0).sum(axis=1)
    wide["total_draw"] = wide[tank_cols].clip(upper=0).sum(axis=1)
    wide.index.name = "date"
    return wide


def resolve_kayrros_shape_ids(cfg: dict[str, Any], names: list[str]) -> list[tuple[str, list[str]]]:
    """Map friendly terminal ids OR raw Kayrros shape ids -> (label, [shape_ids]).

    Accepts either ``zirku`` (config key) or ``zirkuIslandTerminal_adnoc`` (Kayrros id).
    """
    resolved: list[tuple[str, list[str]]] = []
    known_shapes = {
        sid: tid
        for tid, spec in cfg["terminals"].items()
        for sid in (spec.get("kayrros_ids") or [])
    }
    for name in names:
        if name in cfg["terminals"]:
            ids = list(cfg["terminals"][name].get("kayrros_ids") or [])
            label = cfg["terminals"][name]["display_name"]
            if not ids:
                warnings.warn(f"{name!r} has no kayrros_ids in config.yaml")
            resolved.append((label, ids))
        elif name in known_shapes:
            tid = known_shapes[name]
            resolved.append((cfg["terminals"][tid]["display_name"], [name]))
        else:
            # Raw Kayrros shape id not listed in config — still fetch it.
            resolved.append((name, [name]))
    return resolved


def pull_tbt_bundle(
    cfg: dict[str, Any],
    *,
    start: date,
    end: date,
    session=None,
    refresh: bool = False,
    terminal_ids: list[str] | None = None,
    display_unit: str = "kb",
) -> dict[str, pd.DataFrame]:
    """Wide tank-change tables. Keys are display labels; accepts config or Kayrros ids."""
    _guard_no_satellite(cfg)
    session = session or kayrros_session()
    wanted = terminal_ids or [
        tid for tid, spec in cfg["terminals"].items() if spec.get("kayrros_ids")
    ]
    fetch_start = start - timedelta(days=3)
    out: dict[str, pd.DataFrame] = {}
    for label, shape_ids in resolve_kayrros_shape_ids(cfg, wanted):
        frames = [
            fetch_tbt_series(
                session,
                sid,
                start=fetch_start,
                end=end,
                daily=cfg["kayrros"]["daily"],
                refresh=refresh,
            )
            for sid in shape_ids
        ]
        tbt = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        out[label] = tank_change_wide(tbt, start=start, end=end, display_unit=display_unit)
    return out


def tank_draw_vs_kpler(
    *,
    shape_id: str,
    kpler_name: str,
    kpler_kind: str = "zone",
    start: date,
    end: date,
    email: str,
    password: str,
    session=None,
    product: str = "crude",
    refresh: bool = False,
    daily: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Excel-style compare: Kayrros tank draws vs Kpler loadings (kb).

    Returns
    -------
    wide : date x tank_id + total_build + total_draw (kb)
    compare : observation dates with tank_draw_kb, kpler_loadings_kb, gap_kb

    ``gap_kb = |tank_draw_kb| - kpler_loadings_kb``. Positive gap ⇒ Kayrros saw
    more offtake than Kpler booked (possible missed loading).
    """
    session = session or kayrros_session()
    fetch_start = start - timedelta(days=3)
    tbt = fetch_tbt_series(
        session, shape_id, start=fetch_start, end=end, daily=daily, refresh=refresh
    )
    if tbt.empty:
        raise ValueError(f"No tank-level COI for Kayrros shape {shape_id!r}")

    wide = tank_change_wide(tbt, start=start, end=end, display_unit="kb")

    # Days tanks actually moved (Kayrros update cadence) — matches Excel observation rows.
    obs_dates = list(wide.index[(wide["total_build"] + wide["total_draw"].abs()) > 0.5])
    if len(obs_dates) < 2:
        raise ValueError(
            f"Not enough tank-change days for {shape_id!r} between {start} and {end}. "
            "Try refresh=True or a wider window."
        )

    kpler = fetch_kpler_flow_mbbl(
        name=kpler_name,
        kind=kpler_kind,
        product=product,
        start=start,
        end=end,
        email=email,
        password=password,
    )
    # mbbl -> kb
    kpler_kb = (kpler * 1000.0).reindex(pd.date_range(start, end, freq="D")).fillna(0.0)

    rows = []
    prev = None
    for dt in obs_dates:
        if prev is None:
            prev = dt
            continue
        # Draws / builds between previous and current observation (inclusive of current).
        window = wide.loc[(wide.index > prev) & (wide.index <= dt)]
        tank_draw_kb = float(window["total_draw"].sum())
        tank_build_kb = float(window["total_build"].sum())
        kpler_loadings_kb = float(kpler_kb.loc[(kpler_kb.index > prev) & (kpler_kb.index <= dt)].sum())
        rows.append(
            {
                "date": dt,
                "from_date": prev,
                "tank_draw_kb": tank_draw_kb,
                "tank_build_kb": tank_build_kb,
                "kpler_loadings_kb": kpler_loadings_kb,
                # Compare magnitudes: draw is negative, loadings positive.
                "gap_kb": abs(tank_draw_kb) - kpler_loadings_kb,
            }
        )
        prev = dt

    compare = pd.DataFrame(rows)
    if not compare.empty:
        compare = compare.set_index("date")
    return wide, compare


def fetch_floating_series(
    session,
    shape_id: str,
    *,
    start: date,
    end: date,
    idle_days: int = 10,
    refresh: bool = False,
) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = _cache_path(f"floating{idle_days}", shape_id)
    if path.exists() and not refresh:
        df = pd.read_csv(path)
        df["value_date"] = pd.to_datetime(df["value_date"])
        return df

    from kayrros_client.storage_floating import stocks as flosto

    df = flosto.get_timeseries(
        session,
        _id=shape_id,
        num_days_idle=idle_days,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
    )
    if df is None or df.empty:
        empty = pd.DataFrame(columns=["value_date", "total_volume", "total_vessel_count"])
        empty.to_csv(path, index=False)
        return empty
    df = df.copy()
    df["value_date"] = pd.to_datetime(df["value_date"])
    df.to_csv(path, index=False)
    return df


def daily_stock_mbbl(df: pd.DataFrame, *, unit: str, start: date, end: date) -> pd.Series:
    """Calendar daily stock (ffill). Change is measured, not interpolated."""
    idx = pd.date_range(start, end, freq="D")
    if df is None or df.empty or "stock" not in df.columns:
        return pd.Series(np.nan, index=idx, dtype=float)
    s = (
        df.dropna(subset=["stock"])
        .assign(value_date=lambda x: pd.to_datetime(x["value_date"]).dt.normalize())
        .groupby("value_date")["stock"]
        .last()
        .sort_index()
    )
    s = to_mbbl(s, unit)
    return s.reindex(idx).ffill()


def fetch_kpler_flow_mbbl(
    *,
    name: str,
    kind: str,
    product: str,
    start: date,
    end: date,
    email: str,
    password: str,
    alt_names: list[str] | None = None,
) -> pd.Series:
    """Daily realized crude exports in mbbl. Tries alt names if Kpler rejects the first."""
    last_empty = pd.Series(np.nan, index=pd.date_range(start, end, freq="D"), dtype=float, name=name)
    for candidate in [name, *(alt_names or [])]:
        cfg = LoadingGapConfig(
            location=LocationSpec(name=candidate, kind=kind),  # type: ignore[arg-type]
            product=product,
            start_date=start,
            end_date=end,
            flow_unit="kb",
            with_forecast=False,
            only_realized_flows=True,
        )
        try:
            kb = fetch_export_flows(cfg, email=email, password=password)
        except Exception as exc:
            warnings.warn(f"Kpler flows skipped for {kind} {candidate!r}: {exc}")
            continue
        if kb is None or kb.empty:
            continue
        out = to_mbbl(kb.astype(float), "kb")
        out.name = candidate
        return out
    return last_empty


def mapping_table(cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for tid, spec in cfg["terminals"].items():
        rows.append(
            {
                "terminal_id": tid,
                "display_name": spec["display_name"],
                "role": spec["role"],
                "kpler": spec["kpler"]["name"],
                "kayrros_n": len(spec.get("kayrros_ids") or []),
                "kayrros_ids": ", ".join(spec.get("kayrros_ids") or []),
                "sat_aois": ", ".join(spec.get("sat_aois") or []),
                "note": spec.get("note", ""),
            }
        )
    return pd.DataFrame(rows)


def load_soh_crude_daily(cfg: dict[str, Any], *, start: date, end: date) -> pd.DataFrame:
    """Laden crude (+ VLCC-inferred) SoH exits, barrels/day."""
    crossing_path = BASE / cfg["paths"]["crossing"]
    afra_path = BASE / cfg["paths"]["afra"]
    raw = pd.read_excel(crossing_path, sheet_name="SoH Crossing")
    afra = pd.read_excel(afra_path)

    def parse_dmy(series: pd.Series) -> pd.Series:
        return pd.to_datetime(series, dayfirst=True, errors="coerce")

    def to_number(series: pd.Series) -> pd.Series:
        return pd.to_numeric(
            series.astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce",
        )

    def map_afra_class(dwt_kdwt: float) -> str | None:
        if pd.isna(dwt_kdwt):
            return None
        table = afra.sort_values("capacity_lower_bound")
        for _, row in table.iterrows():
            if row["capacity_lower_bound"] <= dwt_kdwt <= row["capacity_upper_bound"]:
                return row["vessel_class"]
        if dwt_kdwt < table["capacity_lower_bound"].min():
            return "Below GP"
        return "Above ULCC"

    df = raw.copy()
    df["Crossing Date"] = parse_dmy(df["Crossing Date"])
    df["Quantity"] = to_number(df["Quantity"]).fillna(0.0)
    df["Deadweight"] = to_number(df["Deadweight"])
    df["dwt_kdwt"] = df["Deadweight"] / 1000.0
    df["afra_class"] = df["dwt_kdwt"].map(map_afra_class)

    crude_tags = {"Crude/Co"}
    crude_like = {"VLCC", "ULCC", "Above ULCC"}

    def cargo_group(row: pd.Series) -> str:
        if row["Loading State"] == "Ballast":
            return "ballast"
        tag = row["Cargo Type"]
        if pd.notna(tag):
            if tag in crude_tags:
                return "crude"
            return "other"
        if row["afra_class"] in crude_like:
            return "likely_crude"
        return "unknown"

    df["cargo_group"] = df.apply(cargo_group, axis=1)
    crude = df[
        (df["Direction"] == "Exited MEG")
        & (df["cargo_group"].isin(["crude", "likely_crude"]))
        & (df["Crossing Date"] >= pd.Timestamp(start))
        & (df["Crossing Date"] <= pd.Timestamp(end))
    ].copy()
    daily = (
        crude.groupby(crude["Crossing Date"].dt.normalize())["Quantity"]
        .sum()
        .rename("soh_crude_bbl")
    )
    idx = pd.date_range(start, end, freq="D")
    daily = daily.reindex(idx, fill_value=0.0)
    out = pd.DataFrame({"soh_crude_mbbl": daily / 1e6})
    out.index.name = "date"
    return out


def load_cached_sat(cfg: dict[str, Any]) -> pd.DataFrame:
    """On-disk Copernicus detections only — never downloads a scene."""
    path = BASE / cfg["paths"]["detections"]
    if not path.exists():
        return pd.DataFrame(columns=["date", "aoi", "terminal_id", "size_class", "vessel_count"])
    det = pd.read_csv(path)
    det["date"] = pd.to_datetime(det["date"], errors="coerce")
    ok = det["quality"].fillna("").eq("ok")
    det = det.loc[ok].copy()
    aoi_to_term = {}
    for tid, spec in cfg["terminals"].items():
        for aoi in spec.get("sat_aois") or []:
            aoi_to_term[aoi] = tid
    det["terminal_id"] = det["aoi"].map(aoi_to_term)
    return det.dropna(subset=["date", "terminal_id"])


def kpler_credentials(cfg: dict[str, Any]) -> tuple[str, str]:
    env = parse_env(BASE / cfg["paths"]["env"])
    email = env.get("KPLER_EMAIL") or env.get("EMAIL")
    password = env.get("KPLER_PASSWORD") or env.get("PASSWORD")
    if not email or not password:
        raise RuntimeError(f"Missing KPLER_EMAIL / KPLER_PASSWORD in {BASE / cfg['paths']['env']}")
    return email, password


def pull_kayrros_bundle(
    cfg: dict[str, Any],
    *,
    start: date,
    end: date,
    session=None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Country + bypass + terminal COI, plus MEG floating storage."""
    _guard_no_satellite(cfg)
    session = session or kayrros_session()
    kcfg = cfg["kayrros"]
    unit = kcfg["onshore_unit"]
    idx = pd.date_range(start, end, freq="D")

    countries = {}
    for cid in cfg["countries_inside_gulf"]:
        raw = fetch_onshore_series(
            session, cid, start=start, end=end, daily=kcfg["daily"], refresh=refresh
        )
        countries[cid] = daily_stock_mbbl(raw, unit=unit, start=start, end=end)

    country_sum = pd.DataFrame(countries).sum(axis=1).rename("country_sum_mbbl")

    bypass = {}
    for sid in cfg["bypass_kayrros_ids"]:
        raw = fetch_onshore_series(
            session, sid, start=start, end=end, daily=kcfg["daily"], refresh=refresh
        )
        bypass[sid] = daily_stock_mbbl(raw, unit=unit, start=start, end=end)
    bypass_sum = pd.DataFrame(bypass).sum(axis=1).rename("bypass_stock_mbbl") if bypass else pd.Series(0.0, index=idx)

    i_inside = (country_sum - bypass_sum).rename("i_inside_mbbl")

    terminals = {}
    for tid, spec in cfg["terminals"].items():
        ids = spec.get("kayrros_ids") or []
        if not ids:
            terminals[tid] = pd.Series(np.nan, index=idx, name=tid)
            continue
        parts = []
        for sid in ids:
            raw = fetch_onshore_series(
                session, sid, start=start, end=end, daily=kcfg["daily"], refresh=refresh
            )
            parts.append(daily_stock_mbbl(raw, unit=unit, start=start, end=end))
        terminals[tid] = pd.concat(parts, axis=1).sum(axis=1).rename(tid)

    float_raw = fetch_floating_series(
        session,
        kcfg["floating_shape"],
        start=start,
        end=end,
        idle_days=int(kcfg["floating_idle_days"]),
        refresh=refresh,
    )
    if float_raw.empty:
        floating = pd.Series(np.nan, index=idx, name="floating_mbbl")
    else:
        fv = (
            float_raw.assign(value_date=lambda x: pd.to_datetime(x["value_date"]).dt.normalize())
            .groupby("value_date")["total_volume"]
            .last()
        )
        floating = to_mbbl(fv, kcfg["floating_unit"]).reindex(idx).ffill().rename("floating_mbbl")

    return {
        "countries": countries,
        "country_sum": country_sum,
        "bypass": bypass,
        "bypass_sum": bypass_sum,
        "i_inside": i_inside,
        "terminals": terminals,
        "floating": floating,
    }


def pull_kpler_bundle(
    cfg: dict[str, Any],
    *,
    start: date,
    end: date,
    email: str,
    password: str,
) -> dict[str, Any]:
    _guard_no_satellite(cfg)
    product = cfg["product"]
    countries = {}
    for name in cfg["kpler_inside_countries"]:
        countries[name] = fetch_kpler_flow_mbbl(
            name=name, kind="zone", product=product, start=start, end=end,
            email=email, password=password,
        )
    country_export = pd.DataFrame(countries).sum(axis=1).rename("kpler_country_mbbl")

    bypass = {}
    for spec in cfg["kpler_bypass"]:
        s = fetch_kpler_flow_mbbl(
            name=spec["name"],
            kind=spec["kind"],
            product=product,
            start=start,
            end=end,
            email=email,
            password=password,
            alt_names=list(spec.get("alt_names") or []),
        )
        bypass[spec["bucket"]] = s
    bypass_df = pd.DataFrame(bypass)
    bypass_sum = bypass_df.sum(axis=1).rename("kpler_bypass_mbbl") if not bypass_df.empty else pd.Series(dtype=float)

    terminals = {}
    for tid, spec in cfg["terminals"].items():
        k = spec["kpler"]
        terminals[tid] = fetch_kpler_flow_mbbl(
            name=k["name"],
            kind=k["kind"],
            product=product,
            start=start,
            end=end,
            email=email,
            password=password,
            alt_names=list(k.get("alt_names") or []),
        )

    kpler_inside = (country_export.sub(bypass_sum, fill_value=0)).rename("kpler_inside_mbbl")
    return {
        "countries": countries,
        "country_export": country_export,
        "bypass": bypass,
        "bypass_sum": bypass_sum,
        "kpler_inside": kpler_inside,
        "terminals": terminals,
    }


def align_daily(*series: pd.Series, start: date, end: date) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="D")
    cols = {}
    for s in series:
        if s is None:
            continue
        if s.empty:
            cols[s.name or f"s{len(cols)}"] = pd.Series(np.nan, index=idx)
            continue
        x = s.copy()
        x.index = pd.to_datetime(x.index, errors="coerce")
        x = x.loc[x.index.notna()]
        x.index = x.index.normalize()
        cols[x.name or f"s{len(cols)}"] = x.reindex(idx)
    return pd.DataFrame(cols)


def build_gulf_panel(
    kay: dict[str, Any],
    kpler: dict[str, Any],
    soh: pd.DataFrame,
    *,
    start: date,
    end: date,
    lag_days: int,
    ma_days: int,
) -> pd.DataFrame:
    """S_t = SoH exits + ΔI_inside + Δfloating + bypass loadings.

    If production − refinery runs is sticky, S_t should stay in a band.
    A break lower with no inventory build is the SoH-undercount cue.
    """
    i_in = kay["i_inside"].rename("i_inside_mbbl")
    fl = kay["floating"].rename("floating_mbbl")
    panel = align_daily(
        i_in,
        fl,
        kpler["kpler_inside"].rename("kpler_inside_mbbl"),
        kpler["bypass_sum"].rename("kpler_bypass_mbbl"),
        soh["soh_crude_mbbl"],
        start=start,
        end=end,
    )
    # Flows can be missing; stock levels must not be zero-filled (that fakes a giant draw/build).
    for col in ("kpler_inside_mbbl", "kpler_bypass_mbbl", "soh_crude_mbbl"):
        if col in panel.columns:
            panel[col] = panel[col].fillna(0.0)
    panel["d_i_inside"] = panel["i_inside_mbbl"].diff()
    panel["d_floating"] = panel["floating_mbbl"].diff()
    panel["implied_offtake"] = -panel["d_i_inside"]
    panel["kpler_residual"] = panel["implied_offtake"] - panel["kpler_inside_mbbl"]
    panel["soh_lagged"] = panel["soh_crude_mbbl"].shift(-lag_days)
    panel["s_t"] = (
        panel["soh_crude_mbbl"]
        + panel["d_i_inside"]
        + panel["d_floating"]
        + panel["kpler_bypass_mbbl"]
    )
    for col in ("soh_crude_mbbl", "kpler_inside_mbbl", "implied_offtake", "s_t", "kpler_residual"):
        panel[f"{col}_ma"] = panel[col].rolling(ma_days, min_periods=1).mean()
    return panel


def build_terminal_panel(
    cfg: dict[str, Any],
    kay: dict[str, Any],
    kpler: dict[str, Any],
    *,
    start: date,
    end: date,
    ma_days: int,
) -> dict[str, pd.DataFrame]:
    out = {}
    for tid, spec in cfg["terminals"].items():
        stock = kay["terminals"][tid].rename("stock_mbbl")
        load = kpler["terminals"].get(tid, pd.Series(dtype=float)).rename("kpler_mbbl")
        df = align_daily(stock, load, start=start, end=end)
        df["kpler_mbbl"] = df["kpler_mbbl"].fillna(0.0)
        df["d_stock"] = df["stock_mbbl"].diff()
        df["implied_offtake"] = -df["d_stock"]
        df["residual"] = df["implied_offtake"] - df["kpler_mbbl"]
        df["residual_ma"] = df["residual"].rolling(ma_days, min_periods=1).mean()
        df["kpler_ma"] = df["kpler_mbbl"].rolling(ma_days, min_periods=1).mean()
        df["implied_ma"] = df["implied_offtake"].rolling(ma_days, min_periods=1).mean()
        df.attrs["role"] = spec["role"]
        df.attrs["display_name"] = spec["display_name"]
        out[tid] = df
    return out


def sat_vs_kpler_occupancy(
    cfg: dict[str, Any],
    sat: pd.DataFrame,
    *,
    email: str,
    password: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Cached sat occupancy vs Kpler port-call occupancy. No new imagery."""
    if sat.empty:
        return pd.DataFrame()
    product = cfg["product"]
    rows = []
    by_term = sat.groupby("terminal_id")
    for tid, chunk in by_term:
        spec = cfg["terminals"][tid]
        gap_cfg = LoadingGapConfig(
            location=LocationSpec(name=spec["kpler"]["name"], kind=spec["kpler"]["kind"]),  # type: ignore[arg-type]
            product=product,
            start_date=start,
            end_date=end,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                calls = fetch_port_calls(
                    gap_cfg, email=email, password=password, require_berthing=True
                )
            except Exception as exc:
                warnings.warn(f"Kpler port calls skipped for {tid}: {exc}")
                calls = pd.DataFrame()
        for rec in chunk.itertuples(index=False):
            ts = pd.Timestamp(rec.date).normalize() + pd.Timedelta(hours=12)
            kpler_n = occupancy_at(calls, ts) if not calls.empty else 0
            rows.append(
                {
                    "date": rec.date,
                    "terminal_id": tid,
                    "display_name": spec["display_name"],
                    "aoi": rec.aoi,
                    "size_class": getattr(rec, "size_class", None),
                    "sat_vessels": rec.vessel_count,
                    "kpler_occupancy": kpler_n,
                    "sat_only": int(kpler_n == 0),
                }
            )
    return pd.DataFrame(rows).sort_values(["date", "terminal_id"])


def score_window(panel: pd.DataFrame, *, band_days: int) -> dict[str, float | str]:
    """Under/over cues from the Gulf identity -- not a cargo bill of lading.

    implied offtake = -sum(dI). Only compare Kpler to tanks when that offtake is
    a real draw (positive). A build makes the ratio meaningless.
    """
    s = panel["s_t"].dropna()
    if s.empty:
        return {"verdict": "NO DATA"}
    recent = s.tail(max(band_days, 7))
    band = s.rolling(band_days, min_periods=max(7, band_days // 2)).mean()
    last_s = float(recent.mean())
    last_band = float(band.dropna().iloc[-1]) if band.dropna().size else float("nan")
    soh = float(panel["soh_crude_mbbl"].sum())
    kpler_in = float(panel["kpler_inside_mbbl"].sum())
    implied = float(panel["implied_offtake"].sum())
    d_i = float(panel["d_i_inside"].sum())
    soh_vs_kpler = (soh / kpler_in) if kpler_in else float("nan")
    kpler_vs_implied = (kpler_in / implied) if implied > 1.0 else float("nan")
    s_gap = last_s - last_band
    tanks_drew = implied > 1.0

    if soh_vs_kpler < 0.85 and s_gap < -0.15 * abs(last_band or 0):
        verdict = (
            "SOH UNDERCOUNT CUE -- exits lag Kpler loadings and the inventory identity"
        )
    elif soh_vs_kpler < 0.85:
        verdict = "SOH LIGHT VS KPLER -- loadings may still be leaving (check lag / floating)"
    elif soh_vs_kpler > 1.15:
        verdict = (
            "SOH HEAVY VS KPLER -- cargo-tag overcount on crossings, or Kpler missed "
            "origin loadings that still crossed the strait"
        )
    elif tanks_drew and kpler_vs_implied < 0.85:
        verdict = "KPLER LOADINGS LIGHT VS TANKS -- origin gap; SoH may still be OK"
    else:
        verdict = (
            "COUNT LOOKS CONSISTENT -- SoH, Kpler inside-Gulf loadings, and dI in the same band"
        )

    return {
        "verdict": verdict,
        "soh_mbbl": soh,
        "kpler_inside_mbbl": kpler_in,
        "implied_offtake_mbbl": implied,
        "inventory_change_mbbl": d_i,
        "soh_over_kpler": soh_vs_kpler,
        "kpler_over_implied": kpler_vs_implied,
        "s_recent_mbbl_d": last_s,
        "s_band_mbbl_d": last_band,
        "s_gap_mbbl_d": s_gap,
    }


def analysis_window(cfg: dict[str, Any], crossing_path: Path | None = None) -> tuple[date, date]:
    start = pd.Timestamp(cfg["start_date"]).date()
    if cfg.get("end_date"):
        end = pd.Timestamp(cfg["end_date"]).date()
        return start, end
    path = crossing_path or (BASE / cfg["paths"]["crossing"])
    raw = pd.read_excel(path, sheet_name="SoH Crossing", usecols=["Crossing Date"])
    end = pd.to_datetime(raw["Crossing Date"], dayfirst=True, errors="coerce").max().date()
    return start, end


def write_outputs(panel: pd.DataFrame, term_panels: dict[str, pd.DataFrame], sat_gaps: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT / "gulf_panel.csv")
    frames = []
    for tid, df in term_panels.items():
        x = df.copy()
        x["terminal_id"] = tid
        frames.append(x)
    if frames:
        pd.concat(frames).to_csv(OUT / "terminal_panels.csv")
    if sat_gaps is not None and not sat_gaps.empty:
        sat_gaps.to_csv(OUT / "sat_vs_kpler_occupancy.csv", index=False)
