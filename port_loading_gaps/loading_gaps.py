"""
Port loading-gap analysis — reusable helpers for any Kpler zone/installation.

Top panel: idle breaks between crude (or other) loadings from Port Calls berthing times.
Bottom panel: export flows (7d MA) from the Flows endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

LocationKind = Literal["zone", "installation"]


@dataclass(frozen=True)
class LocationSpec:
    """Kpler location filter — zone (port) or single installation."""

    name: str
    kind: LocationKind = "zone"

    @property
    def label(self) -> str:
        return self.name


@dataclass(frozen=True)
class LoadingGapConfig:
    """Parameters for one port / product loading-gap chart."""

    location: LocationSpec
    product: str = "crude"
    start_date: date = field(default_factory=lambda: date(2026, 2, 1))
    end_date: date = field(default_factory=lambda: date(2026, 8, 1))
    flow_unit: Literal["kbd", "kb"] = "kbd"
    with_forecast: bool = False
    only_realized_flows: bool = True
    ma_days: int = 7
    min_gap_hours: float = 1.0  # ignore sub-hour noise between adjacent calls


@dataclass(frozen=True)
class TimelineEvent:
    ts: pd.Timestamp
    delta: int


@dataclass(frozen=True)
class OccupancySegment:
    start: pd.Timestamp
    end: pd.Timestamp
    occupancy: int


@dataclass(frozen=True)
class IdleGap:
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600.0


@dataclass(frozen=True)
class BerthArrival:
    ts: pd.Timestamp
    occupancy_after: int


def parse_env(path: Path) -> dict[str, str]:
    """Load KEY=value .env without printing secrets."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def kpler_config(email: str, password: str):
    from kpler.sdk import Platform
    from kpler.sdk.configuration import Configuration

    return Configuration(Platform.Liquids, email, password)


def fetch_port_calls(config: LoadingGapConfig, *, email: str, password: str) -> pd.DataFrame:
    """Pull berthing intervals for loadings at the configured location."""
    from kpler.sdk.resources.port_calls import PortCalls

    client = PortCalls(kpler_config(email, password))
    kwargs: dict = {
        "products": [config.product],
        "start_date": config.start_date,
        "end_date": config.end_date,
        "with_forecast": config.with_forecast,
        "columns": [
            "vessel_name",
            "vessel_imo",
            "vessel_type",
            "installation_name",
            "start",
            "end",
            "closest_ancestor_group",
            #"closest_ancestor_product"
            "cargo_origin_barrels_split_by_product",
            "is_sts",
            "is_reexport",
            "multiple_forecasted_zones"
        ],
    }
    if config.location.kind == "zone":
        kwargs["zones"] = [config.location.name]
    else:
        kwargs["installations"] = [config.location.name]

    with pd.option_context("mode.chained_assignment", None):
        df = client.get(**kwargs)

    if df is None or df.empty:
        return pd.DataFrame(
            columns=["vessel_name", "installation_name", "start", "end"]
        )

    df = df.copy()
    for col in ("start", "end"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Realized loadings only — berthing timestamps required for occupancy math.
    df = df.dropna(subset=["start", "end"])
    df = df[df["end"] > df["start"]]
    return df.sort_values("start").reset_index(drop=True)


def fetch_export_flows(config: LoadingGapConfig, *, email: str, password: str) -> pd.Series:
    """Daily export series for the configured location and product."""
    from kpler.sdk import FlowsDirection, FlowsMeasurementUnit, FlowsPeriod, FlowsSplit
    from kpler.sdk.resources.flows import Flows

    unit_map = {
        "kbd": FlowsMeasurementUnit.KBD,
        "kb": FlowsMeasurementUnit.KB,
    }
    client = Flows(kpler_config(email, password))
    kwargs: dict = {
        "flow_direction": [FlowsDirection.Export],
        "split": [FlowsSplit.Total],
        "granularity": [FlowsPeriod.Daily],
        "unit": [unit_map[config.flow_unit]],
        "products": [config.product],
        "start_date": config.start_date,
        "end_date": config.end_date,
        "with_forecast": config.with_forecast,
        "only_realized": config.only_realized_flows,
    }
    if config.location.kind == "zone":
        kwargs["from_zones"] = [config.location.name]
    else:
        kwargs["from_installations"] = [config.location.name]

    df = client.get(**kwargs)
    if df is None or df.empty:
        return pd.Series(dtype=float)

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    series = out.set_index("Date")["Total"].sort_index()
    series.index = series.index.normalize()
    return series


def _as_ts(value: date | datetime | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value)


def occupancy_at(intervals: pd.DataFrame, ts: pd.Timestamp) -> int:
    """Count vessels at berth at instant ts (start inclusive, end exclusive)."""
    if intervals.empty:
        return 0
    mask = (intervals["start"] <= ts) & (intervals["end"] > ts)
    return int(mask.sum())


def build_occupancy_segments(
    intervals: pd.DataFrame,
    *,
    t_min: pd.Timestamp,
    t_max: pd.Timestamp,
) -> list[OccupancySegment]:
    """Piecewise-constant occupancy from port-call start/end events."""
    events: list[TimelineEvent] = []
    for start, end in intervals[["start", "end"]].itertuples(index=False, name=None):
        events.append(TimelineEvent(pd.Timestamp(start), 1))
        events.append(TimelineEvent(pd.Timestamp(end), -1))

    events.sort(key=lambda e: (e.ts, -e.delta))
    occupancy = 0
    cursor = t_min
    segments: list[OccupancySegment] = []

    for event in events:
        if event.ts > t_max:
            break
        if event.ts > cursor:
            segments.append(OccupancySegment(cursor, event.ts, occupancy))
        occupancy += event.delta
        cursor = event.ts

    if cursor < t_max:
        segments.append(OccupancySegment(cursor, t_max, occupancy))

    return [s for s in segments if s.end > s.start]


def idle_gaps_from_segments(
    segments: list[OccupancySegment],
    *,
    min_gap_hours: float,
) -> list[IdleGap]:
    gaps: list[IdleGap] = []
    for seg in segments:
        if seg.occupancy != 0:
            continue
        gap = IdleGap(seg.start, seg.end)
        if gap.hours >= min_gap_hours:
            gaps.append(gap)
    return gaps


def daily_idle_hours(
    gaps: list[IdleGap],
    *,
    start: date,
    end: date,
) -> pd.Series:
    """Hours per calendar day with zero vessels at berth."""
    idx = pd.date_range(_as_ts(start), _as_ts(end), freq="D")
    hours = pd.Series(0.0, index=idx)

    for gap in gaps:
        day = gap.start.normalize()
        last_day = gap.end.normalize()
        while day <= last_day and day <= idx.max():
            if day >= idx.min():
                day_end = day + pd.Timedelta(days=1)
                overlap_start = max(gap.start, day)
                overlap_end = min(gap.end, day_end)
                if overlap_end > overlap_start:
                    hours.loc[day] += (overlap_end - overlap_start).total_seconds() / 3600.0
            day += pd.Timedelta(days=1)

    return hours


def berth_arrivals(intervals: pd.DataFrame) -> list[BerthArrival]:
    """One marker per loading start; occupancy counts the arriving vessel."""
    arrivals: list[BerthArrival] = []
    for start in intervals["start"]:
        ts = pd.Timestamp(start)
        occ = occupancy_at(intervals, ts)
        arrivals.append(BerthArrival(ts, occ))
    return arrivals


# --- Imagery vs Kpler port-call comparison -----------------------------------

# Default mapping for copernicus_vessel_detection.ipynb AOIs.
# Installations for tight berths; UAE SPMs use zone-level Kpler (one zone may cover several SPMs).
DEFAULT_AOI_KPLER_LOCATIONS: dict[str, LocationSpec] = {
    "yanbu_north_crude_terminal": LocationSpec("Yanbu Crude", kind="installation"),
    "muajjiz": LocationSpec("Muajjiz", kind="installation"),
    "zirku_spm_1": LocationSpec("Zirku", kind="zone"),
    "zirku_spm_2": LocationSpec("Zirku", kind="zone"),
    "das_spm_1": LocationSpec("Das Island", kind="zone"),
    "das_spm_2": LocationSpec("Das Island", kind="zone"),
    "um_lulu_spm": LocationSpec("Umm Lulu", kind="installation"),
    "cpc": LocationSpec("CPC Terminal", kind="installation"),
    # "cpc": LocationSpec("CPC", kind="installation"),  # fill name from Kpler port_calls
}


def aggregate_imagery_by_date(df_imagery: pd.DataFrame) -> pd.DataFrame:
    """Collapse long-format detection rows to one row per (aoi, date)."""
    if df_imagery.empty:
        return pd.DataFrame(
            columns=["aoi", "date", "imagery_count", "sensor"]
        )
    df = df_imagery.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return (
        df.groupby(["aoi", "date"], as_index=False)
        .agg(imagery_count=("vessel_count", "max"), sensor=("sensor", "first"))
        .sort_values(["aoi", "date"])
        .reset_index(drop=True)
    )


def fetch_port_calls_for_aois(
    aoi_locations: dict[str, LocationSpec],
    *,
    start_date: date,
    end_date: date,
    product: str = "crude",
    email: str,
    password: str,
) -> dict[str, pd.DataFrame]:
    """
    Pull Kpler port calls for each imagery AOI.

    Deduplicates API calls when multiple AOIs share the same Kpler location
    (not expected today, but safe for extensions).
    """
    unique: dict[tuple[str, str], LocationSpec] = {}
    for loc in aoi_locations.values():
        unique[(loc.name, loc.kind)] = loc

    calls_by_key: dict[tuple[str, str], pd.DataFrame] = {}
    for key, loc in unique.items():
        cfg = LoadingGapConfig(
            location=loc,
            product=product,
            start_date=start_date,
            end_date=end_date,
        )
        calls_by_key[key] = fetch_port_calls(cfg, email=email, password=password)

    return {
        aoi: calls_by_key[(loc.name, loc.kind)].copy()
        for aoi, loc in aoi_locations.items()
    }


def vessels_at_berth(port_calls: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    """Port calls active at instant ts (start inclusive, end exclusive)."""
    if port_calls.empty:
        return port_calls.iloc[0:0]
    mask = (port_calls["start"] <= ts) & (port_calls["end"] > ts)
    return port_calls.loc[mask]


def _comparison_flag(imagery_count: int, kpler_count: int, count_tolerance: int) -> str:
    if abs(imagery_count - kpler_count) <= count_tolerance:
        return "agree"
    if imagery_count > kpler_count:
        return "imagery_high"  # false positives and/or dark loading if kpler_count == 0
    return "imagery_low"  # missed detection and/or S2 timing vs berthing window


def compare_imagery_with_kpler(
    df_imagery: pd.DataFrame,
    port_calls_by_aoi: dict[str, pd.DataFrame],
    *,
    aoi_locations: dict[str, LocationSpec] | None = None,
    count_tolerance: int = 0,
    scene_hour_utc: int = 12,
) -> pd.DataFrame:
    """
    Compare Sentinel occupancy (berth-slot / blob count) vs Kpler berthing intervals.

    Returns one row per (aoi, imagery scene date) with imagery_count, kpler_count,
    delta, and a coarse flag. Kpler occupancy uses port-call start/end at scene_hour_utc
    on each scene date (default noon UTC — S2 pass is usually mid-morning local).
    """
    aoi_locations = aoi_locations or DEFAULT_AOI_KPLER_LOCATIONS
    imagery = aggregate_imagery_by_date(df_imagery)

    rows: list[dict] = []
    for _, row in imagery.iterrows():
        aoi = row["aoi"]
        if aoi not in port_calls_by_aoi:
            continue
        loc = aoi_locations.get(aoi)
        ts = pd.Timestamp(row["date"]) + pd.Timedelta(hours=scene_hour_utc)
        calls = port_calls_by_aoi[aoi]
        kpler_count = occupancy_at(calls, ts)
        imagery_count = int(row["imagery_count"])
        active = vessels_at_berth(calls, ts)
        vessel_names = (
            active["vessel_name"].astype(str).tolist()
            if "vessel_name" in active.columns
            else []
        )

        rows.append(
            {
                "aoi": aoi,
                "date": row["date"],
                "imagery_count": imagery_count,
                "kpler_count": kpler_count,
                "delta": imagery_count - kpler_count,
                "match": abs(imagery_count - kpler_count) <= count_tolerance,
                "flag": _comparison_flag(imagery_count, kpler_count, count_tolerance),
                "kpler_location": loc.label if loc else None,
                "kpler_location_kind": loc.kind if loc else None,
                "kpler_vessels": "; ".join(vessel_names),
                "sensor": row["sensor"],
            }
        )

    return pd.DataFrame(rows)


def summarize_imagery_kpler_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    """Per-AOI agreement and error-direction counts."""
    if comparison.empty:
        return pd.DataFrame()

    def _summarize(group: pd.DataFrame) -> pd.Series:
        n = len(group)
        return pd.Series(
            {
                "scene_dates": n,
                "exact_match_pct": round(100 * group["match"].mean(), 1),
                "mean_abs_delta": round(group["delta"].abs().mean(), 2),
                "imagery_high_days": int((group["flag"] == "imagery_high").sum()),
                "imagery_low_days": int((group["flag"] == "imagery_low").sum()),
                "agree_days": int((group["flag"] == "agree").sum()),
            }
        )

    return comparison.groupby("aoi").apply(_summarize, include_groups=False).reset_index()


def plot_imagery_kpler_comparison(
    comparison: pd.DataFrame,
    *,
    aoi: str | None = None,
    figsize: tuple[float, float] = (12, 4),
) -> plt.Figure:
    """Side-by-side occupancy counts per scene date."""
    df = comparison.copy()
    if aoi is not None:
        df = df[df["aoi"] == aoi]
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("No comparison rows to plot")
        return fig

    title_aoi = aoi or "all AOIs"
    fig, ax = plt.subplots(figsize=figsize)
    for name, grp in df.groupby("aoi"):
        ax.plot(
            grp["date"],
            grp["imagery_count"],
            marker="o",
            ls="-",
            label=f"{name} imagery",
        )
        ax.plot(
            grp["date"],
            grp["kpler_count"],
            marker="s",
            ls="--",
            alpha=0.85,
            label=f"{name} Kpler",
        )

    ax.set_ylabel("vessels at berth")
    ax.set_title(f"Imagery vs Kpler occupancy — {title_aoi}")
    ax.legend(fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    return fig


def analyze_loading_gaps(
    port_calls: pd.DataFrame,
    config: LoadingGapConfig,
) -> dict:
    """Derive gaps, daily idle hours, and arrival markers from port calls."""
    t_min = _as_ts(config.start_date)
    t_max = _as_ts(config.end_date) + pd.Timedelta(days=1)

    if port_calls.empty:
        empty_daily = daily_idle_hours([], start=config.start_date, end=config.end_date)
        return {
            "segments": [],
            "gaps": [],
            "daily_idle_hours": empty_daily,
            "daily_idle_ma": empty_daily.rolling(config.ma_days, min_periods=1).mean(),
            "arrivals": [],
        }

    intervals = port_calls[["start", "end"]].copy()
    segments = build_occupancy_segments(intervals, t_min=t_min, t_max=t_max)
    gaps = idle_gaps_from_segments(segments, min_gap_hours=config.min_gap_hours)
    daily_idle = daily_idle_hours(gaps, start=config.start_date, end=config.end_date)
    daily_idle_ma = daily_idle.rolling(config.ma_days, min_periods=1).mean()

    return {
        "segments": segments,
        "gaps": gaps,
        "daily_idle_hours": daily_idle,
        "daily_idle_ma": daily_idle_ma,
        "arrivals": berth_arrivals(intervals),
    }


def plot_loading_gaps(
    config: LoadingGapConfig,
    analysis: dict,
    export_flows: pd.Series,
    *,
    events: list[tuple[date, str]] | None = None,
    figsize: tuple[float, float] = (14, 8),
) -> plt.Figure:
    """Two-panel chart matching the Ras Laffan loading-break style."""
    loc = config.location.label
    product = config.product
    ma = config.ma_days
    flow_label = config.flow_unit

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.08},
    )

    t_min = _as_ts(config.start_date)
    t_max = _as_ts(config.end_date) + pd.Timedelta(hours=12)

    # Active periods (>=1 vessel at berth).
    for seg in analysis["segments"]:
        if seg.occupancy >= 1:
            ax_top.axvspan(seg.start, seg.end, color="#cfe8cf", alpha=0.55, lw=0)

    # Idle gaps as red bars anchored at gap start.
    gap_hours = [g.hours for g in analysis["gaps"]]
    gap_starts = [g.start for g in analysis["gaps"]]
    if gap_hours:
        ax_top.bar(
            gap_starts,
            gap_hours,
            width=0.9,
            align="edge",
            color="#d62728",
            alpha=0.85,
            label="Break (all berths empty)",
        )

    ax_top.set_ylabel("hours with all berths empty")
    ax_top.set_title(f"{loc}: Breaks Between {product.title()} Loadings")
    ax_top.set_xlim(t_min, t_max)
    ax_top.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # 7d MA of daily idle hours on right axis.
    ax_top_r = ax_top.twinx()
    idle_ma = analysis["daily_idle_ma"]
    if not idle_ma.empty:
        ax_top_r.plot(
            idle_ma.index,
            idle_ma.values,
            color="#9467bd",
            lw=2.0,
            label=f"hours/day empty ({ma}d MA)",
        )
    ax_top_r.set_ylabel(f"hours/day empty ({ma}d MA)")
    ax_top_r.set_ylim(0, 24)
    ax_top_r.spines["top"].set_visible(False)

    # Arrival markers along the baseline.
    y_marker = ax_top.get_ylim()[0]
    for arr in analysis["arrivals"]:
        color = "#1f4e79" if arr.occupancy_after >= 2 else "#bdbdbd"
        ax_top.scatter(
            [arr.ts],
            [y_marker],
            marker="^",
            s=28 if arr.occupancy_after >= 2 else 18,
            color=color,
            zorder=5,
            clip_on=False,
        )

    legend_handles = [
        Patch(facecolor="#d62728", alpha=0.85, label="Break (all berths empty)"),
        Patch(facecolor="#cfe8cf", alpha=0.55, label=">=1 vessel at berth"),
        plt.Line2D([0], [0], color="#9467bd", lw=2, label=f"hours/day empty ({ma}d MA)"),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="#bdbdbd",
            markersize=7,
            label="Berth arrival",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="#1f4e79",
            markersize=8,
            label="Arrival with >=2 vessels at berth",
        ),
    ]
    ax_top.legend(handles=legend_handles, loc="upper left", fontsize=8, frameon=True)

    # Bottom panel — export flows.
    if not export_flows.empty:
        export_ma = export_flows.rolling(ma, min_periods=1).mean()
        ax_bot.plot(export_ma.index, export_ma.values, color="#1f77b4", lw=2)
    ax_bot.set_ylabel(flow_label)
    ax_bot.set_title(f"{loc} {product.title()} Exports ({ma}d moving avg)")
    ax_bot.set_xlim(t_min, t_max)

    if events:
        for ax in (ax_top, ax_bot):
            for evt_date, label in events:
                ax.axvline(_as_ts(evt_date), color="0.35", ls="--", lw=0.9, alpha=0.8)
            # Label only on top panel to avoid clutter.
        for evt_date, label in events:
            ax_top.text(
                _as_ts(evt_date) + pd.Timedelta(hours=8),
                ax_top.get_ylim()[1] * 0.97,
                label,
                rotation=90,
                va="top",
                ha="left",
                fontsize=7,
                color="0.25",
            )

    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax_bot.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.subplots_adjust(left=0.08, right=0.92, top=0.93, bottom=0.08)
    return fig
