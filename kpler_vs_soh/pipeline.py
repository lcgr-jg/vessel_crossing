"""One-shot run: crossings + Kpler → vessel state snapshot → alerts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from kpler_vs_soh.alerts import diff_snapshots
from kpler_vs_soh.config import BUCKET_ORDER, Settings
from kpler_vs_soh.crossings import load_crossings
from kpler_vs_soh.kpler_pull import (
    build_loads,
    fetch_goo_port_calls,
    fetch_port_calls,
    fetch_ship_to_ships,
    fetch_trades,
    kpler_window,
)
from kpler_vs_soh.match import classify_all, pending_kpler
from kpler_vs_soh.store import export_latest, previous_state, replace_events, replace_snapshot
from kpler_vs_soh.sts import build_sts_events, enrich_state, sts_window


@dataclass
class RunResult:
    as_of: date
    kpler_start: date
    kpler_end: date
    crossings: pd.DataFrame
    loads: pd.DataFrame
    sts_events: pd.DataFrame
    state: pd.DataFrame
    alerts: pd.DataFrame
    pending: pd.DataFrame
    summary: pd.DataFrame
    export_paths: dict


def _summary(state: pd.DataFrame) -> pd.DataFrame:
    if state.empty:
        return pd.DataFrame()
    g = state.groupby("bucket", dropna=False)
    out = g.agg(
        vessels=("imo", "nunique"),
        dark=("fleet", lambda s: int((s == "dark fleet").sum())),
        mbbl=("volume_bbl", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).sum() / 1e6)),
        median_days_since_load=("days_since_load", "median"),
    ).reset_index()
    out["bucket"] = pd.Categorical(out["bucket"], categories=list(BUCKET_ORDER), ordered=True)
    return out.sort_values("bucket")


def run(settings: Settings | None = None, *, refresh_kpler: bool = False) -> RunResult:
    settings = settings or Settings()
    crossings = load_crossings(settings)
    crossing_max = crossings["crossing_ts"].max().date()
    as_of = settings.end_date or crossing_max
    kpler_start, kpler_end = kpler_window(settings, crossing_max)
    sts_start, sts_end = sts_window(settings, kpler_end)

    trades_raw = fetch_trades(settings, start=kpler_start, end=kpler_end, refresh=refresh_kpler)
    pc_raw = fetch_port_calls(settings, start=kpler_start, end=kpler_end, refresh=refresh_kpler)
    loads = build_loads(trades_raw, pc_raw, berth_match_days=settings.berth_match_days)

    sts_raw = fetch_ship_to_ships(settings, start=sts_start, end=sts_end, refresh=refresh_kpler)
    goo_pc = fetch_goo_port_calls(settings, start=sts_start, end=sts_end, refresh=refresh_kpler)
    sts_events = build_sts_events(
        sts_raw, goo_pc, pair_match_days=settings.sts_pair_match_days
    )

    state = classify_all(
        crossings,
        loads,
        settings=settings,
        as_of=as_of,
        start=settings.start_date,
    )
    state = enrich_state(
        state, sts_events, loads, settings=settings, as_of=as_of
    )
    pending = pending_kpler(loads, as_of)

    prev = previous_state(settings, as_of)
    alerts = diff_snapshots(state, prev, as_of=as_of, settings=settings)

    replace_events(settings, crossings=crossings, loads=loads, sts=sts_events)
    replace_snapshot(settings, as_of=as_of, state=state, alerts=alerts)
    paths = export_latest(settings, state, alerts)

    return RunResult(
        as_of=as_of,
        kpler_start=kpler_start,
        kpler_end=kpler_end,
        crossings=crossings,
        loads=loads,
        sts_events=sts_events,
        state=state,
        alerts=alerts,
        pending=pending,
        summary=_summary(state),
        export_paths=paths,
    )
