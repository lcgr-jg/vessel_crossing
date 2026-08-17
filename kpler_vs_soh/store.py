"""DuckDB event store + as_of snapshots. Cache parquet stays out of git; this is the history."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from kpler_vs_soh.config import Settings


def connect(settings: Settings) -> duckdb.DuckDBPyConnection:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(settings.db_path))


def _replace_table(con: duckdb.DuckDBPyConnection, name: str, df: pd.DataFrame) -> None:
    payload = df if df is not None else pd.DataFrame()
    # DuckDB refuses a 0-column frame; keep an as_of stub so history tables always exist.
    if payload.empty and payload.shape[1] == 0:
        payload = pd.DataFrame({"as_of": pd.Series(dtype="datetime64[ns]")})
    con.register("_tmp_df", payload)
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _tmp_df")
    con.unregister("_tmp_df")


def replace_events(
    settings: Settings,
    *,
    crossings: pd.DataFrame,
    loads: pd.DataFrame,
    sts: pd.DataFrame | None = None,
) -> None:
    con = connect(settings)
    try:
        _replace_table(con, "crossings", crossings)
        _replace_table(con, "loads", loads)
        if sts is not None:
            _replace_table(con, "sts_events", sts)
    finally:
        con.close()


def _table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    n = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [name],
    ).fetchone()[0]
    return n > 0


def replace_snapshot(
    settings: Settings,
    *,
    as_of: date,
    state: pd.DataFrame,
    alerts: pd.DataFrame,
) -> None:
    """Keep history by concatenating; re-running the same as_of replaces that day only."""
    con = connect(settings)
    try:
        state_hist = pd.DataFrame()
        alert_hist = pd.DataFrame()
        if _table_exists(con, "vessel_state"):
            state_hist = con.execute("SELECT * FROM vessel_state").df()
            if not state_hist.empty and "as_of" in state_hist.columns:
                as_of_s = pd.to_datetime(state_hist["as_of"]).dt.date
                state_hist = state_hist.loc[as_of_s != as_of]
        if _table_exists(con, "alerts"):
            alert_hist = con.execute("SELECT * FROM alerts").df()
            if not alert_hist.empty and "as_of" in alert_hist.columns:
                as_of_a = pd.to_datetime(alert_hist["as_of"]).dt.date
                alert_hist = alert_hist.loc[as_of_a != as_of]

        combined_state = pd.concat([state_hist, state], ignore_index=True) if not state.empty else state_hist
        combined_alerts = pd.concat([alert_hist, alerts], ignore_index=True) if not alerts.empty else alert_hist
        _replace_table(con, "vessel_state", combined_state)
        _replace_table(con, "alerts", combined_alerts)
    finally:
        con.close()


def previous_state(settings: Settings, as_of: date) -> pd.DataFrame:
    if not settings.db_path.exists():
        return pd.DataFrame()
    con = connect(settings)
    try:
        if not _table_exists(con, "vessel_state"):
            return pd.DataFrame()
        n = con.execute("SELECT COUNT(*) FROM vessel_state").fetchone()[0]
        if n == 0:
            return pd.DataFrame()
        return con.execute(
            """
            SELECT * FROM vessel_state
            WHERE CAST(as_of AS DATE) = (
                SELECT MAX(CAST(as_of AS DATE)) FROM vessel_state
                WHERE CAST(as_of AS DATE) < ?
            )
            """,
            [as_of],
        ).df()
    finally:
        con.close()


def export_latest(settings: Settings, state: pd.DataFrame, alerts: pd.DataFrame) -> dict[str, Path]:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "state": settings.output_dir / "vessel_state_latest.csv",
        "alerts": settings.output_dir / "alerts_latest.csv",
        "watchlist": settings.output_dir / "watchlist_latest.csv",
        "sts": settings.output_dir / "sts_overlay_latest.csv",
    }
    state.to_csv(paths["state"], index=False)
    alerts.to_csv(paths["alerts"], index=False)
    watch = state[state["watchlist"]] if not state.empty and "watchlist" in state.columns else state
    watch.to_csv(paths["watchlist"], index=False)
    if not state.empty and "sts_flag" in state.columns:
        sts_view = state[state["sts_flag"].notna() & ~state["sts_flag"].astype(str).isin(["", "nan", "<NA>"])]
        sts_view.to_csv(paths["sts"], index=False)
    else:
        pd.DataFrame().to_csv(paths["sts"], index=False)
    return paths
