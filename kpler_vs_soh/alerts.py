"""Diff this as_of vs the previous snapshot so the desk sees what changed, not a static list."""

from __future__ import annotations

from datetime import date

import pandas as pd

from kpler_vs_soh.config import STS_WATCH_FLAGS, Settings, WATCHLIST_BUCKETS


ALERT_COLUMNS = [
    "as_of",
    "imo",
    "vessel_name",
    "alert_type",
    "from_bucket",
    "to_bucket",
    "fleet",
    "origin",
    "volume_bbl",
    "days_since_load",
    "detail",
]


def diff_snapshots(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    *,
    as_of: date,
    settings: Settings,
) -> pd.DataFrame:
    empty = pd.DataFrame(columns=ALERT_COLUMNS)
    if current.empty:
        return empty
    if previous is None or previous.empty:
        return empty

    cur = current.set_index("imo")
    prev = previous.set_index("imo")
    rows = []

    for imo, row in cur.iterrows():
        if imo not in prev.index:
            if row.get("watchlist"):
                rows.append(
                    _alert(
                        as_of,
                        imo,
                        row,
                        "new_vessel",
                        None,
                        row.get("bucket"),
                        f"First seen in {row.get('bucket')}.",
                    )
                )
            continue
        old = prev.loc[imo]
        if isinstance(old, pd.DataFrame):
            old = old.iloc[-1]
        old_bucket = old.get("bucket")
        new_bucket = row.get("bucket")
        if old_bucket != new_bucket:
            kind = "resolved" if old_bucket in WATCHLIST_BUCKETS and new_bucket not in WATCHLIST_BUCKETS else "bucket_change"
            rows.append(
                _alert(
                    as_of,
                    imo,
                    row,
                    kind,
                    old_bucket,
                    new_bucket,
                    f"{old_bucket} → {new_bucket}. {row.get('reason')}",
                )
            )
        old_days = old.get("days_since_load")
        new_days = row.get("days_since_load")
        if (
            row.get("bucket") in {"inside_laden", "loaded_no_exit"}
            and _as_int(new_days) is not None
            and _as_int(old_days) is not None
            and _as_int(old_days) <= settings.alert_lag_days < _as_int(new_days)
        ):
            rows.append(
                _alert(
                    as_of,
                    imo,
                    row,
                    "stale_load",
                    old_bucket,
                    new_bucket,
                    f"Days since load crossed {settings.alert_lag_days} ({old_days} → {new_days}).",
                )
            )
        old_sts = _norm_flag(old.get("sts_flag"))
        new_sts = _norm_flag(row.get("sts_flag"))
        if old_sts != new_sts and (new_sts in STS_WATCH_FLAGS or old_sts in STS_WATCH_FLAGS):
            rows.append(
                _alert(
                    as_of,
                    imo,
                    row,
                    "sts_flag",
                    old_sts,
                    new_sts,
                    f"STS {old_sts or '—'} → {new_sts or '—'}"
                    + (f" ({row.get('sts_zone')}, {row.get('sts_counterparty')})" if new_sts else ""),
                )
            )

    for imo, old in prev.iterrows():
        if imo in cur.index:
            continue
        if old.get("watchlist"):
            rows.append(
                _alert(
                    as_of,
                    imo,
                    old,
                    "dropped",
                    old.get("bucket"),
                    None,
                    "Left the universe (no longer inside / no crude load or exit in window).",
                )
            )

    return pd.DataFrame(rows, columns=ALERT_COLUMNS)


def _as_int(value) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _norm_flag(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if text in {"", "nan", "<NA>", "None"}:
        return None
    return text


def _alert(as_of, imo, row, alert_type, from_bucket, to_bucket, detail) -> dict:
    return {
        "as_of": as_of,
        "imo": int(imo),
        "vessel_name": row.get("vessel_name"),
        "alert_type": alert_type,
        "from_bucket": from_bucket,
        "to_bucket": to_bucket,
        "fleet": row.get("fleet"),
        "origin": row.get("last_load_origin"),
        "volume_bbl": row.get("volume_bbl"),
        "days_since_load": row.get("days_since_load"),
        "detail": detail,
    }
