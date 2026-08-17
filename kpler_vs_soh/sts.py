"""GoO STS overlay on the load→strait hull book. Does not change `bucket`."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from kpler_vs_soh.config import STS_WATCH_FLAGS, Settings
from kpler_vs_soh.crossings import ofac_imo_sets
from kpler_vs_soh.util import normalize_imo, normalize_sts_zone, to_number


STS_STATE_COLS = [
    "sts_flag",
    "sts_role",
    "sts_zone",
    "sts_ts",
    "sts_lag_days",
    "sts_counterparty",
    "sts_counterparty_imo",
    "sts_volume_bbl",
]


def build_sts_events(
    sts_raw: pd.DataFrame,
    goo_pc_raw: pd.DataFrame,
    *,
    pair_match_days: int,
) -> pd.DataFrame:
    """Tidy IMO-level STS events: named pairs plus unmatched GoO PortCalls."""
    named = _named_events(sts_raw)
    unknown = _unknown_partner_events(goo_pc_raw, named, pair_match_days=pair_match_days)
    parts = [p for p in (named, unknown) if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame(
            columns=[
                "event_id",
                "source",
                "imo",
                "vessel_name",
                "role",
                "sts_ts",
                "sts_zone",
                "counterparty",
                "counterparty_imo",
                "volume_bbl",
            ]
        )
    return pd.concat(parts, ignore_index=True).sort_values(["imo", "sts_ts"]).reset_index(drop=True)


def _named_events(sts_raw: pd.DataFrame) -> pd.DataFrame:
    if sts_raw is None or sts_raw.empty:
        return pd.DataFrame()
    df = sts_raw.copy()
    df["sts_ts"] = pd.to_datetime(df["start"], errors="coerce")
    zone_col = "zone_name" if "zone_name" in df.columns else "_query_zone"
    df["sts_zone"] = df[zone_col].map(normalize_sts_zone)
    if "cargo_origin_barrels_split_by_product" in df.columns:
        df["volume_bbl"] = to_number(df["cargo_origin_barrels_split_by_product"])
    else:
        df["volume_bbl"] = pd.NA
    if "ship_to_ship_id" in df.columns:
        df["event_id"] = df["ship_to_ship_id"].astype(str)
    else:
        df["event_id"] = (
            df["load_vessel_imo"].astype(str)
            + "|"
            + df["discharge_vessel_imo"].astype(str)
            + "|"
            + df["sts_ts"].astype(str)
        )
    rows = []
    for _, row in df.iterrows():
        if pd.isna(row["sts_ts"]):
            continue
        giver_imo = pd.to_numeric(row.get("load_vessel_imo"), errors="coerce")
        recv_imo = pd.to_numeric(row.get("discharge_vessel_imo"), errors="coerce")
        if pd.notna(giver_imo):
            rows.append(
                {
                    "event_id": row["event_id"],
                    "source": "named",
                    "imo": int(giver_imo),
                    "vessel_name": row.get("load_vessel_name"),
                    "role": "giver",
                    "sts_ts": row["sts_ts"],
                    "sts_zone": row["sts_zone"],
                    "counterparty": row.get("discharge_vessel_name"),
                    "counterparty_imo": int(recv_imo) if pd.notna(recv_imo) else pd.NA,
                    "volume_bbl": row["volume_bbl"],
                }
            )
        if pd.notna(recv_imo):
            rows.append(
                {
                    "event_id": row["event_id"],
                    "source": "named",
                    "imo": int(recv_imo),
                    "vessel_name": row.get("discharge_vessel_name"),
                    "role": "receiver",
                    "sts_ts": row["sts_ts"],
                    "sts_zone": row["sts_zone"],
                    "counterparty": row.get("load_vessel_name"),
                    "counterparty_imo": int(giver_imo) if pd.notna(giver_imo) else pd.NA,
                    "volume_bbl": row["volume_bbl"],
                }
            )
    return pd.DataFrame(rows)


def _unknown_partner_events(
    goo_pc_raw: pd.DataFrame,
    named: pd.DataFrame,
    *,
    pair_match_days: int,
) -> pd.DataFrame:
    if goo_pc_raw is None or goo_pc_raw.empty:
        return pd.DataFrame()
    pc = goo_pc_raw.copy()
    if "is_sts" in pc.columns:
        pc["is_sts"] = pc["is_sts"].fillna(False).astype(bool)
        pc = pc.loc[pc["is_sts"]].copy()
    if pc.empty:
        return pd.DataFrame()
    pc["sts_ts"] = pd.to_datetime(pc["start"], errors="coerce")
    pc["imo"] = normalize_imo(pc["vessel_imo"])
    vol_col = "cargo_origin_barrels_split_by_product"
    pc["volume_bbl"] = to_number(pc[vol_col]) if vol_col in pc.columns else pd.NA
    zone_src = pc["zone_name"] if "zone_name" in pc.columns else pd.Series(pd.NA, index=pc.index)
    if "location_name" in pc.columns:
        zone_src = zone_src.fillna(pc["location_name"])
    pc["sts_zone"] = zone_src.map(normalize_sts_zone)
    pc = pc.dropna(subset=["imo", "sts_ts"])

    named_idx = named[["imo", "sts_ts"]] if not named.empty else pd.DataFrame(columns=["imo", "sts_ts"])
    window = pd.Timedelta(days=pair_match_days)
    unknown_rows = []
    for _, row in pc.iterrows():
        imo = int(row["imo"])
        ts = row["sts_ts"]
        if not named_idx.empty:
            hit = named_idx[
                (named_idx["imo"] == imo)
                & ((named_idx["sts_ts"] - ts).abs() <= window)
            ]
            if not hit.empty:
                continue
        pc_id = row.get("port_call_id")
        event_id = str(pc_id) if pd.notna(pc_id) else f"pc|{imo}|{ts}"
        unknown_rows.append(
            {
                "event_id": event_id,
                "source": "unknown_pc",
                "imo": imo,
                "vessel_name": row.get("vessel_name"),
                "role": "receiver",
                "sts_ts": ts,
                "sts_zone": row["sts_zone"],
                "counterparty": pd.NA,
                "counterparty_imo": pd.NA,
                "volume_bbl": row["volume_bbl"],
            }
        )
    return pd.DataFrame(unknown_rows)


def enrich_state(
    state: pd.DataFrame,
    sts_events: pd.DataFrame,
    loads: pd.DataFrame,
    *,
    settings: Settings,
    as_of: date,
) -> pd.DataFrame:
    """Attach sts_* columns. Watchlist ORs in STS_WATCH_FLAGS without changing bucket."""
    out = state.copy() if state is not None else pd.DataFrame()
    for col in STS_STATE_COLS:
        if col not in out.columns:
            out[col] = pd.NA

    as_of_ts = pd.Timestamp(as_of) + pd.Timedelta(hours=23, minutes=59)
    events = sts_events.copy() if sts_events is not None else pd.DataFrame()
    if not events.empty:
        events = events[events["sts_ts"] <= as_of_ts]

    if out.empty and events.empty:
        return out

    iran_imos, sdn_imos = ofac_imo_sets(settings)
    events_by: dict[int, pd.DataFrame] = {}
    if not events.empty:
        events_by = {int(imo): grp.sort_values("sts_ts") for imo, grp in events.groupby("imo")}

    loads_by: dict[int, pd.DataFrame] = {}
    if loads is not None and not loads.empty and "is_sts" in loads.columns:
        inside_sts = loads.loc[
            loads["is_sts"].fillna(False).astype(bool)
            & loads["origin_bucket"].eq("inside_gulf")
        ]
        if not inside_sts.empty:
            loads_by = {
                int(imo): grp.sort_values("load_ts")
                for imo, grp in inside_sts.dropna(subset=["vessel_imo"]).groupby("vessel_imo")
            }

    known = set(out["imo"].astype(int)) if not out.empty and "imo" in out.columns else set()
    extra_rows = []
    for imo in sorted(set(events_by) - known):
        stub = pd.Series({"imo": imo, "soh_pos": "never_seen", "last_cross_ts": pd.NaT, "watchlist": False})
        overlay = _flag_one(stub, events_by.get(imo), None)
        flag = overlay.get("sts_flag")
        if not (isinstance(flag, str) and flag in STS_WATCH_FLAGS):
            continue
        extra_rows.append(_sts_only_row(imo, events_by[imo], as_of, iran_imos, sdn_imos, overlay))
    if extra_rows:
        extra_df = pd.DataFrame(extra_rows)
        extra_df = extra_df.reindex(columns=list(dict.fromkeys(list(out.columns) + list(extra_df.columns))))
        aligned = out.reindex(columns=extra_df.columns)
        out = pd.concat([aligned, extra_df], ignore_index=True, sort=False)

    if out.empty:
        return out

    records = []
    for _, row in out.iterrows():
        imo = int(row["imo"])
        overlay = _flag_one(
            row,
            events_by.get(imo),
            loads_by.get(imo),
        )
        merged = row.to_dict()
        merged.update(overlay)
        bucket = merged.get("bucket")
        bucket_watch = bool(merged.get("watchlist"))
        flag = merged.get("sts_flag")
        sts_watch = isinstance(flag, str) and flag in STS_WATCH_FLAGS
        # East-coast / Red Sea loaders STS in Fujairah/Sohar is expected, not a missing-Hormuz cue.
        if isinstance(flag, str) and flag == "giver_no_soh" and bucket == "bypass":
            sts_watch = False
        merged["watchlist"] = bucket_watch or sts_watch
        records.append(merged)
    return pd.DataFrame(records)


def _flag_one(
    row: pd.Series,
    goo_events: pd.DataFrame | None,
    gulf_sts_loads: pd.DataFrame | None,
) -> dict:
    blank = {col: pd.NA for col in STS_STATE_COLS}
    last_cross = pd.to_datetime(row.get("last_cross_ts"), errors="coerce")
    pos = row.get("soh_pos")
    relevant = goo_events
    if relevant is not None and not relevant.empty and pd.notna(last_cross):
        # Only STS on/after the last SoH event belongs to this call.
        relevant = relevant[relevant["sts_ts"] >= last_cross]
    latest = None if relevant is None or relevant.empty else relevant.iloc[-1]

    if latest is not None:
        flag = _flag_from_goo(pos, latest)
        if flag is None:
            return blank
        lag = None
        if pd.notna(last_cross) and pos == "outside":
            lag = int((pd.Timestamp(latest["sts_ts"]).normalize() - last_cross.normalize()).days)
        return {
            "sts_flag": flag,
            "sts_role": latest.get("role"),
            "sts_zone": latest.get("sts_zone"),
            "sts_ts": latest.get("sts_ts"),
            "sts_lag_days": lag,
            "sts_counterparty": latest.get("counterparty"),
            "sts_counterparty_imo": latest.get("counterparty_imo"),
            "sts_volume_bbl": latest.get("volume_bbl"),
        }

    if pos == "inside" and gulf_sts_loads is not None and not gulf_sts_loads.empty:
        gulf = gulf_sts_loads
        if pd.notna(last_cross):
            gulf = gulf[gulf["load_ts"] >= last_cross]
        if not gulf.empty:
            g = gulf.iloc[-1]
            return {
                "sts_flag": "sts_inside",
                "sts_role": "giver",
                "sts_zone": g.get("origin_location_name"),
                "sts_ts": g.get("load_ts"),
                "sts_lag_days": pd.NA,
                "sts_counterparty": pd.NA,
                "sts_counterparty_imo": pd.NA,
                "sts_volume_bbl": g.get("volume_bbl"),
            }
    return blank


def _flag_from_goo(pos, latest: pd.Series) -> str | None:
    source = latest.get("source")
    role = latest.get("role")
    if source == "unknown_pc":
        return "unknown_partner"
    if role == "giver":
        if pos in {"inside", "never_seen"}:
            return "giver_no_soh"
        return "giver_after_exit"
    if pos == "outside":
        return "receiver_after_exit"
    # Named receiver in GoO with no SoH exit is not the dark-mother set.
    return None


def _sts_only_row(
    imo: int,
    events: pd.DataFrame,
    as_of: date,
    iran_imos: set[int],
    sdn_imos: set[int],
    overlay: dict,
) -> dict:
    """Hulls that STS in GoO but never made the load→strait universe."""
    latest = events.iloc[-1]
    sanctioned = imo in iran_imos or imo in sdn_imos
    flag = overlay.get("sts_flag")
    row = {
        "as_of": as_of,
        "imo": imo,
        "vessel_name": latest.get("vessel_name"),
        "bucket": "outside_other",
        "soh_pos": "never_seen",
        "reason": "GoO STS with no SoH tape and no inside-gulf Kpler load.",
        "watchlist": flag in STS_WATCH_FLAGS,
        "fleet": "dark fleet" if sanctioned else "clean fleet",
        "sanctioned_iran": imo in iran_imos,
        "sanctioned_sdn": imo in sdn_imos,
        "crude_relevant": True,
    }
    row.update(overlay)
    return row


def sts_window(settings: Settings, kpler_end: date) -> tuple[date, date]:
    start = settings.start_date
    end = kpler_end + timedelta(days=settings.sts_lookahead_days)
    return start, end
