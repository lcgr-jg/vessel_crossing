"""Per-IMO state: last SoH event vs last inside-gulf Kpler crude load."""

from __future__ import annotations

from datetime import date

import pandas as pd

from kpler_vs_soh.config import (
    CRUDE_CAPABLE_CLASSES,
    LOADING_STATUSES,
    Settings,
)
from kpler_vs_soh.crossings import ofac_imo_sets


CRUDE_EXIT_GROUPS = {"crude", "likely_crude"}


def _as_ts(value) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value)


def _days(later: pd.Timestamp, earlier: pd.Timestamp | None) -> int | None:
    if earlier is None:
        return None
    return int((later.normalize() - earlier.normalize()).days)


def _last_row(df: pd.DataFrame) -> pd.Series | None:
    if df is None or df.empty:
        return None
    return df.iloc[-1]


def classify_one(
    *,
    imo: int,
    as_of: pd.Timestamp,
    settings: Settings,
    crosses: pd.DataFrame,
    loads: pd.DataFrame,
    iran_imos: set[int],
    sdn_imos: set[int],
) -> dict | None:
    """One hull, one row.

    Decision order (why this sequence):
    1. Bypass-only loads with no SoH tape — not a Hormuz puzzle.
    2. Last event is an exit — then a later Kpler load is a missing inbound crossing.
    3. Laden crude exit with/without a prior Kpler load — accounted vs dark loading.
    4. Still inside (or never seen) — loading / steaming / stale / intra-gulf.
    """
    last_cross = _last_row(crosses)
    inside_loads = loads[loads["origin_bucket"] == "inside_gulf"] if not loads.empty else loads
    last_inside = _last_row(inside_loads)
    last_any = _last_row(loads)

    if last_cross is not None:
        direction = last_cross["Direction"]
        pos = "inside" if direction == "Entered MEG" else "outside"
    else:
        direction = None
        pos = "never_seen"

    # Nothing to say unless Kpler loaded inside the gulf or the hull is on the SoH tape.
    if last_cross is None and last_inside is None:
        if last_any is not None and str(last_any["origin_bucket"]).startswith("bypass"):
            return _pack(
                imo=imo,
                as_of=as_of,
                bucket="bypass",
                pos=pos,
                last_cross=last_cross,
                last_load=last_any,
                settings=settings,
                iran_imos=iran_imos,
                sdn_imos=sdn_imos,
                reason=f"Kpler load at {last_any['origin_location_name']} is east/Red Sea — no SoH exit expected.",
            )
        return None

    lookback = pd.Timedelta(days=settings.load_lookback_days)

    if pos == "outside":
        exit_ts = _as_ts(last_cross["crossing_ts"])
        load_after = (
            inside_loads[inside_loads["load_ts"] > exit_ts]
            if not inside_loads.empty
            else inside_loads
        )
        last_after = _last_row(load_after)
        if last_after is not None:
            return _pack(
                imo=imo,
                as_of=as_of,
                bucket="load_without_reentry",
                pos=pos,
                last_cross=last_cross,
                last_load=last_after,
                settings=settings,
                iran_imos=iran_imos,
                sdn_imos=sdn_imos,
                reason=(
                    f"Last SoH event was exit on {exit_ts.date()}; Kpler then shows an inside-gulf "
                    f"load at {last_after['origin_location_name']} on {pd.Timestamp(last_after['load_ts']).date()} "
                    "with no re-entry in the crossing file."
                ),
            )

        match = _load_for_exit(crosses, inside_loads, exit_ts, lookback)
        cargo = last_cross.get("cargo_group")
        loading_state = last_cross.get("Loading State")
        if loading_state == "Laden" and cargo in CRUDE_EXIT_GROUPS:
            if match is not None:
                return _pack(
                    imo=imo,
                    as_of=as_of,
                    bucket="accounted",
                    pos=pos,
                    last_cross=last_cross,
                    last_load=match,
                    settings=settings,
                    iran_imos=iran_imos,
                    sdn_imos=sdn_imos,
                    reason=(
                        f"Kpler loaded at {match['origin_location_name']} on "
                        f"{pd.Timestamp(match['load_ts']).date()} then SoH laden exit on {exit_ts.date()}."
                    ),
                )
            return _pack(
                imo=imo,
                as_of=as_of,
                bucket="exit_no_load",
                pos=pos,
                last_cross=last_cross,
                last_load=last_inside,
                settings=settings,
                iran_imos=iran_imos,
                sdn_imos=sdn_imos,
                reason=(
                    f"SoH laden crude exit on {exit_ts.date()} with no matching inside-gulf "
                    f"Kpler load in the {settings.load_lookback_days}d before the crossing."
                ),
            )
        if loading_state == "Ballast" and match is not None:
            return _pack(
                imo=imo,
                as_of=as_of,
                bucket="conflict",
                pos=pos,
                last_cross=last_cross,
                last_load=match,
                settings=settings,
                iran_imos=iran_imos,
                sdn_imos=sdn_imos,
                reason=(
                    f"Kpler loaded at {match['origin_location_name']} on "
                    f"{pd.Timestamp(match['load_ts']).date()} but SoH recorded a ballast exit on {exit_ts.date()}."
                ),
            )
        return _pack(
            imo=imo,
            as_of=as_of,
            bucket="outside_other",
            pos=pos,
            last_cross=last_cross,
            last_load=match if match is not None else last_any,
            settings=settings,
            iran_imos=iran_imos,
            sdn_imos=sdn_imos,
            reason=f"Last SoH event is {loading_state or 'unknown'} exit on {exit_ts.date()} — not a crude-load puzzle.",
        )

    # inside or never_seen (never_seen with an inside load is treated as assumed-inside).
    enter_ts = _as_ts(last_cross["crossing_ts"]) if last_cross is not None else None
    if enter_ts is not None and not inside_loads.empty:
        since_enter = inside_loads[inside_loads["load_ts"] >= enter_ts]
    else:
        since_enter = inside_loads
    last_since = _last_row(since_enter)

    if last_since is None:
        entered_laden = (
            last_cross is not None
            and last_cross.get("Loading State") == "Laden"
            and last_cross.get("cargo_group") in CRUDE_EXIT_GROUPS
        )
        bucket = "inside_laden" if entered_laden else "inside_ballast"
        why = (
            f"Last SoH event is entry on {enter_ts.date()} "
            + ("already laden; no later Kpler crude load." if entered_laden else "ballast/unknown; no Kpler crude load yet.")
            if enter_ts is not None
            else "No SoH history and no inside-gulf Kpler load."
        )
        return _pack(
            imo=imo,
            as_of=as_of,
            bucket=bucket,
            pos=pos,
            last_cross=last_cross,
            last_load=last_any,
            settings=settings,
            iran_imos=iran_imos,
            sdn_imos=sdn_imos,
            reason=why,
        )

    status = str(last_since.get("status_norm") or "").strip()
    # Fixtures with no SoH tape are too weak to treat as dark activity.
    if pos == "never_seen" and status == "scheduled":
        return None
    dest = str(last_since.get("dest_bucket") or "")
    load_end = _as_ts(last_since.get("load_end_ts")) or _as_ts(last_since.get("load_ts"))
    days_since = _days(as_of, load_end)
    origin = last_since.get("origin_location_name")

    if status in LOADING_STATUSES:
        return _pack(
            imo=imo,
            as_of=as_of,
            bucket="inside_loading",
            pos=pos,
            last_cross=last_cross,
            last_load=last_since,
            settings=settings,
            iran_imos=iran_imos,
            sdn_imos=sdn_imos,
            reason=f"Kpler status {last_since.get('status')} at {origin}; last SoH event is still inside.",
        )

    if dest == "inside_gulf":
        return _pack(
            imo=imo,
            as_of=as_of,
            bucket="intra_gulf",
            pos=pos,
            last_cross=last_cross,
            last_load=last_since,
            settings=settings,
            iran_imos=iran_imos,
            sdn_imos=sdn_imos,
            reason=f"Kpler dest looks intra-gulf ({last_since.get('destination_location_name')}); no SoH exit required.",
        )

    stale = days_since is not None and days_since > settings.alert_lag_days
    bucket = "loaded_no_exit" if stale else "inside_laden"
    lag_note = f"{days_since}d since load end" if days_since is not None else "unknown lag"
    why = (
        f"Kpler loaded at {origin} on {load_end.date() if load_end is not None else '?'}"
        f" ({last_since.get('status')}); no SoH exit after that ({lag_note})."
    )
    if pos == "never_seen":
        why = "Never in the SoH crossing file. " + why
    return _pack(
        imo=imo,
        as_of=as_of,
        bucket=bucket,
        pos=pos,
        last_cross=last_cross,
        last_load=last_since,
        settings=settings,
        iran_imos=iran_imos,
        sdn_imos=sdn_imos,
        reason=why,
    )


def _load_for_exit(
    crosses: pd.DataFrame,
    inside_loads: pd.DataFrame,
    exit_ts: pd.Timestamp,
    lookback: pd.Timedelta,
) -> pd.Series | None:
    """Match the load from this MEG call, not a 21-day proximity guess.

    VLCCs can sit laden for weeks. The cargo that belongs to an exit is the last
    inside-gulf load after the most recent entry and before the exit.
    """
    if inside_loads is None or inside_loads.empty:
        return None
    if crosses is not None and not crosses.empty:
        entries = crosses[
            (crosses["Direction"] == "Entered MEG") & (crosses["crossing_ts"] < exit_ts)
        ]
        last_entry = _last_row(entries.sort_values("crossing_ts") if not entries.empty else entries)
        if last_entry is not None:
            start = _as_ts(last_entry["crossing_ts"])
            # Crossing dates are often midnight; allow a load on the day before the recorded entry.
            window = inside_loads[
                (inside_loads["load_ts"] > start - pd.Timedelta(days=1))
                & (inside_loads["load_ts"] <= exit_ts)
            ]
            return _last_row(window)
    return _load_before(inside_loads, exit_ts, lookback)


def _load_before(inside_loads: pd.DataFrame, exit_ts: pd.Timestamp, lookback: pd.Timedelta) -> pd.Series | None:
    if inside_loads is None or inside_loads.empty:
        return None
    window = inside_loads[
        (inside_loads["load_ts"] <= exit_ts) & (inside_loads["load_ts"] >= exit_ts - lookback)
    ]
    return _last_row(window)


def _pack(
    *,
    imo: int,
    as_of: pd.Timestamp,
    bucket: str,
    pos: str,
    last_cross: pd.Series | None,
    last_load: pd.Series | None,
    settings: Settings,
    iran_imos: set[int],
    sdn_imos: set[int],
    reason: str,
) -> dict:
    sanctioned_iran = imo in iran_imos
    sanctioned_sdn = imo in sdn_imos
    if last_cross is not None:
        fleet = last_cross.get("fleet")
        vessel = last_cross.get("Vessel Name")
        afra = last_cross.get("afra_class")
        loading_state = last_cross.get("Loading State")
        cargo_group = last_cross.get("cargo_group")
        crossing_type = last_cross.get("Crossing Type")
        deadweight = last_cross.get("Deadweight")
        volume_soh = last_cross.get("volume_bbl")
        last_cross_ts = _as_ts(last_cross.get("crossing_ts"))
        last_cross_dir = last_cross.get("Direction")
        is_dark_route = bool(last_cross.get("is_dark_route"))
    else:
        fleet = "dark fleet" if (sanctioned_iran or sanctioned_sdn) else "clean fleet"
        vessel = last_load.get("vessel_name") if last_load is not None else None
        afra = None
        loading_state = None
        cargo_group = None
        crossing_type = None
        deadweight = None
        volume_soh = None
        last_cross_ts = None
        last_cross_dir = None
        is_dark_route = False

    if last_load is not None and (vessel is None or (last_cross is not None and last_load is not None)):
        # Prefer the name on the more recent event.
        load_ts = _as_ts(last_load.get("load_ts"))
        if last_cross_ts is None or (load_ts is not None and load_ts >= last_cross_ts):
            vessel = last_load.get("vessel_name") or vessel

    load_ts = _as_ts(last_load.get("load_ts")) if last_load is not None else None
    load_end = _as_ts(last_load.get("load_end_ts")) if last_load is not None else None
    days_inside = _days(as_of, last_cross_ts) if pos == "inside" and last_cross_ts is not None else None
    days_since_load = _days(as_of, load_end or load_ts)
    days_load_to_exit = None
    if pos == "outside" and last_cross_ts is not None and (load_end or load_ts) is not None:
        days_load_to_exit = _days(last_cross_ts, load_end or load_ts)

    volume = None
    if last_load is not None and pd.notna(last_load.get("volume_bbl")) and bucket != "exit_no_load":
        volume = float(last_load.get("volume_bbl") or 0)
    elif volume_soh is not None and pd.notna(volume_soh):
        volume = float(volume_soh)

    crude_relevant = bool(
        (last_load is not None and last_load.get("origin_bucket") == "inside_gulf")
        or (cargo_group in CRUDE_EXIT_GROUPS)
        or (afra in CRUDE_CAPABLE_CLASSES)
    )

    return {
        "as_of": as_of.date() if hasattr(as_of, "date") else as_of,
        "imo": int(imo),
        "vessel_name": vessel,
        "bucket": bucket,
        "soh_pos": pos,
        "reason": reason,
        "last_cross_ts": last_cross_ts,
        "last_cross_dir": last_cross_dir,
        "loading_state_soh": loading_state,
        "cargo_group": cargo_group,
        "crossing_type": crossing_type,
        "last_load_ts": load_ts,
        "last_load_end_ts": load_end,
        "last_load_origin": last_load.get("origin_location_name") if last_load is not None else None,
        "last_load_dest": last_load.get("destination_location_name") if last_load is not None else None,
        "last_load_status": last_load.get("status") if last_load is not None else None,
        "last_load_source": last_load.get("source") if last_load is not None else None,
        "origin_bucket": last_load.get("origin_bucket") if last_load is not None else None,
        "dest_bucket": last_load.get("dest_bucket") if last_load is not None else None,
        "grade": last_load.get("closest_ancestor_grade") if last_load is not None else None,
        "berth_confirmed": bool(last_load.get("berth_confirmed")) if last_load is not None else False,
        "volume_bbl": volume,
        "days_inside": days_inside,
        "days_since_load": days_since_load,
        "days_load_to_exit": days_load_to_exit,
        "expected_lag_days": settings.expected_lag_days,
        "alert_lag_days": settings.alert_lag_days,
        "afra_class": afra,
        "deadweight": deadweight,
        "fleet": fleet,
        "sanctioned_iran": sanctioned_iran,
        "sanctioned_sdn": sanctioned_sdn,
        "is_dark_route": is_dark_route,
        "crude_relevant": crude_relevant,
        "watchlist": bucket in {
            "loaded_no_exit",
            "load_without_reentry",
            "exit_no_load",
            "conflict",
            "inside_laden",
            "inside_loading",
        },
    }


def classify_all(
    crossings: pd.DataFrame,
    loads: pd.DataFrame,
    *,
    settings: Settings,
    as_of: date,
    start: date,
) -> pd.DataFrame:
    as_of_ts = pd.Timestamp(as_of) + pd.Timedelta(hours=23, minutes=59)
    start_ts = pd.Timestamp(start)

    crosses = crossings[crossings["crossing_ts"] <= as_of_ts].copy()
    load_cut = loads[loads["load_ts"] <= as_of_ts].copy() if not loads.empty else loads

    # Currently inside = last SoH event is an entry, regardless of when they came in.
    # Windowed crude exits still feed exit_no_load; Kpler loads catch hulls the xlsx never saw.
    last_events = (
        crosses.sort_values(["IMO", "crossing_ts"])
        .groupby("IMO", as_index=False)
        .tail(1)
    )
    currently_inside = last_events[last_events["Direction"] == "Entered MEG"]
    crude_exits = crosses[
        (crosses["Direction"] == "Exited MEG")
        & (crosses["cargo_group"].isin(CRUDE_EXIT_GROUPS))
        & (crosses["crossing_ts"] >= start_ts)
    ]
    kpler_inside = (
        load_cut[load_cut["origin_bucket"] == "inside_gulf"]["vessel_imo"]
        if not load_cut.empty
        else pd.Series(dtype="Int64")
    )
    kpler_bypass = (
        load_cut[load_cut["origin_bucket"].astype(str).str.startswith("bypass")]["vessel_imo"]
        if not load_cut.empty
        else pd.Series(dtype="Int64")
    )
    imos = set(currently_inside["IMO"].dropna().astype(int))
    imos |= set(crude_exits["IMO"].dropna().astype(int))
    imos |= set(pd.to_numeric(kpler_inside, errors="coerce").dropna().astype(int))
    # Bypass book is hulls Kpler loaded east/Red Sea that the SoH tape never saw.
    # Do not re-open old SoH exits just because the same IMO later called Fujairah/Yanbu.
    crossing_imos = set(pd.to_numeric(crosses["IMO"], errors="coerce").dropna().astype(int))
    bypass_imos = set(pd.to_numeric(kpler_bypass, errors="coerce").dropna().astype(int))
    imos |= bypass_imos - crossing_imos

    iran_imos, sdn_imos = ofac_imo_sets(settings)
    crosses_by = {int(imo): grp.sort_values("crossing_ts") for imo, grp in crosses.groupby("IMO")}
    loads_by = {}
    if not load_cut.empty:
        loads_by = {
            int(imo): grp.sort_values("load_ts")
            for imo, grp in load_cut.dropna(subset=["vessel_imo"]).groupby("vessel_imo")
        }

    rows = []
    for imo in sorted(imos):
        row = classify_one(
            imo=imo,
            as_of=as_of_ts,
            settings=settings,
            crosses=crosses_by.get(imo, pd.DataFrame()),
            loads=loads_by.get(imo, pd.DataFrame()),
            iran_imos=iran_imos,
            sdn_imos=sdn_imos,
        )
        if row is not None:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(["watchlist", "bucket", "days_since_load"], ascending=[False, True, False])


def pending_kpler(loads: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Loads after the crossing-file as_of — cannot be checked until the xlsx catches up."""
    if loads.empty:
        return loads
    as_of_ts = pd.Timestamp(as_of) + pd.Timedelta(hours=23, minutes=59)
    pending = loads[loads["load_ts"] > as_of_ts].copy()
    return pending[pending["origin_bucket"] == "inside_gulf"]
