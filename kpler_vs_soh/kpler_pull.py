"""Kpler Trades + PortCalls with parquet cache so re-runs do not re-hit the API."""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import pandas as pd

from kpler_vs_soh.config import PORTCALL_COLUMNS, STS_COLUMNS, TRADE_COLUMNS, Settings
from kpler_vs_soh.util import dest_bucket, normalize_imo, origin_bucket, parse_env, to_number


def _sdk_config(settings: Settings):
    cred = parse_env(settings.env_path)
    email = cred.get("KPLER_EMAIL") or cred.get("EMAIL")
    password = cred.get("KPLER_PASSWORD") or cred.get("PASSWORD")
    if not email or not password:
        raise RuntimeError(f"Missing KPLER_EMAIL / KPLER_PASSWORD in {settings.env_path}")

    from kpler.sdk import Platform
    from kpler.sdk.configuration import Configuration

    return Configuration(Platform.Liquids, email, password)


def _clients(settings: Settings):
    from kpler.sdk.resources.port_calls import PortCalls
    from kpler.sdk.resources.trades import Trades

    config = _sdk_config(settings)
    return Trades(config), PortCalls(config)


def _available_columns(client, wanted: list[str]) -> list[str]:
    """Ask the SDK which ids exist so a renamed column does not kill the pull."""
    try:
        cols = client.get_columns()
        available = set(cols["id"].astype(str))
        kept = [c for c in wanted if c in available]
        return kept or wanted
    except Exception:
        return wanted


def _read_cache(path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def _write_cache(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except ImportError:
        df.to_csv(path.with_suffix(".csv"), index=False)


def fetch_trades(
    settings: Settings,
    *,
    start: date,
    end: date,
    refresh: bool = False,
) -> pd.DataFrame:
    cache_path = settings.cache_dir / f"trades_{start}_{end}.parquet"
    if not refresh:
        cached = _read_cache(cache_path)
        if cached is not None:
            return cached

    trades_client, _ = _clients(settings)
    from kpler.sdk import TradesStatus

    columns = _available_columns(trades_client, TRADE_COLUMNS)
    frames = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for zone in settings.meg_origin_zones:
            chunk = trades_client.get(
                products=[settings.product],
                from_zones=[zone],
                start_date=start,
                end_date=end,
                with_forecast=True,
                with_intra_country=True,
                with_intra_region=True,
                trade_status=[
                    TradesStatus.Scheduled,
                    TradesStatus.Loading,
                    TradesStatus.InTransit,
                    TradesStatus.Delivered,
                ],
                columns=columns,
                size=10000,
            )
            if chunk is None or chunk.empty:
                continue
            chunk = chunk.copy()
            chunk["_from_zone"] = zone
            frames.append(chunk)

    if not frames:
        empty = pd.DataFrame(columns=columns)
        _write_cache(empty, cache_path)
        return empty

    df = pd.concat(frames, ignore_index=True)
    if len(df) >= 10000:
        warnings.warn("Kpler trades hit the 10k cap on at least one zone — results may be truncated.")
    _write_cache(df, cache_path)
    return df


def fetch_port_calls(
    settings: Settings,
    *,
    start: date,
    end: date,
    refresh: bool = False,
) -> pd.DataFrame:
    cache_path = settings.cache_dir / f"portcalls_{start}_{end}.parquet"
    if not refresh:
        cached = _read_cache(cache_path)
        if cached is not None:
            return cached

    _, pc_client = _clients(settings)
    columns = _available_columns(pc_client, PORTCALL_COLUMNS)
    frames = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for zone in settings.meg_origin_zones:
            chunk = pc_client.get(
                zones=[zone],
                products=[settings.product],
                start_date=start,
                end_date=end,
                with_forecast=False,
                columns=columns,
                size=10000,
            )
            if chunk is None or chunk.empty:
                continue
            chunk = chunk.copy()
            chunk["_from_zone"] = zone
            frames.append(chunk)

    if not frames:
        empty = pd.DataFrame(columns=columns)
        _write_cache(empty, cache_path)
        return empty

    df = pd.concat(frames, ignore_index=True)
    _write_cache(df, cache_path)
    return df


def fetch_ship_to_ships(
    settings: Settings,
    *,
    start: date,
    end: date,
    refresh: bool = False,
) -> pd.DataFrame:
    cache_path = settings.cache_dir / f"sts_{start}_{end}.parquet"
    if not refresh:
        cached = _read_cache(cache_path)
        if cached is not None:
            return cached

    from kpler.sdk.resources.ship_to_ships import ShipToShips

    client = ShipToShips(_sdk_config(settings))
    columns = _available_columns(client, STS_COLUMNS)
    frames = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for zone in settings.sts_zones:
            chunk = client.get(
                zones=[zone],
                products=[settings.product],
                start_date=start,
                end_date=end,
                columns=columns,
                size=10000,
            )
            if chunk is None or chunk.empty:
                continue
            chunk = chunk.copy()
            chunk["_query_zone"] = zone
            frames.append(chunk)

    if not frames:
        empty = pd.DataFrame(columns=columns)
        _write_cache(empty, cache_path)
        return empty

    df = pd.concat(frames, ignore_index=True)
    _write_cache(df, cache_path)
    return df


def fetch_goo_port_calls(
    settings: Settings,
    *,
    start: date,
    end: date,
    refresh: bool = False,
) -> pd.DataFrame:
    """Gulf of Oman PortCalls — needed for unknown-partner STS (is_sts, no named pair)."""
    cache_path = settings.cache_dir / f"portcalls_goo_{start}_{end}.parquet"
    if not refresh:
        cached = _read_cache(cache_path)
        if cached is not None:
            return cached

    _, pc_client = _clients(settings)
    columns = _available_columns(pc_client, PORTCALL_COLUMNS)
    frames = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for zone in settings.sts_zones:
            chunk = pc_client.get(
                zones=[zone],
                products=[settings.product],
                start_date=start,
                end_date=end,
                with_forecast=False,
                columns=columns,
                size=10000,
            )
            if chunk is None or chunk.empty:
                continue
            chunk = chunk.copy()
            chunk["_query_zone"] = zone
            frames.append(chunk)

    if not frames:
        empty = pd.DataFrame(columns=columns)
        _write_cache(empty, cache_path)
        return empty

    df = pd.concat(frames, ignore_index=True)
    _write_cache(df, cache_path)
    return df


def _clean_trades(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    for col in ("start", "end", "origin_start", "origin_end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            df[col] = pd.NaT
    df["vessel_imo"] = normalize_imo(df["vessel_imo"])
    df["volume_bbl"] = to_number(df["cargo_origin_barrels_split_by_product"])
    df["status"] = df["status"].astype(str).str.strip()
    df["status_norm"] = df["status"].str.lower()
    # origin_end is when the ship left the load berth; fall back to voyage start.
    df["load_ts"] = df["origin_end"].fillna(df["origin_start"]).fillna(df["start"])
    df["load_end_ts"] = df["origin_end"].fillna(df["load_ts"])
    origin_country = df["origin_country_name"] if "origin_country_name" in df.columns else None
    dest_country = df["destination_country_name"] if "destination_country_name" in df.columns else None
    df["origin_bucket"] = [
        origin_bucket(o, c)
        for o, c in zip(
            df["origin_location_name"],
            origin_country if origin_country is not None else [None] * len(df),
        )
    ]
    df["dest_bucket"] = [
        dest_bucket(o, c)
        for o, c in zip(
            df["destination_location_name"] if "destination_location_name" in df.columns else [None] * len(df),
            dest_country if dest_country is not None else [None] * len(df),
        )
    ]
    if "load_port_call_id" in df.columns:
        df["load_id"] = df["load_port_call_id"].astype(str)
    else:
        df["load_id"] = (
            df["vessel_imo"].astype(str)
            + "|"
            + df["origin_location_name"].astype(str)
            + "|"
            + df["load_ts"].astype(str)
        )
    df["source"] = "trade"
    df["is_sts"] = False
    df["berth_confirmed"] = False
    return df.dropna(subset=["vessel_imo", "load_ts"]).drop_duplicates(subset=["load_id"], keep="last")


def _clean_port_calls(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    for col in ("start", "end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            df[col] = pd.NaT
    df["vessel_imo"] = normalize_imo(df["vessel_imo"])
    df["volume_bbl"] = to_number(df["cargo_origin_barrels_split_by_product"])
    if "is_sts" in df.columns:
        df["is_sts"] = df["is_sts"].fillna(False).astype(bool)
    else:
        df["is_sts"] = False
    origin_name = df["installation_name"] if "installation_name" in df.columns else pd.Series(pd.NA, index=df.index)
    for col in ("location_name", "zone_name"):
        if col in df.columns:
            origin_name = origin_name.fillna(df[col])
    country = df["country_name"] if "country_name" in df.columns else None
    df["origin_location_name"] = origin_name
    df["origin_bucket"] = [
        origin_bucket(o, c)
        for o, c in zip(origin_name, country if country is not None else [None] * len(df))
    ]
    df["destination_location_name"] = pd.NA
    df["dest_bucket"] = "unknown"
    df["status"] = "PortCall"
    df["status_norm"] = "portcall"
    df["load_ts"] = df["end"].fillna(df["start"])
    df["load_end_ts"] = df["end"].fillna(df["start"])
    df["closest_ancestor_grade"] = df.get("closest_ancestor_grade")
    df["load_id"] = df["port_call_id"].astype(str) if "port_call_id" in df.columns else (
        df["vessel_imo"].astype(str) + "|" + origin_name.astype(str) + "|" + df["load_ts"].astype(str)
    )
    df["source"] = "portcall"
    df["berth_confirmed"] = True
    df = df.dropna(subset=["vessel_imo", "load_ts"])
    # Negative barrels are discharges; they are not gulf loadings.
    vol = pd.to_numeric(df["volume_bbl"], errors="coerce")
    return df.loc[vol.fillna(0) > 0].copy()


def build_loads(
    trades_raw: pd.DataFrame,
    portcalls_raw: pd.DataFrame,
    *,
    berth_match_days: int,
) -> pd.DataFrame:
    """Trades are canonical cargo rows; unmatched PortCalls fill Kpler loadings with no trade."""
    trades = _clean_trades(trades_raw)
    calls = _clean_port_calls(portcalls_raw)
    if trades.empty and calls.empty:
        return pd.DataFrame()

    if not trades.empty and not calls.empty:
        # Same IMO + nearby berth start means the PortCall is the physical twin of the trade.
        window = pd.Timedelta(days=berth_match_days)
        matched_pc_ids: set[str] = set()
        berth_flags = []
        calls_by = {imo: grp for imo, grp in calls.groupby("vessel_imo")}
        for _, trade in trades.iterrows():
            cand = calls_by.get(trade["vessel_imo"])
            if cand is None or cand.empty:
                berth_flags.append(False)
                continue
            delta = (cand["start"] - trade["load_ts"]).abs()
            nearby = delta[delta <= window]
            if nearby.empty:
                berth_flags.append(False)
                continue
            berth_flags.append(True)
            matched_pc_ids.add(str(cand.loc[nearby.idxmin(), "load_id"]))
        trades = trades.copy()
        trades["berth_confirmed"] = berth_flags
        extra = calls[~calls["load_id"].astype(str).isin(matched_pc_ids)].copy()
    else:
        extra = calls

    parts = [p for p in (trades, extra) if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame()
    keep = [
        "load_id",
        "source",
        "vessel_name",
        "vessel_imo",
        "origin_location_name",
        "destination_location_name",
        "origin_bucket",
        "dest_bucket",
        "status",
        "status_norm",
        "load_ts",
        "load_end_ts",
        "volume_bbl",
        "closest_ancestor_grade",
        "is_sts",
        "berth_confirmed",
    ]
    aligned = []
    for part in parts:
        for col in keep:
            if col not in part.columns:
                part[col] = pd.NA
        aligned.append(part[keep])
    out = pd.concat(aligned, ignore_index=True)
    return out.sort_values(["vessel_imo", "load_ts"]).reset_index(drop=True)


def kpler_window(settings: Settings, crossing_max: date) -> tuple[date, date]:
    start = settings.start_date - timedelta(days=settings.trade_lookback_days)
    end = settings.end_date or crossing_max
    return start, end
