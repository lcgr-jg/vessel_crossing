"""Kpler crude loadings vs Strait of Hormuz crossings — v1 settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent

# Same window as soh_crossing_analysis / goo_sts_vs_soh.
DEFAULT_START = date(2026, 6, 17)

# Pull Kpler further back so a May load can still match a June/July exit.
TRADE_LOOKBACK_DAYS = 60

# Typical Basrah/Ras Tanura -> Hormuz steaming. Residual notebook uses 3.
EXPECTED_LAG_DAYS = 3

# Still inside after a completed load past this → loaded_no_exit (watchlist).
ALERT_LAG_DAYS = 7

# How far before an SoH exit we still accept a Kpler load as the same cargo.
LOAD_LOOKBACK_DAYS = 21

# PortCall berth vs trade load_ts join window.
BERTH_MATCH_DAYS = 3

# How far after an SoH exit we still treat a GoO STS as the same cargo overlay.
# goo_sts_vs_soh uses 5; overlay keeps any STS after the last crossing and reports lag.
STS_LAG_DAYS = 14
STS_LOOKAHEAD_DAYS = 5
STS_PAIR_MATCH_DAYS = 3
STS_ZONES = ["Gulf of Oman"]

MEG_ORIGIN_ZONES = [
    "Saudi Arabia",
    "Iraq",
    "United Arab Emirates",
    "Kuwait",
    "Qatar",
    "Iran",
    "Oman",
    "Bahrain",
]

INSIDE_GULF_COUNTRIES = {
    "saudi arabia",
    "iraq",
    "united arab emirates",
    "kuwait",
    "qatar",
    "iran",
    "bahrain",
}

# Cargo loaded east of Hormuz never needs an SoH exit. Applied to origin names,
# not to STS zone labels alone — same rule as goo_sts_vs_soh.
EAST_COAST_LOAD_KEYWORDS = (
    "adcop",
    "fujairah",
    "sohar",
    "kalba",
    "dibba",
    "khor fakkan",
    "khorfakkan",
    "fott",
    "jask",
    "mina al fahal",
    "mina al-fahal",
    "fahal",
    "gulf of oman",
    "oman gulf",
    "ras markaz",
)

RED_SEA_ORIGIN_KEYWORDS = (
    "yanbu",
    "jeddah",
    "rabigh",
    "muajjiz",
    "jizan",
    "jazan",
    "neom",
    "duba",
    "shoaiba",
)

PRODUCT_TAGS = {"CPP", "DPP", "Chemicals", "Methanol", "Chemicals, Methanol"}
CRUDE_TAGS = {"Crude/Co"}
CRUDE_LIKE_CLASSES = {"VLCC", "ULCC", "Above ULCC"}
CRUDE_CAPABLE_CLASSES = {"LR1", "AFRA", "LR2", "VLCC", "ULCC", "Above ULCC"}

LOADING_STATUSES = {"loading", "scheduled"}
COMPLETED_STATUSES = {"in transit", "delivered"}

WATCHLIST_BUCKETS = (
    "loaded_no_exit",
    "load_without_reentry",
    "exit_no_load",
    "conflict",
    "inside_laden",
    "inside_loading",
)

# Overlay flags that put a hull on the watchlist even if the load→strait bucket is quiet.
STS_WATCH_FLAGS = (
    "giver_no_soh",
    "unknown_partner",
    "sts_inside",
)

BUCKET_ORDER = (
    "loaded_no_exit",
    "load_without_reentry",
    "exit_no_load",
    "conflict",
    "inside_laden",
    "inside_loading",
    "inside_ballast",
    "intra_gulf",
    "accounted",
    "bypass",
    "outside_other",
)

TRADE_COLUMNS = [
    "vessel_name",
    "vessel_imo",
    "status",
    "closest_ancestor_product",
    "closest_ancestor_grade",
    "origin_location_name",
    "origin_country_name",
    "destination_location_name",
    "destination_country_name",
    "start",
    "end",
    "origin_start",
    "origin_end",
    "cargo_origin_barrels_split_by_product",
    "load_port_call_id",
]

PORTCALL_COLUMNS = [
    "vessel_name",
    "vessel_imo",
    "zone_name",
    "location_name",
    "installation_name",
    "country_name",
    "start",
    "end",
    "is_sts",
    "cargo_origin_barrels_split_by_product",
    "closest_ancestor_grade",
    "closest_ancestor_product",
    "port_call_id",
    "voyage_id",
]

STS_COLUMNS = [
    "load_vessel_name",
    "load_vessel_imo",
    "discharge_vessel_name",
    "discharge_vessel_imo",
    "zone_name",
    "start",
    "end",
    "cargo_origin_barrels_split_by_product",
    "closest_ancestor_product",
    "closest_ancestor_grade",
    "ship_to_ship_id",
]


@dataclass
class Settings:
    start_date: date = DEFAULT_START
    end_date: date | None = None
    trade_lookback_days: int = TRADE_LOOKBACK_DAYS
    expected_lag_days: int = EXPECTED_LAG_DAYS
    alert_lag_days: int = ALERT_LAG_DAYS
    load_lookback_days: int = LOAD_LOOKBACK_DAYS
    berth_match_days: int = BERTH_MATCH_DAYS
    sts_lag_days: int = STS_LAG_DAYS
    sts_lookahead_days: int = STS_LOOKAHEAD_DAYS
    sts_pair_match_days: int = STS_PAIR_MATCH_DAYS
    sts_zones: list[str] = field(default_factory=lambda: list(STS_ZONES))
    product: str = "crude"
    meg_origin_zones: list[str] = field(default_factory=lambda: list(MEG_ORIGIN_ZONES))

    crossing_path: Path = BASE / "soh_crossing_hist.xlsx"
    afra_path: Path = BASE / "afra_scale_mapping.xlsx"
    ofac_iran_path: Path = BASE / "ofac_iran_vessels_history.csv"
    env_path: Path = BASE / ".env"
    cache_dir: Path = HERE / "cache"
    data_dir: Path = HERE / "data"
    output_dir: Path = HERE / "output"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "kpler_vs_soh.duckdb"

    @property
    def ofac_sdn_path(self) -> Path:
        files = sorted(BASE.glob("ofac_sdn_vessels_*.csv"))
        if files:
            return files[-1]
        return BASE / "ofac_sdn_vessels_2026-07-13.csv"
