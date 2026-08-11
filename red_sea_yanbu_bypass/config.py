"""
Red Sea W. Coast Saudi crude bypass tracing — knobs.

Houthi strike announcement on W. Coast Saudi crude via Bab el Mandeb (20 Jul 2026).
Patterns of interest:
  full_cycle  — W. Coast -> Ain Sukhna (partial) discharge -> Sidi Kerir reload -> final dest
  ain_sukhna_only — W. Coast -> Ain Sukhna, no Sidi / Suez exit within AIN_SUKHNA_ONLY_DAYS
  sidi_topup — W. Coast partial load (Suez-constrained) -> Sidi Kerir top-up (no Ain Sukhna)
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ENV_PATH = BASE / ".env"
OUT_DIR = HERE / "output"

# Announcement / analysis window
START_DATE = date(2026, 7, 16)
# Pull a few days of pre-window history so early chains still resolve
LOOKBACK_DAYS = 7
# Forward fixtures / in-transit legs
FORWARD_DAYS = 45

PRODUCT = "crude"

# Days after Ain Sukhna discharge with no Sidi Kerir reload / Suez exit
# before labelling the journey ain_sukhna_only
AIN_SUKHNA_ONLY_DAYS = 7

# Partial W. Coast load vs vessel cubic capacity (Suez draft constraint proxy)
PARTIAL_CAPACITY_FRAC = 0.65
# If capacity missing: VLCC/ULCC treated as partial below this kbbl
VLCC_PARTIAL_KBBL_FALLBACK = 1500.0
# m³ -> barrels (oil); kbbl = m3 * M3_TO_BBL / 1000
M3_TO_BBL = 6.2898

# Classes that count toward Med-side Sidi volume charts (collapsed, once each)
SIDI_VOLUME_CLASSES = ("full_cycle", "sidi_topup")

# Country-level Kpler pull; W. Coast rows kept via origin keywords below
FROM_ZONES = ["Saudi Arabia"]

# Substrings matched against origin / installation / zone names (case-insensitive)
W_COAST_ORIGIN_KEYWORDS = (
    "yanbu",
    "muajjiz",
    "jeddah",
    "rabigh",
    "jizan",
    "jazan",
    "neom",
    "duba",
)

# Discharge / reload / Suez aliases (case-insensitive contains)
AIN_SUKHNA_ALIASES = (
    "ain sukhna",
    "ayn sukhna",
    "sukna",
)
SIDI_KERIR_ALIASES = (
    "sidi kerir",
    "sidi keir",
    "sidikerir",
)
SUEZ_EXIT_ALIASES = (
    "suez canal",
    "suez",
    "port said",
    "port-said",
)

# Destination columns to probe (fixtures often only have forecast names)
DEST_COLS = (
    "destination_location_name",
    "installation_destination_name",
    "next_forecasted_destination_location_name",
    "zone_destination_name",
)

ORIGIN_COLS = (
    "origin_location_name",
    "installation_origin_name",
    "zone_origin_name",
)

TRADE_COLS = [
    "vessel_name",
    "vessel_imo",
    "vessel_type",
    "vessel_capacity_cubic_meters",
    "status",
    "closest_ancestor_product",
    "closest_ancestor_grade",
    "origin_location_name",
    "installation_origin_name",
    "zone_origin_name",
    "destination_location_name",
    "installation_destination_name",
    "next_forecasted_destination_location_name",
    "zone_destination_name",
    "destination_country_name",
    "continent_destination_name",
    "destination_subcontinent_name",
    "start",
    "end",
    "origin_start",
    "origin_end",
    "origin_eta_date",
    "destination_start",
    "destination_end",
    "destination_eta",  # Kpler name (not destination_eta_date)
    "cargo_origin_barrels_split_by_product",
]

PORT_CALL_COLS = [
    "vessel_name",
    "vessel_imo",
    "vessel_type",
    "installation_name",
    "location_name",
    "zone_name",
    "start",
    "end",
    "closest_ancestor_group",
    "cargo_origin_barrels_split_by_product",
    "is_sts",
    "is_reexport",
]
