"""Shared parsing + origin geography (why a load should or should not cross SoH)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from kpler_vs_soh.config import (
    CRUDE_LIKE_CLASSES,
    CRUDE_TAGS,
    EAST_COAST_LOAD_KEYWORDS,
    INSIDE_GULF_COUNTRIES,
    PRODUCT_TAGS,
    RED_SEA_ORIGIN_KEYWORDS,
)


def parse_env(path: Path) -> dict[str, str]:
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


def parse_dmy(series: pd.Series) -> pd.Series:
    """Crossing file dates are DD/MM/YYYY; dayfirst avoids US month/day swaps."""
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def normalize_imo(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def map_afra_class(dwt_kdwt: float, afra_table: pd.DataFrame) -> str | None:
    if pd.isna(dwt_kdwt):
        return None
    table = afra_table.sort_values("capacity_lower_bound")
    for _, row in table.iterrows():
        if row["capacity_lower_bound"] <= dwt_kdwt <= row["capacity_upper_bound"]:
            return row["vessel_class"]
    if dwt_kdwt < table["capacity_lower_bound"].min():
        return "Below GP"
    return "Above ULCC"


def infer_cargo(row: pd.Series) -> tuple[str, str]:
    """Same rules as soh_crossing_analysis so crude exits stay comparable."""
    loading = row["Loading State"]
    tag = row["Cargo Type"]
    vclass = row["afra_class"]
    if loading == "Ballast":
        return "ballast", "loading_state"
    if pd.notna(tag):
        if tag in CRUDE_TAGS:
            return "crude", "tagged"
        if tag in PRODUCT_TAGS:
            return "products", "tagged"
        return "other", "tagged"
    if vclass in CRUDE_LIKE_CLASSES:
        return "likely_crude", "inferred_afra"
    return "unknown", "untagged"


def _contains_any(name, keywords: tuple[str, ...]) -> bool:
    """Phrase match with letter boundaries so 'duba' (Red Sea) does not hit 'Dubai'."""
    if pd.isna(name) or str(name).strip() == "":
        return False
    text = str(name).lower()
    return any(re.search(rf"(?<![a-z]){re.escape(key)}(?![a-z])", text) for key in keywords)


def origin_bucket(origin_name, origin_country=None) -> str:
    """inside_gulf must cross SoH; bypass origins must not be treated as missing exits."""
    if _contains_any(origin_name, RED_SEA_ORIGIN_KEYWORDS):
        return "bypass_red_sea"
    if _contains_any(origin_name, EAST_COAST_LOAD_KEYWORDS):
        return "bypass_east"
    country = str(origin_country).strip().lower() if pd.notna(origin_country) else ""
    if country == "oman":
        return "bypass_east"
    if origin_name is None or str(origin_name).strip() == "":
        return "unknown"
    return "inside_gulf"


def dest_bucket(dest_name, dest_country=None) -> str:
    """Intra-gulf discharges should not be flagged as missing Hormuz exits."""
    if _contains_any(dest_name, RED_SEA_ORIGIN_KEYWORDS):
        return "bypass_red_sea"
    if _contains_any(dest_name, EAST_COAST_LOAD_KEYWORDS):
        return "bypass_east"
    country = str(dest_country).strip().lower() if pd.notna(dest_country) else ""
    if country == "oman":
        return "bypass_east"
    if country in INSIDE_GULF_COUNTRIES:
        return "inside_gulf"
    if _contains_any(dest_name, tuple(INSIDE_GULF_COUNTRIES)):
        return "inside_gulf"
    if dest_name is None or str(dest_name).strip() == "":
        return "unknown"
    return "outside"


def normalize_sts_zone(zone_name) -> str:
    """Group GoO query rows into Fujairah Light / Sohar Light / Gulf of Oman."""
    if pd.isna(zone_name):
        return "Unknown"
    z = str(zone_name).strip().rstrip(".")
    zl = z.lower()
    if "fujairah" in zl:
        return "Fujairah Light"
    if "sohar" in zl:
        return "Sohar Light"
    if "gulf of oman" in zl or "oman gulf" in zl:
        return "Gulf of Oman"
    return z
