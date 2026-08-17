"""Load and enrich soh_crossing_hist.xlsx (same cleaning as soh_crossing_analysis)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kpler_vs_soh.config import Settings
from kpler_vs_soh.util import infer_cargo, map_afra_class, normalize_imo, parse_dmy, to_number


def load_crossings(settings: Settings) -> pd.DataFrame:
    raw = pd.read_excel(settings.crossing_path, sheet_name="SoH Crossing")
    afra = pd.read_excel(settings.afra_path)
    ofac_iran = pd.read_csv(settings.ofac_iran_path)
    ofac_sdn = pd.read_csv(settings.ofac_sdn_path)

    df = raw.copy()
    df["Crossing Date"] = parse_dmy(df["Crossing Date"])
    df["IMO"] = normalize_imo(df["IMO"])
    df["Quantity"] = to_number(df["Quantity"]).fillna(0.0)
    df["Deadweight"] = to_number(df["Deadweight"])
    df["dwt_kdwt"] = df["Deadweight"] / 1000.0
    df["afra_class"] = df["dwt_kdwt"].map(lambda x: map_afra_class(x, afra))

    cargo = df.apply(infer_cargo, axis=1, result_type="expand")
    cargo.columns = ["cargo_group", "cargo_source"]
    df = pd.concat([df, cargo], axis=1)

    volume_groups = {"crude", "likely_crude", "products"}
    df["volume_bbl"] = np.where(df["cargo_group"].isin(volume_groups), df["Quantity"], 0.0)

    iran_imos = set(normalize_imo(ofac_iran["imo"]).dropna().astype(int))
    sdn_imos = set(normalize_imo(ofac_sdn["imo"]).dropna().astype(int))
    df["sanctioned_iran"] = df["IMO"].isin(iran_imos)
    df["sanctioned_sdn"] = df["IMO"].isin(sdn_imos)
    df["sanctioned_any"] = df["sanctioned_iran"] | df["sanctioned_sdn"]
    df["is_dark_route"] = df["Crossing Type"].eq("Dark/Unknown Route")
    df["is_dark_fleet"] = df["sanctioned_any"]
    df["fleet"] = np.where(df["is_dark_fleet"], "dark fleet", "clean fleet")
    df["crossing_ts"] = df["Crossing Date"]
    return df.dropna(subset=["IMO", "crossing_ts"]).sort_values(["IMO", "crossing_ts"])


def ofac_imo_sets(settings: Settings) -> tuple[set[int], set[int]]:
    """Keep a separate lookup so Kpler-only hulls (never in the xlsx) still get a fleet flag."""
    ofac_iran = pd.read_csv(settings.ofac_iran_path)
    ofac_sdn = pd.read_csv(settings.ofac_sdn_path)
    iran = set(normalize_imo(ofac_iran["imo"]).dropna().astype(int))
    sdn = set(normalize_imo(ofac_sdn["imo"]).dropna().astype(int))
    return iran, sdn
