"""
AOI configuration helpers for copernicus_vessel_detection.ipynb.

Use **templates** so new terminals inherit tuned detect params instead of
copy-pasting an entire ``detect`` dict from another site.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

# ---------------------------------------------------------------------------
# Detection templates — pick ONE per terminal, then override sparingly
# ---------------------------------------------------------------------------

TERMINAL_TEMPLATES: dict[str, dict[str, Any]] = {
    # Multi-berth jetty (Yanbu, Muajjiz): fixed pixel slots + S2 diff/hull
    "berth_slots_jetty": {
        "mode": "berth_slots",
        "sensor": "s2",
        "detect": {
            "water_gray_hi": 120,
            "use_adaptive_diff": True,
            "diff_percentile": 88,
            "use_hull_colors": True,
            "slot_diff_abs": 45,
            # Absolute size kills pier-glint FPs (Muajjiz berth_1 style).
            "slot_min_blob_px": 60,
            "slot_min_length_m": 110,
            "scene_min_valid_frac": 0.55,
            "slot_hull_frac": 0.04,
            "slot_min_blob_frac": 0.28,
            "slot_diff_frac": 0.15,
            "slot_diff_blob_frac": 0.14,
            # Optional A/B: set use_yolo=True (or AOI detector=\"yolo\") for mayrajeo S2 YOLO.
            "use_yolo": False,
            "yolo_conf": 0.25,
            "yolo_min_length_m": 80,
        },
        "description": "Crude jetty with named berth boxes. Tune slot_* after picker + Kpler compare.",
        "setup_steps": [
            "Draw polygon in Copernicus Browser -> paste as GEOJSON",
            "Pick berth slot boxes (picker cell) -> paste as BERTH_SLOTS list",
            "Add AOI entry with template='berth_slots_jetty' + geojson + berth_slots_px",
            "Run plot_berth_slots on 2–3 reference dates; adjust slot boxes if needed",
            "Run Kpler compare; tune detect overrides only on mismatch dates",
            "Add Kpler installation name to AOI_KPLER_LOCATIONS",
        ],
        "tunable_detect_keys": [
            "slot_min_blob_px / slot_min_length_m — raise to kill pier FPs",
            "slot_diff_abs — raise if empty berths false-positive (background drift)",
            "slot_min_blob_frac — raise to reduce FPs; lower to catch grey hulls",
            "slot_hull_frac — lower if green/rust hulls missed",
            "scene_min_valid_frac — skip damaged/cloudy scenes",
        ],
    },
    # Multi-SPM terminal (CPC): S2 + S1 vote, NDWI anomaly with RGB support
    "berth_slots_spm": {
        "mode": "berth_slots",
        "sensor": "s2_s1",
        "detect": {
            "water_gray_hi": 100,
            "use_adaptive_diff": True,
            "diff_percentile": 88,
            "use_hull_colors": True,
            "slot_water_only": False,
            "use_ndwi_water": False,
            "ndwi_vessel_enabled": True,
            "ndwi_vessel_pct": 15,
            "ndwi_vessel_delta": 0.05,
            "ndwi_support_blob_px": 40,
            "slot_min_blob_px": 80,
            "slot_min_length_m": 150,
            "scene_min_valid_frac": 0.55,
            "s1_slot_vote": True,
            "s1_water_max_db": -12.0,
            "s1_adaptive_percentile": 99.5,
            "s1_min_area_px": 6,
            "s1_min_length_m": 80,
            "slot_diff_abs": 40,
            "slot_hull_frac": 0.025,
            "slot_min_hull_blob_frac": 0.02,
            "slot_min_blob_frac": 0.32,
            "slot_diff_frac": 0.18,
            "slot_diff_blob_frac": 0.16,
            "slot_use_relative": True,
            "slot_relative_blob_ratio": 0.4,
            "use_yolo": False,
            "yolo_conf": 0.25,
            "yolo_min_length_m": 80,
        },
        "description": "Several SPMs in one AOI. Shrink boxes to the vessel footprint on water only.",
        "setup_steps": [
            "Draw polygon covering all SPMs",
            "Pick tight boxes on each SPM mooring (water + vessel only, not wide wake)",
            "Use template='berth_slots_spm'",
            "Run diagnose_berth_slots() on known occupied/empty dates",
            "Run Kpler compare; override slot_* only if needed",
        ],
        "tunable_detect_keys": [
            "slot_min_blob_px / slot_min_length_m — absolute size (prefer over frac on large boxes)",
            "ndwi_vessel_pct / ndwi_support_blob_px — anomaly + RGB support",
            "s1_adaptive_percentile — SAR second vote sensitivity",
            "scene_min_valid_frac — skip damaged scenes",
            "use_yolo / yolo_conf — mayrajeo S2 YOLO A/B trial",
        ],
    },
    # One SPM per polygon (Das, Zirku, Um Lulu): S2 + S1 vote
    "single_spm_polygon": {
        "mode": "berth_slots",
        "sensor": "s2_s1",
        "detect": {
            "water_gray_hi": 100,
            "use_adaptive_diff": True,
            "diff_percentile": 88,
            "use_hull_colors": True,
            "slot_water_only": False,
            "use_ndwi_water": False,
            "ndwi_vessel_enabled": True,
            "ndwi_vessel_pct": 15,
            "ndwi_vessel_delta": 0.05,
            "ndwi_support_blob_px": 40,
            "slot_min_blob_px": 80,
            "slot_min_length_m": 150,
            "scene_min_valid_frac": 0.55,
            "s1_slot_vote": True,
            "s1_water_max_db": -12.0,
            "s1_adaptive_percentile": 99.5,
            "s1_min_area_px": 6,
            "s1_min_length_m": 80,
            "slot_diff_abs": 40,
            "slot_hull_frac": 0.025,
            "slot_min_hull_blob_frac": 0.02,
            "slot_min_blob_frac": 0.32,
            "slot_diff_frac": 0.18,
            "slot_diff_blob_frac": 0.16,
            "slot_use_relative": False,
            "use_yolo": False,
            "yolo_conf": 0.25,
            "yolo_min_length_m": 80,
        },
        "description": "Single SPM in its own polygon. Pick one tight box on the mooring footprint.",
        "setup_steps": [
            "Draw polygon around one SPM in Copernicus Browser",
            "Pick a tight box on water + vessel (picker cell)",
            "Use template='single_spm_polygon' + geojson + berth_slots_px (one slot)",
            "Run diagnose_berth_slots() on known occupied/empty dates",
            "Run Kpler compare; override slot_* only if needed",
        ],
        "tunable_detect_keys": [
            "slot_min_blob_px / slot_min_length_m — absolute size gates for tankers",
            "ndwi_support_blob_px — require RGB support with NDWI anomaly",
            "s1_min_length_m — SAR second vote",
            "scene_min_valid_frac — skip damaged scenes",
            "use_yolo / yolo_conf — mayrajeo S2 YOLO A/B trial",
        ],
    },
    # Open-water single-point SPM (legacy): S2 blob + S1 SAR fallback
    "single_spm": {
        "mode": "single_berth",
        "sensor": "s2_s1",
        "detect": {
            "s1_water_max_db": -12.0,
            "s1_adaptive_percentile": 99.5,
            "s1_min_area_px": 4,
            "s1_min_length_m": 40,
        },
        "description": "Single loading point in open water. Needs lat/lon + buffer_m.",
        "setup_steps": [
            "Set type='point', lat, lon, buffer_m (~1200 m for SPM)",
            "Add template='single_spm' (no berth slots)",
            "Verify AOI frames the SPM in Copernicus Browser",
            "Add Kpler zone or installation to AOI_KPLER_LOCATIONS",
        ],
        "tunable_detect_keys": [
            "s1_adaptive_percentile — raise if SAR false positives on clutter",
            "s1_water_max_db — water mask for SAR",
        ],
    },
    # Legacy blob detector on a polygon (rarely used now)
    "multi_berth_blob": {
        "mode": "multi_berth",
        "sensor": "s2",
        "detect": {},
        "description": "Blob counting in a zone without fixed slots. Prefer berth_slots_jetty when possible.",
        "setup_steps": [
            "Define polygon AOI",
            "Use template='multi_berth_blob'",
            "Tune diff_percentile / min_length_m against known scenes",
        ],
        "tunable_detect_keys": ["diff_percentile", "min_length_m", "dedupe_px"],
    },
}

# Kpler defaults (extend in notebook AOI_KPLER_LOCATIONS)
DEFAULT_KPLER_HINTS: dict[str, str] = {
    "yanbu_north_crude_terminal": 'LocationSpec("Yanbu Crude", kind="installation")',
    "muajjiz": 'LocationSpec("Muajjiz", kind="installation")',
    "zirku_spm_1": 'LocationSpec("Zirku", kind="zone")  # zone covers both SPMs',
    "zirku_spm_2": 'LocationSpec("Zirku", kind="zone")',
    "das_spm_1": 'LocationSpec("Das Island", kind="zone")',
    "das_spm_2": 'LocationSpec("Das Island", kind="zone")',
    "um_lulu_spm": 'LocationSpec("Umm Lulu", kind="zone")',
    "cpc": 'LocationSpec("CPC Terminal", kind="installation")',
}


def apply_template(cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge template defaults into an AOI config (does not mutate input)."""
    out = deepcopy(cfg)
    template_name = out.pop("template", None)
    if not template_name:
        return out
    if template_name not in TERMINAL_TEMPLATES:
        raise KeyError(
            f"Unknown template {template_name!r}. "
            f"Choose from: {list(TERMINAL_TEMPLATES)}"
        )
    tpl = TERMINAL_TEMPLATES[template_name]
    for key in ("mode", "sensor"):
        out.setdefault(key, tpl[key])
    merged_detect = deepcopy(tpl["detect"])
    merged_detect.update(out.get("detect") or {})
    out["detect"] = merged_detect
    out["_template"] = template_name
    return out


def get_detect_params_from_cfg(
    cfg: dict[str, Any],
    detect_profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Resolve final detect params for one AOI.

    Priority: DETECT_PROFILES base (legacy) → template detect → cfg['detect'] overrides.
    """
    cfg = apply_template(cfg)
    mode = cfg.get("mode", "single_berth")
    profile = "multi_berth" if mode in ("multi_berth", "berth_slots") else mode
    params = deepcopy(detect_profiles.get(profile, {}))
    params.update(cfg.get("detect") or {})
    return params


def default_single_spm_slot(name: str = "spm") -> list[dict]:
    """One placeholder box per SPM polygon — replace via picker."""
    return [{"name": name, "box": [20, 20, 120, 120]}]


def placeholder_berth_slots(
    n: int,
    *,
    prefix: str = "berth",
    grid_w: int = 50,
    grid_h: int = 35,
    gap: int = 8,
) -> list[dict]:
    """Generate evenly spaced placeholder boxes — replace via picker."""
    slots = []
    for i in range(n):
        x1 = 10 + i * (grid_w + gap)
        slots.append(
            {
                "name": f"{prefix}_{i + 1}",
                "box": [x1, 10 + i * 5, x1 + grid_w, 10 + i * 5 + grid_h],
            }
        )
    return slots


def validate_terminal(name: str, cfg: dict[str, Any]) -> list[str]:
    """Return human-readable issues (empty list = structurally OK)."""
    issues: list[str] = []
    cfg = apply_template(cfg)
    t = cfg.get("type")
    if t not in ("point", "bbox", "polygon"):
        issues.append(f"type must be point, bbox, or polygon (got {t!r})")
    if t == "point":
        for key in ("lat", "lon", "buffer_m"):
            if key not in cfg:
                issues.append(f"point AOI missing {key}")
    elif t == "bbox":
        if "bbox" not in cfg:
            issues.append("bbox AOI missing bbox=(min_lon, min_lat, max_lon, max_lat)")
    elif t == "polygon":
        if not cfg.get("geojson"):
            issues.append("polygon AOI missing geojson")
    mode = cfg.get("mode", "single_berth")
    if mode == "berth_slots":
        slots = cfg.get("berth_slots_px") or []
        if not slots:
            issues.append("berth_slots mode requires berth_slots_px (use picker)")
        else:
            for s in slots:
                if "box" not in s or len(s["box"]) != 4:
                    issues.append(f"slot {s.get('name')} needs box [x1,y1,x2,y2]")
    if name not in DEFAULT_KPLER_HINTS and "kpler" not in str(cfg).lower():
        issues.append("add Kpler mapping in AOI_KPLER_LOCATIONS (see DEFAULT_KPLER_HINTS)")
    return issues


def print_setup_checklist(name: str, cfg: dict[str, Any]) -> None:
    """Print what to do next when onboarding a terminal."""
    cfg_resolved = apply_template(cfg)
    template_name = cfg.get("template") or cfg_resolved.get("_template")
    print(f"=== Setup checklist: {name} ===\n")
    if template_name:
        tpl = TERMINAL_TEMPLATES[template_name]
        print(f"Template: {template_name}")
        print(f"  {tpl['description']}\n")
        print("Steps:")
        for i, step in enumerate(tpl["setup_steps"], 1):
            print(f"  {i}. {step}")
        print("\nTunable detect keys (only if Kpler compare shows mismatches):")
        for hint in tpl.get("tunable_detect_keys", []):
            print(f"  • {hint}")
    else:
        print("No template set — consider template='berth_slots_jetty' or 'single_spm'")
    print("\nValidation:")
    issues = validate_terminal(name, cfg)
    if issues:
        for msg in issues:
            print(f"  [!] {msg}")
    else:
        print("  [ok] Required fields present")
    kpler = DEFAULT_KPLER_HINTS.get(name)
    if kpler:
        print(f"\nKpler hint: {kpler}")
    overrides = cfg.get("detect") or {}
    if overrides:
        print(f"\nActive detect overrides ({len(overrides)} keys):")
        for k, v in overrides.items():
            print(f"  {k}: {v}")
    else:
        print("\nNo detect overrides — using template defaults only (recommended for new sites).")
    print()


def format_aoi_entry_python(
    name: str,
    *,
    aoi_type: str = "polygon",
    template: str = "berth_slots_jetty",
    geojson_var: str | None = None,
    slots_var: str | None = None,
    sensor: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Print a minimal AOI_CONFIG snippet for copy-paste."""
    lines = [f'    "{name}": {{']
    lines.append(f'        "type": "{aoi_type}",')
    if geojson_var:
        lines.append(f'        "geojson": {geojson_var},')
    if aoi_type == "point":
        lines.append('        "lat": 0.0,  # TODO')
        lines.append('        "lon": 0.0,  # TODO')
        lines.append('        "buffer_m": 1200,')
    lines.append(f'        "template": "{template}",')
    if slots_var:
        lines.append(f'        "berth_slots_px": {slots_var},')
    if sensor:
        lines.append(f'        "sensor": "{sensor}",')
    if extra:
        for k, v in extra.items():
            lines.append(f'        "{k}": {v!r},')
    lines.append("        # Optional after tuning: detect={\"slot_diff_abs\": 45},")
    lines.append("    },")
    return "\n".join(lines)
