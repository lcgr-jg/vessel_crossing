"""Vessel detection: Sentinel-2 blob/slot detection and Sentinel-1 SAR."""

from __future__ import annotations

import cv2
import numpy as np

SIZE_CLASSES = [
    (300, 380, "VLCC"),
    (250, 300, "Suezmax"),
    (220, 250, "Aframax"),
    (0, 220, "smaller/unclassified"),
]


def classify_length(length_m: float) -> str:
    for lo, hi, label in SIZE_CLASSES:
        if lo <= length_m < hi:
            return label
    return "unclassified"


def water_mask(rgb: np.ndarray, gray_lo: int = 8, gray_hi: int = 80) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return ((gray > gray_lo) & (gray < gray_hi)).astype(np.uint8)


def ndwi_water_mask(ndwi: np.ndarray, min_ndwi: float = 0.5) -> np.ndarray:
    """McFeeters NDWI water mask: (B03 - B08) / (B03 + B08) >= min_ndwi."""
    finite = np.isfinite(ndwi)
    return (finite & (ndwi >= min_ndwi)).astype(np.uint8)


def resolve_water_mask(
    scene: np.ndarray,
    params: dict,
    ndwi: np.ndarray | None = None,
) -> np.ndarray:
    """Prefer NDWI when enabled and available; fall back to gray RGB heuristic."""
    if params.get("use_ndwi_water") and ndwi is not None:
        return ndwi_water_mask(ndwi, params.get("ndwi_water_min", 0.5))
    return water_mask(
        scene,
        params.get("water_gray_lo", 8),
        params.get("water_gray_hi", 120),
    )


def aoi_valid_mask(rgb: np.ndarray, black_thresh: int = 5) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return (gray > black_thresh).astype(np.uint8)


def hull_color_mask(scene: np.ndarray, valid: np.ndarray) -> np.ndarray:
    r, g, b = scene[:, :, 0], scene[:, :, 1], scene[:, :, 2]
    gs = cv2.cvtColor(scene, cv2.COLOR_RGB2GRAY)
    base = valid.astype(bool)
    orange = (r > 120) & (r > g * 1.25) & (r > b * 1.15) & base
    red = (r > 80) & (r > g * 1.08) & (r > b * 1.05) & (gs < 140) & base
    olive = (g > 35) & (g > r) & (g > b) & (gs > 15) & (gs < 130) & base
    green_hull = (g > 45) & (g > r * 1.08) & (g > b * 1.05) & (gs > 25) & (gs < 150) & base
    dark_green = (g > 28) & (g > r * 1.01) & (g > b * 0.95) & (gs > 18) & (gs < 115) & base
    mask = (orange | red | olive | green_hull | dark_green).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    return mask


def build_change_mask(scene: np.ndarray, background: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
    gray_scene = cv2.cvtColor(scene, cv2.COLOR_RGB2GRAY)
    gray_bg = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
    wm = water_mask(scene, params["water_gray_lo"], params["water_gray_hi"])
    diff = cv2.absdiff(gray_scene, gray_bg)
    diff[~wm.astype(bool)] = 0

    if params.get("use_adaptive_diff"):
        vals = diff[diff > 0]
        if len(vals) == 0:
            combined = np.zeros_like(diff, dtype=np.uint8)
        else:
            t = float(np.percentile(vals, params.get("diff_percentile", 90)))
            combined = (diff >= t).astype(np.uint8) * 255
    else:
        _, combined = cv2.threshold(diff, params["diff_thresh"], 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    return combined, wm


def _contours_to_detections(contours, params: dict, resolution: float, max_blob_area: int) -> list[dict]:
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < params["min_area_px"] or area > max_blob_area:
            continue
        (_, _), (rw, rh), angle = cv2.minAreaRect(cnt)
        length_px = max(rw, rh)
        width_px = min(rw, rh)
        aspect = length_px / max(width_px, 1)
        if aspect < params["min_aspect"]:
            continue
        length_m = length_px * resolution
        width_m = width_px * resolution
        if length_m < params["min_length_m"] or length_m > params["max_length_m"]:
            continue
        m = cv2.moments(cnt)
        if not m["m00"]:
            continue
        detections.append(
            {
                "length_m": round(length_m, 1),
                "width_m": round(width_m, 1),
                "size_class": classify_length(length_m),
                "centroid_xy": (round(m["m10"] / m["m00"], 1), round(m["m01"] / m["m00"], 1)),
                "angle_deg": round(float(angle), 1),
            }
        )
    return detections


def detect_vessels(
    scene: np.ndarray,
    background: np.ndarray,
    resolution: float = 10.0,
    detect_params: dict | None = None,
) -> list[dict]:
    params = detect_params or {}
    if scene.shape != background.shape:
        raise ValueError(f"scene/background shape mismatch: {scene.shape} vs {background.shape}")

    mask, wm = build_change_mask(scene, background, params)
    water_px = max(int(wm.sum()), 1)
    max_blob_area = int(water_px * params.get("max_blob_frac", 0.15))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = _contours_to_detections(contours, params, resolution, max_blob_area)

    if params.get("use_hull_colors"):
        valid = aoi_valid_mask(scene)
        hull_contours, _ = cv2.findContours(
            hull_color_mask(scene, valid), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        hull_params = {
            **params,
            "min_aspect": params.get("hull_min_aspect", params["min_aspect"]),
            "min_length_m": params.get("hull_min_length_m", params["min_length_m"]),
        }
        hull_dets = _contours_to_detections(hull_contours, hull_params, resolution, max_blob_area)
        min_sep = params.get("hull_min_sep_px", 45)
        for hd in hull_dets:
            hx, hy = hd["centroid_xy"]
            if all(np.hypot(hx - d["centroid_xy"][0], hy - d["centroid_xy"][1]) >= min_sep for d in detections):
                detections.append(hd)

    dedupe_px = params.get("dedupe_px", 20)
    deduped = []
    for det in sorted(detections, key=lambda d: -d["length_m"]):
        cx, cy = det["centroid_xy"]
        if any(np.hypot(cx - d["centroid_xy"][0], cy - d["centroid_xy"][1]) < dedupe_px for d in deduped):
            continue
        deduped.append(det)
    return deduped


# --- Berth slots (Muajjiz) ---------------------------------------------------


def build_berth_slot_masks(shape: tuple[int, ...], slots_px: list[dict]) -> list[tuple[str, np.ndarray]]:
    """Build uint8 masks from slot boxes [x1, y1, x2, y2] in image pixel coords."""
    h, w = shape[:2]
    out = []
    for slot in slots_px:
        x1, y1, x2, y2 = slot["box"]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        mask = np.zeros((h, w), np.uint8)
        mask[y1:y2, x1:x2] = 1
        out.append((slot["name"], mask))
    return out


def _largest_blob_frac(binary: np.ndarray, valid_px: int) -> float:
    """Fraction of valid pixels in the largest connected component."""
    area, _, _, _ = _largest_blob_geom(binary, resolution=1.0)
    if valid_px <= 0 or area <= 0:
        return 0.0
    return area / valid_px


def _largest_blob_geom(
    binary: np.ndarray,
    resolution: float = 10.0,
) -> tuple[float, float, float | None, float | None]:
    """Largest connected component: (area_px, length_m, cx, cy)."""
    if binary.sum() == 0:
        return 0.0, 0.0, None, None
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return 0.0, 0.0, None, None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = float(stats[best, cv2.CC_STAT_AREA])
    component = (labels == best).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    length_m = 0.0
    if contours:
        (_, _), (rw, rh), _ = cv2.minAreaRect(contours[0])
        length_m = max(rw, rh) * resolution
    cx, cy = float(centroids[best][0]), float(centroids[best][1])
    return area, length_m, cx, cy


def _slot_valid_region(
    scene: np.ndarray,
    slot_mask: np.ndarray,
    params: dict,
    ndwi: np.ndarray | None = None,
) -> np.ndarray:
    """Pixels used for occupancy scoring inside a berth slot."""
    valid = aoi_valid_mask(scene) & slot_mask
    if params.get("slot_water_only"):
        wm = resolve_water_mask(scene, params, ndwi)
        valid = valid & wm.astype(bool)
    return valid


def _ndwi_vessel_mask(
    ndwi: np.ndarray,
    valid: np.ndarray,
    params: dict,
) -> tuple[np.ndarray, float]:
    """
    Low-NDWI anomaly mask inside the slot (ships << local water).

    Threshold = min(percentile, median - delta) so empty water stays mostly above cut.
    """
    vals = ndwi[valid.astype(bool)]
    finite = vals[np.isfinite(vals)]
    empty = np.zeros(valid.shape, dtype=np.uint8)
    if finite.size < 50:
        return empty, float("nan")
    pct = params.get("ndwi_vessel_pct", 15)
    delta = params.get("ndwi_vessel_delta", 0.05)
    thr = min(float(np.percentile(finite, pct)), float(np.median(finite)) - delta)
    vessel = (np.isfinite(ndwi) & (ndwi <= thr) & valid.astype(bool)).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    vessel = cv2.morphologyEx(vessel, cv2.MORPH_OPEN, k)
    vessel = cv2.morphologyEx(vessel, cv2.MORPH_CLOSE, k, iterations=2)
    return vessel, thr


def _slot_signal(
    scene: np.ndarray,
    background: np.ndarray,
    slot_mask: np.ndarray,
    params: dict,
    ndwi: np.ndarray | None = None,
) -> tuple[bool, dict]:
    """Decide if one berth slot is occupied (hull / diff / optional NDWI anomaly)."""
    resolution = float(params.get("_resolution", 10))
    valid = _slot_valid_region(scene, slot_mask, params, ndwi)
    valid_px = int(valid.sum())
    if valid_px < params.get("slot_min_valid_px", 80):
        return False, {"valid_px": valid_px}

    gray_s = cv2.cvtColor(scene, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(gray_s, gray_b)
    diff[~valid.astype(bool)] = 0

    # Absolute diff threshold — pier/water drift vs a monthly background shows up below ~35.
    abs_t = params.get("slot_diff_abs", 40)
    diff_binary = ((diff >= abs_t) & valid.astype(bool)).astype(np.uint8)
    diff_frac = diff_binary.sum() / valid_px
    blob_frac = _largest_blob_frac(diff_binary, valid_px)
    blob_px, blob_length_m, blob_cx, blob_cy = _largest_blob_geom(diff_binary, resolution)

    hull = (hull_color_mask(scene, valid.astype(np.uint8)) > 0).astype(np.uint8) & slot_mask
    hull = hull & valid.astype(np.uint8)
    hull_frac = hull.sum() / valid_px
    hull_blob_frac = _largest_blob_frac(hull, valid_px)
    hull_px, hull_length_m, hull_cx, hull_cy = _largest_blob_geom(hull, resolution)

    min_blob_px = params.get("slot_min_blob_px")
    min_length_m = params.get("slot_min_length_m")
    # Absolute size gates — stable on large SPM boxes where frac-of-slot stays tiny.
    abs_diff_ok = False
    if min_blob_px is not None or min_length_m is not None:
        abs_diff_ok = True
        if min_blob_px is not None:
            abs_diff_ok = abs_diff_ok and blob_px >= min_blob_px
        if min_length_m is not None:
            abs_diff_ok = abs_diff_ok and blob_length_m >= min_length_m

    abs_hull_ok = False
    if min_blob_px is not None or min_length_m is not None:
        abs_hull_ok = True
        if min_blob_px is not None:
            abs_hull_ok = abs_hull_ok and hull_px >= max(20, int(min_blob_px) // 4)
        if min_length_m is not None:
            # Painted-hull fragments are shorter than full LOA; allow ~60% of min length.
            abs_hull_ok = abs_hull_ok and hull_length_m >= 0.6 * float(min_length_m)

    ndwi_ok = False
    ndwi_px = 0.0
    ndwi_length_m = 0.0
    ndwi_cx = ndwi_cy = None
    ndwi_thr = float("nan")
    if params.get("ndwi_vessel_enabled") and ndwi is not None:
        ndwi_mask, ndwi_thr = _ndwi_vessel_mask(ndwi, valid, params)
        ndwi_px, ndwi_length_m, ndwi_cx, ndwi_cy = _largest_blob_geom(ndwi_mask, resolution)
        ndwi_size_ok = True
        if min_blob_px is not None:
            ndwi_size_ok = ndwi_size_ok and ndwi_px >= min_blob_px
        else:
            ndwi_size_ok = ndwi_size_ok and ndwi_px >= 80
        if min_length_m is not None:
            ndwi_size_ok = ndwi_size_ok and ndwi_length_m >= min_length_m
        else:
            ndwi_size_ok = ndwi_size_ok and ndwi_length_m >= 150
        # Dual evidence cuts wake/glint FPs: NDWI blob alone is not enough.
        support_px = params.get("ndwi_support_blob_px", 40)
        rgb_support = blob_px >= support_px or hull_px >= max(15, support_px // 2)
        ndwi_ok = bool(ndwi_size_ok and rgb_support)

    occupied = (
        hull_frac >= params.get("slot_hull_frac", 0.04)
        or hull_blob_frac >= params.get("slot_min_hull_blob_frac", 0.03)
        or blob_frac >= params.get("slot_min_blob_frac", 0.28)
        or (
            diff_frac >= params.get("slot_diff_frac", 0.15)
            and blob_frac >= params.get("slot_diff_blob_frac", 0.14)
        )
        or abs_diff_ok
        or abs_hull_ok
        or ndwi_ok
    )
    meta = {
        "valid_px": valid_px,
        "diff_frac": round(diff_frac, 3),
        "blob_frac": round(blob_frac, 3),
        "blob_px": int(blob_px),
        "blob_length_m": round(blob_length_m, 1),
        "hull_frac": round(hull_frac, 3),
        "hull_blob_frac": round(hull_blob_frac, 3),
        "hull_px": int(hull_px),
        "ndwi_ok": bool(ndwi_ok),
        "ndwi_px": int(ndwi_px),
        "ndwi_length_m": round(ndwi_length_m, 1),
    }
    if np.isfinite(ndwi_thr):
        meta["ndwi_thr"] = round(float(ndwi_thr), 3)
    if ndwi is not None and valid_px > 0:
        slot_ndwi = ndwi[valid.astype(bool)]
        min_ndwi = params.get("ndwi_water_min", 0.5)
        meta["mean_ndwi"] = round(float(np.nanmean(slot_ndwi)), 3)
        meta["water_frac"] = round(float(np.nanmean(slot_ndwi >= min_ndwi)), 3)
    if not occupied:
        return False, meta

    # Prefer NDWI / hull / diff centroid in that order when that path fired.
    if ndwi_ok and ndwi_cx is not None:
        cx, cy = ndwi_cx, ndwi_cy
        length_m = ndwi_length_m
    elif abs_hull_ok and hull_cx is not None:
        cx, cy = hull_cx, hull_cy
        length_m = hull_length_m
    elif (abs_diff_ok or blob_frac > 0) and blob_cx is not None:
        cx, cy = blob_cx, blob_cy
        length_m = blob_length_m
    else:
        signal = np.maximum(diff_binary, hull) * 255
        m = cv2.moments(signal)
        ys, xs = np.where(valid)
        if m["m00"]:
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        else:
            cx, cy = xs.mean(), ys.mean()
        length_m = max(xs.max() - xs.min(), ys.max() - ys.min()) * resolution

    return True, {
        "length_m": round(float(length_m), 1),
        "width_m": None,
        "size_class": classify_length(length_m),
        "centroid_xy": (round(float(cx), 1), round(float(cy), 1)),
        "angle_deg": None,
        **meta,
    }


def _apply_relative_slot_filter(signals: list[dict], params: dict) -> None:
    """
    For multi-SPM AOIs: drop weak prelim hits that are tiny vs the strongest slot on the same scene.

    Stops empty open-water slots from triggering on glint/speckle while keeping real tankers.
    """
    if not params.get("slot_use_relative"):
        return

    max_blob = max((s.get("blob_frac", 0) for s in signals), default=0.0)
    max_ndwi_px = max((s.get("ndwi_px", 0) for s in signals), default=0)
    ratio = params.get("slot_relative_blob_ratio", 0.4)
    min_blob = params.get("slot_min_blob_frac", 0.28)
    combo_blob = params.get("slot_diff_blob_frac", 0.14)
    hull_frac_min = params.get("slot_hull_frac", 0.04)
    hull_blob_min = params.get("slot_min_hull_blob_frac", 0.03)

    for sig in signals:
        if not sig.get("prelim"):
            continue
        # Absolute NDWI / size hits are not relative-filtered (SPM tankers).
        if sig.get("ndwi_ok"):
            continue
        min_blob_px = params.get("slot_min_blob_px")
        if min_blob_px is not None and sig.get("blob_px", 0) >= min_blob_px:
            continue
        hull_ok = sig.get("hull_frac", 0) >= hull_frac_min or sig.get("hull_blob_frac", 0) >= hull_blob_min
        blob = sig.get("blob_frac", 0)
        strong_blob = blob >= min_blob
        relative_ok = max_blob > 0 and blob >= ratio * max_blob and blob >= combo_blob
        ndwi_relative = max_ndwi_px > 0 and sig.get("ndwi_px", 0) >= ratio * max_ndwi_px
        if hull_ok or strong_blob or relative_ok or ndwi_relative:
            continue
        sig["prelim"] = False
        sig["rejected_by"] = "relative_filter"


def scene_slot_quality(
    scene: np.ndarray,
    slots_px: list[dict],
    params: dict | None = None,
) -> dict:
    """
    Cheap AOI quality gate. Low valid coverage → skip (don't call empty/occupied).

    Bright-ship SCL holes and heavy cloud both shrink aoi_valid_mask.
    """
    params = params or {}
    valid = aoi_valid_mask(scene)
    slot_masks = build_berth_slot_masks(scene.shape, slots_px)
    slot_px = 0
    slot_valid = 0
    for _, sm in slot_masks:
        slot_px += int(sm.sum())
        slot_valid += int((sm.astype(bool) & valid.astype(bool)).sum())
    frac = (slot_valid / slot_px) if slot_px else 0.0
    min_frac = params.get("scene_min_valid_frac", 0.55)
    ok = frac >= min_frac and slot_valid >= params.get("slot_min_valid_px", 80)
    return {
        "quality": "ok" if ok else "skip",
        "slot_valid_frac": round(frac, 3),
        "slot_valid_px": int(slot_valid),
        "slot_px": int(slot_px),
    }


def _s1_dets_in_slots(
    s1_dets: list[dict],
    shape: tuple[int, ...],
    slots_px: list[dict],
    resolution: float,
) -> list[dict]:
    """Map free S1 blob detections into berth slots by centroid."""
    out = []
    for name, slot_mask in build_berth_slot_masks(shape, slots_px):
        hit = None
        for det in s1_dets:
            cx, cy = det["centroid_xy"]
            x, y = int(round(cx)), int(round(cy))
            if 0 <= y < slot_mask.shape[0] and 0 <= x < slot_mask.shape[1] and slot_mask[y, x]:
                if hit is None or det["length_m"] > hit["length_m"]:
                    hit = det
        if hit is None:
            continue
        out.append(
            {
                "berth": name,
                "length_m": hit["length_m"],
                "width_m": hit.get("width_m"),
                "size_class": hit["size_class"],
                "centroid_xy": hit["centroid_xy"],
                "angle_deg": hit.get("angle_deg"),
                "sensor": "s1",
                "quality": "ok",
            }
        )
    return out


def _merge_slot_detections(s2_dets: list[dict], s1_slot_dets: list[dict]) -> list[dict]:
    """Union by berth: prefer S2 row, else S1; tag sensor as s2_s1 when both fire."""
    by_berth: dict[str, dict] = {}
    for d in s2_dets:
        berth = d.get("berth")
        if berth is None:
            continue
        row = {**d, "sensor": d.get("sensor", "s2"), "quality": d.get("quality", "ok")}
        by_berth[berth] = row
    for d in s1_slot_dets:
        berth = d["berth"]
        if berth in by_berth:
            by_berth[berth]["sensor"] = "s2_s1"
        else:
            by_berth[berth] = d
    return list(by_berth.values())


def diagnose_berth_slots(
    scene: np.ndarray,
    background: np.ndarray,
    slots_px: list[dict],
    resolution: float = 10.0,
    detect_params: dict | None = None,
    ndwi: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-slot signal breakdown for tuning (import pandas in notebook if needed)."""
    import pandas as pd

    params = {**(detect_params or {}), "_resolution": resolution}
    quality = scene_slot_quality(scene, slots_px, params)
    rows = []
    for name, slot_mask in build_berth_slot_masks(scene.shape, slots_px):
        prelim, meta = _slot_signal(scene, background, slot_mask, params, ndwi)
        rows.append({"berth": name, "prelim": prelim, **meta, **quality})
    signals = rows
    if quality["quality"] == "skip":
        for row in signals:
            row["occupied"] = False
            row["prelim"] = False
            row["rejected_by"] = "scene_quality"
        return pd.DataFrame(signals)
    _apply_relative_slot_filter(signals, params)
    for row in signals:
        row["occupied"] = row.pop("prelim", False)
    return pd.DataFrame(signals)


def detect_berth_slots(
    scene: np.ndarray,
    background: np.ndarray,
    slots_px: list[dict],
    resolution: float = 10.0,
    detect_params: dict | None = None,
    ndwi: np.ndarray | None = None,
    s1_vv_db: np.ndarray | None = None,
) -> list[dict]:
    """One detection row per occupied berth slot (optional S1 second vote)."""
    params = {**(detect_params or {}), "_resolution": resolution}
    if scene.shape != background.shape:
        raise ValueError(f"scene/background shape mismatch: {scene.shape} vs {background.shape}")

    quality = scene_slot_quality(scene, slots_px, params)
    if quality["quality"] == "skip":
        # Don't treat as empty — downstream should exclude from Kpler FP/FN scoring.
        return [
            {
                "berth": None,
                "length_m": None,
                "width_m": None,
                "size_class": "scene_skip",
                "centroid_xy": None,
                "angle_deg": None,
                "sensor": "s2",
                "quality": "skip",
                "vessel_count_override": None,
                **quality,
            }
        ]

    signals: list[dict] = []
    for name, slot_mask in build_berth_slot_masks(scene.shape, slots_px):
        prelim, meta = _slot_signal(scene, background, slot_mask, params, ndwi)
        signals.append({"berth": name, "prelim": prelim, **meta})

    _apply_relative_slot_filter(signals, params)

    detections = []
    for sig in signals:
        if not sig.get("prelim"):
            continue
        row = {k: v for k, v in sig.items() if k not in ("prelim", "rejected_by")}
        row["sensor"] = "s2"
        row["quality"] = "ok"
        detections.append(row)

    if s1_vv_db is not None and params.get("s1_slot_vote", True):
        s1_dets = detect_s1_vessels(s1_vv_db, resolution, params)
        s1_slots = _s1_dets_in_slots(s1_dets, scene.shape, slots_px, resolution)
        detections = _merge_slot_detections(detections, s1_slots)

    return detections


# --- Sentinel-1 SAR ----------------------------------------------------------


def speckle_filter(arr: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.medianBlur(arr.astype(np.float32), ksize)


def s1_water_mask(vv_db: np.ndarray, max_db: float = -14.0) -> np.ndarray:
    """Open water in SAR: low VV backscatter (dB)."""
    finite = np.isfinite(vv_db)
    return (finite & (vv_db < max_db)).astype(np.uint8)


def detect_s1_vessels(vv_db: np.ndarray, resolution: float = 10.0, detect_params: dict | None = None) -> list[dict]:
    """
    Detect bright VV targets on dark water. Works best for open-water SPMs (Zirku).
    Uses adaptive percentile threshold on water pixels when s1_adaptive_percentile is set.
    """
    params = detect_params or {}
    vv = speckle_filter(vv_db, params.get("s1_speckle_ksize", 5))
    wm = s1_water_mask(vv, params.get("s1_water_max_db", -14.0))
    water_px = max(int(wm.sum()), 1)
    water_vals = vv[wm.astype(bool)]

    if params.get("s1_adaptive_percentile") and len(water_vals) > 50:
        thresh = float(np.percentile(water_vals, params["s1_adaptive_percentile"]))
    else:
        thresh = params.get("s1_vv_thresh_db", -11.0)

    bright = ((vv >= thresh) & wm.astype(bool)).astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, k)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k, iterations=2)

    max_blob = int(water_px * params.get("max_blob_frac", 0.08))
    s1_params = {
        "min_area_px": params.get("s1_min_area_px", 4),
        "min_aspect": params.get("s1_min_aspect", 1.2),
        "min_length_m": params.get("s1_min_length_m", 40),
        "max_length_m": params.get("max_length_m", 400),
    }
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = _contours_to_detections(contours, s1_params, resolution, max_blob)

    dedupe_px = params.get("dedupe_px", 25)
    deduped = []
    for det in sorted(detections, key=lambda d: -d["length_m"]):
        cx, cy = det["centroid_xy"]
        if any(np.hypot(cx - d["centroid_xy"][0], cy - d["centroid_xy"][1]) < dedupe_px for d in deduped):
            continue
        det["sensor"] = "s1"
        deduped.append(det)
    return deduped


def detect_for_aoi(
    scene,
    background,
    aoi_cfg: dict,
    resolution: float,
    detect_params: dict | None = None,
    s1_vv_db: np.ndarray | None = None,
    ndwi: np.ndarray | None = None,
) -> list[dict]:
    """Route to blob, berth-slot, or SAR detection based on AOI config."""
    mode = aoi_cfg.get("mode", "single_berth")
    sensor = aoi_cfg.get("sensor", "s2")

    if sensor == "s1" and s1_vv_db is not None:
        return detect_s1_vessels(s1_vv_db, resolution, detect_params)

    if mode == "berth_slots":
        params = detect_params or {}
        if aoi_cfg.get("detector") == "yolo" or params.get("use_yolo"):
            from yolo_ship_detect import detect_berth_slots_yolo

            return detect_berth_slots_yolo(
                scene,
                aoi_cfg["berth_slots_px"],
                resolution,
                params,
            )
        use_s1 = sensor in ("s2_s1", "s1") and s1_vv_db is not None
        return detect_berth_slots(
            scene,
            background,
            aoi_cfg["berth_slots_px"],
            resolution,
            detect_params,
            ndwi=ndwi,
            s1_vv_db=s1_vv_db if use_s1 else None,
        )

    dets = detect_vessels(scene, background, resolution, detect_params)
    if sensor == "s2_s1" and not dets and s1_vv_db is not None:
        return detect_s1_vessels(s1_vv_db, resolution, detect_params)
    for d in dets:
        d.setdefault("sensor", "s2")
        d.setdefault("quality", "ok")
    return dets
