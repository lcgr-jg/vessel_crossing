"""
SNAP CFAR vs custom S1 detector — isolated comparison (do not wire into detect_for_aoi yet).

Isolates three failure modes of vessel_detect.detect_s1_vessels:
  1. median blur vs real Speckle-Filter (Lee / Refined Lee / Gamma Map)
  2. global water percentile vs local CFAR (guard + background window)
  3. VV-only vs dual-pol VV+VH (VH is available via Process API without SNAP)

Usage (stepwise):
  python snap_cfar_compare.py catalogue
  python snap_cfar_compare.py download --product-id <uuid>   # asks; ~0.7–1.9 GB
  python snap_cfar_compare.py run-local --date 2026-07-22    # A + B on Process API VV
  python snap_cfar_compare.py run-snap --safe path/to.SAFE   # C via gpt graph
  python snap_cfar_compare.py score --date 2026-07-22
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Reuse existing detector without modifying it
from vessel_detect import classify_length, detect_s1_vessels, s1_water_mask, speckle_filter

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / ".cache" / "sentinel"
GRD_DIR = HERE / ".cache" / "s1_grd"
GRAPH_PATH = HERE / "snap_ship_detect_graph.xml"
GPT_EXE = Path(
    os.environ.get(
        "SNAP_GPT",
        r"C:\Users\luiscarlos.gaitan\AppData\Local\Programs\esa-snap\bin\gpt.exe",
    )
)

# Zirku SPM A (single_spm / s2_s1) — same point+buffer as notebook AOI_CONFIG
ZIRKU = {
    "aoi": "zirku_spm_a",
    "lat": 25.00833,
    "lon": 52.98333,
    "buffer_m": 1200,
}

# ---------------------------------------------------------------------------
# Ground truth — fill after eyeballing. Positions in AOI pixel coords for the
# Process API array (same grid as fetch_s1_vv), OR lon/lat for SNAP vectors.
# ---------------------------------------------------------------------------
GROUND_TRUTH: dict[tuple[str, str], list[dict[str, Any]]] = {
    # Example (replace with your confirmed vessels):
    # ("zirku_spm_a", "2026-07-22"): [
    #     {"centroid_xy": (120.0, 130.0), "length_m": 300, "note": "VLCC at SPM"},
    # ],
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def match_detections(
    preds: list[dict],
    truths: list[dict],
    max_dist_px: float = 30.0,
) -> dict[str, Any]:
    """Greedy nearest-neighbour match on centroid_xy; returns P/R/count stats."""
    remaining = list(range(len(truths)))
    tp, pairs = 0, []
    for i, p in enumerate(preds):
        if "centroid_xy" not in p or p["centroid_xy"] is None:
            continue
        px, py = p["centroid_xy"]
        best_j, best_d = None, max_dist_px
        for j in remaining:
            tx, ty = truths[j]["centroid_xy"]
            d = math.hypot(px - tx, py - ty)
            if d < best_d:
                best_d, best_j = d, j
        if best_j is not None:
            remaining.remove(best_j)
            tp += 1
            pairs.append((i, best_j, best_d))
    fp = len(preds) - tp
    fn = len(remaining)
    prec = tp / len(preds) if preds else float("nan")
    rec = tp / len(truths) if truths else float("nan")
    return {
        "n_pred": len(preds),
        "n_truth": len(truths),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": None if prec != prec else round(prec, 3),
        "recall": None if rec != rec else round(rec, 3),
        "count_error": len(preds) - len(truths),
        "pairs": pairs,
    }


def score_method(
    aoi: str,
    date: str,
    preds: list[dict],
    max_dist_px: float = 30.0,
) -> dict[str, Any]:
    truths = GROUND_TRUTH.get((aoi, date), [])
    out = match_detections(preds, truths, max_dist_px=max_dist_px)
    out["aoi"] = aoi
    out["date"] = date
    out["has_ground_truth"] = bool(truths)
    return out


# ---------------------------------------------------------------------------
# Method A: existing global-percentile detector (thin wrapper)
# ---------------------------------------------------------------------------

def run_existing_s1(vv_db: np.ndarray, resolution: float = 10.0, detect_params: dict | None = None) -> list[dict]:
    params = {
        "s1_water_max_db": -12.0,
        "s1_adaptive_percentile": 99.5,
        "s1_min_area_px": 4,
        "s1_min_length_m": 40,
        **(detect_params or {}),
    }
    dets = detect_s1_vessels(vv_db, resolution=resolution, detect_params=params)
    for d in dets:
        d["method"] = "existing_global_pct"
    return dets


# ---------------------------------------------------------------------------
# Method B: local CFAR on the SAME Process API VV array (isolates limitation #2)
# ---------------------------------------------------------------------------

def _sliding_cfar_mask(
    vv: np.ndarray,
    water: np.ndarray,
    target_px: int = 5,
    guard_px: int = 25,
    bg_px: int = 40,
    pfa: float = 1e-6,
) -> np.ndarray:
    """
    Cell-averaging CFAR: for each water pixel, compare to mean/std of a
    background ring (bg window minus guard), excluding the target cell.

    Not identical to SNAP's AdaptiveThresholding (meters-based windows,
    distribution assumptions), but answers: does *local* thresholding alone
    change results vs the global percentile, on the same VV data?
    """
    # Work on a finite float copy; land / nodata stay False in output
    x = np.nan_to_num(vv.astype(np.float64), nan=-999.0)
    h, w = x.shape
    half_bg = bg_px // 2
    half_guard = guard_px // 2
    half_tgt = max(target_px // 2, 1)

    # Gaussian approx: threshold = mu + k * sigma, k from one-sided PFA
    # erfcinv(2*pfa) * sqrt(2) ≈ inverse CDF of N(0,1) for upper tail
    # Use a simple chi approx via erfinv if available; else fixed k for pfa~1e-6
    try:
        from math import erfcinv

        k = math.sqrt(2.0) * erfcinv(2.0 * pfa)
    except Exception:
        k = 4.75  # ~1e-6 one-sided for Gaussian

    out = np.zeros((h, w), dtype=np.uint8)
    ys, xs = np.where(water.astype(bool))
    for y, x0 in zip(ys, xs):
        y0b, y1b = max(0, y - half_bg), min(h, y + half_bg + 1)
        x0b, x1b = max(0, x0 - half_bg), min(w, x0 + half_bg + 1)
        patch = x[y0b:y1b, x0b:x1b]
        # mask guard+target out of background
        local = patch.copy()
        gy0 = max(0, (y - half_guard) - y0b)
        gy1 = min(local.shape[0], (y + half_guard + 1) - y0b)
        gx0 = max(0, (x0 - half_guard) - x0b)
        gx1 = min(local.shape[1], (x0 + half_guard + 1) - x0b)
        local[gy0:gy1, gx0:gx1] = np.nan
        bg = local[np.isfinite(local) & (local > -900)]
        if bg.size < 20:
            continue
        mu, sigma = float(bg.mean()), float(bg.std())
        if sigma < 1e-6:
            sigma = 1e-6
        # target cell mean
        ty0, ty1 = max(0, y - half_tgt), min(h, y + half_tgt + 1)
        tx0, tx1 = max(0, x0 - half_tgt), min(w, x0 + half_tgt + 1)
        tval = float(x[ty0:ty1, tx0:tx1].mean())
        if tval >= mu + k * sigma:
            out[y, x0] = 255
    return out


def detect_local_cfar(
    vv_db: np.ndarray,
    resolution: float = 10.0,
    detect_params: dict | None = None,
) -> list[dict]:
    """Local CFAR on Process API VV — same input as detect_s1_vessels."""
    params = detect_params or {}
    # Keep the same cheap despeckle as production so the *only* change is threshold locality
    vv = speckle_filter(vv_db, params.get("s1_speckle_ksize", 5))
    wm = s1_water_mask(vv, params.get("s1_water_max_db", -12.0))

    # Windows in meters → pixels (Process API is ~10 m/px for this AOI)
    res = resolution
    target_m = params.get("cfar_target_m", 50)
    guard_m = params.get("cfar_guard_m", 500)
    bg_m = params.get("cfar_bg_m", 800)
    # Guard/bg of 500–800 m on a ~2.4 km AOI is huge; clamp to useful fractions
    h, w = vv.shape
    max_win = max(min(h, w) - 2, 5)
    target_px = max(int(round(target_m / res)), 3)
    guard_px = min(max(int(round(guard_m / res)), target_px + 2), max_win // 2)
    bg_px = min(max(int(round(bg_m / res)), guard_px + 4), max_win)

    # Dense per-pixel loop is slow; for small AOIs (~240^2) it is acceptable.
    # Optional stride for speed during iteration:
    stride = int(params.get("cfar_stride", 1))
    if stride > 1:
        # Coarser test grid then dilate — approximate, for debugging only
        bright = np.zeros_like(wm, dtype=np.uint8)
        wm_s = wm[::stride, ::stride]
        vv_s = vv[::stride, ::stride]
        m = _sliding_cfar_mask(
            vv_s,
            wm_s,
            target_px=max(target_px // stride, 1),
            guard_px=max(guard_px // stride, 2),
            bg_px=max(bg_px // stride, 4),
            pfa=params.get("cfar_pfa", 1e-6),
        )
        bright[::stride, ::stride] = m
        bright = cv2.dilate(bright, np.ones((stride, stride), np.uint8))
    else:
        bright = _sliding_cfar_mask(
            vv,
            wm,
            target_px=target_px,
            guard_px=guard_px,
            bg_px=bg_px,
            pfa=params.get("cfar_pfa", 1e-6),
        )

    bright = cv2.bitwise_and(bright, wm * 255)
    k = np.ones((3, 3), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, k)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k, iterations=2)

    water_px = max(int(wm.sum()), 1)
    max_blob = int(water_px * params.get("max_blob_frac", 0.08))
    s1_params = {
        "min_area_px": params.get("s1_min_area_px", 4),
        "min_aspect": params.get("s1_min_aspect", 1.2),
        "min_length_m": params.get("s1_min_length_m", 40),
        "max_length_m": params.get("max_length_m", 400),
    }
    from vessel_detect import _contours_to_detections

    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = _contours_to_detections(contours, s1_params, resolution, max_blob)
    for d in dets:
        d["sensor"] = "s1"
        d["method"] = "local_cfar_vv"
    return dets


# ---------------------------------------------------------------------------
# Method C: SNAP AdaptiveThresholding graph
# ---------------------------------------------------------------------------

def zirku_bbox_wgs84() -> tuple[float, float, float, float]:
    lat, lon, meters = ZIRKU["lat"], ZIRKU["lon"], ZIRKU["buffer_m"]
    dlat = meters / 111_000
    dlon = meters / (111_000 * math.cos(math.radians(lat)))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def zirku_wkt_polygon() -> str:
    min_lon, min_lat, max_lon, max_lat = zirku_bbox_wgs84()
    return (
        f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, "
        f"{max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
    )


def write_snap_graph(path: Path = GRAPH_PATH, use_vh: bool = False) -> Path:
    """
    Read -> orbit -> border noise -> calibrate -> Refined Lee -> land-sea mask
    -> AdaptiveThresholding (CFAR) -> Object-Discrimination -> write vectors.

    Operator names verified via `gpt -h` on this machine (2026-08-05).
    """
    pols = "VV,VH" if use_vh else "VV"
    # Calibration selectedPolarisations is comma-separated; bands follow sigma0_XX
    geo = zirku_wkt_polygon()
    # pfa=6.5 is SNAP's default (−log10 style); windows in meters match SNAP defaults
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<graph id="snap_cfar_ship_detect">
  <version>1.0</version>

  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>${{infile}}</file>
    </parameters>
  </node>

  <node id="Subset">
    <operator>Subset</operator>
    <sources>
      <sourceProduct refid="Read"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <geoRegion>{geo}</geoRegion>
      <copyMetadata>true</copyMetadata>
    </parameters>
  </node>

  <node id="Apply-Orbit-File">
    <operator>Apply-Orbit-File</operator>
    <sources>
      <sourceProduct refid="Subset"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <orbitType>Sentinel Precise (Auto Download)</orbitType>
      <polyDegree>3</polyDegree>
      <continueOnFail>true</continueOnFail>
    </parameters>
  </node>

  <node id="Remove-GRD-Border-Noise">
    <operator>Remove-GRD-Border-Noise</operator>
    <sources>
      <sourceProduct refid="Apply-Orbit-File"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <selectedPolarisations>{pols}</selectedPolarisations>
      <borderLimit>500</borderLimit>
      <trimThreshold>0.5</trimThreshold>
    </parameters>
  </node>

  <node id="Calibration">
    <operator>Calibration</operator>
    <sources>
      <sourceProduct refid="Remove-GRD-Border-Noise"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <selectedPolarisations>{pols}</selectedPolarisations>
      <outputSigmaBand>true</outputSigmaBand>
      <outputImageScaleInDb>false</outputImageScaleInDb>
    </parameters>
  </node>

  <node id="Speckle-Filter">
    <operator>Speckle-Filter</operator>
    <sources>
      <sourceProduct refid="Calibration"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <filter>Refined Lee</filter>
      <filterSizeX>5</filterSizeX>
      <filterSizeY>5</filterSizeY>
    </parameters>
  </node>

  <node id="Land-Sea-Mask">
    <operator>Land-Sea-Mask</operator>
    <sources>
      <sourceProduct refid="Speckle-Filter"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <landMask>true</landMask>
      <useSRTM>true</useSRTM>
      <shorelineExtension>10</shorelineExtension>
    </parameters>
  </node>

  <node id="AdaptiveThresholding">
    <operator>AdaptiveThresholding</operator>
    <sources>
      <sourceProduct refid="Land-Sea-Mask"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <targetWindowSizeInMeter>50</targetWindowSizeInMeter>
      <guardWindowSizeInMeter>500.0</guardWindowSizeInMeter>
      <backgroundWindowSizeInMeter>800.0</backgroundWindowSizeInMeter>
      <pfa>6.5</pfa>
      <estimateBackground>false</estimateBackground>
    </parameters>
  </node>

  <node id="Object-Discrimination">
    <operator>Object-Discrimination</operator>
    <sources>
      <sourceProduct refid="AdaptiveThresholding"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <minTargetSizeInMeter>40.0</minTargetSizeInMeter>
      <maxTargetSizeInMeter>400.0</maxTargetSizeInMeter>
    </parameters>
  </node>

  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="Object-Discrimination"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>${{outfile}}</file>
      <formatName>GeoTIFF</formatName>
    </parameters>
  </node>
</graph>
"""
    path.write_text(xml, encoding="utf-8")
    return path


def run_snap_gpt(safe_path: Path, out_tif: Path, use_vh: bool = False) -> Path:
    if not GPT_EXE.exists():
        raise FileNotFoundError(f"gpt.exe not found at {GPT_EXE}")
    graph = write_snap_graph(use_vh=use_vh)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(GPT_EXE),
        str(graph),
        f"-Pinfile={safe_path}",
        f"-Poutfile={out_tif}",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_tif


def detections_from_snap_mask(
    mask_or_product: np.ndarray,
    resolution: float = 10.0,
    min_length_m: float = 40.0,
    max_length_m: float = 400.0,
) -> list[dict]:
    """
    Convert SNAP ship-detection band (bright blobs) into the same dict schema
    as detect_s1_vessels. FLAG: SNAP pixel grid ≠ Process API AOI grid — centroids
    are in SNAP-subset pixel coords unless reprojected.
    """
    band = mask_or_product
    if band.ndim == 3:
        band = band[..., 0]
    # Ship bits are typically >0 on detection band
    binary = ((band > 0) & np.isfinite(band)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4:
            continue
        (_, _), (rw, rh), angle = cv2.minAreaRect(cnt)
        length_px = max(rw, rh)
        width_px = min(rw, rh)
        length_m = length_px * resolution
        width_m = width_px * resolution
        if length_m < min_length_m or length_m > max_length_m:
            continue
        m = cv2.moments(cnt)
        if not m["m00"]:
            continue
        dets.append(
            {
                "length_m": round(length_m, 1),
                "width_m": round(width_m, 1),
                "size_class": classify_length(length_m),
                "centroid_xy": (round(m["m10"] / m["m00"], 1), round(m["m01"] / m["m00"], 1)),
                "angle_deg": round(float(angle), 1),
                "sensor": "s1_snap",
                "method": "snap_cfar",
                "coord_frame": "snap_subset_pixels",  # not Process API pixels
            }
        )
    return dets


# ---------------------------------------------------------------------------
# CDSE catalogue / download (download requires explicit CLI + credentials)
# ---------------------------------------------------------------------------

CDSE_ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"


def catalogue_zirku_grd(start: str = "2026-07-01", end: str = "2026-08-06", cog: bool = False) -> list[dict]:
    """Public OData search — no auth. All IW GRDH over Zirku are 1SDV = VV+VH."""
    min_lon, min_lat, max_lon, max_lat = zirku_bbox_wgs84()
    poly = (
        f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},"
        f"{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"
    )
    name_clause = "contains(Name,'IW_GRDH_1SDV')"
    if cog:
        name_clause += " and contains(Name,'_COG')"
    else:
        name_clause += " and not contains(Name,'_COG')"
    filt = (
        f"Collection/Name eq 'SENTINEL-1' and {name_clause} and "
        f"ContentDate/Start ge {start}T00:00:00.000Z and "
        f"ContentDate/Start lt {end}T00:00:00.000Z and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{poly}')"
    )
    qs = urllib.parse.urlencode(
        {
            "$filter": filt,
            "$orderby": "ContentDate/Start asc",
            "$top": "40",
            "$select": "Id,Name,ContentDate,ContentLength",
        }
    )
    with urllib.request.urlopen(f"{CDSE_ODATA}?{qs}", timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    rows = []
    for p in data.get("value", []):
        name = p["Name"]
        rows.append(
            {
                "id": p["Id"],
                "name": name,
                "date": p["ContentDate"]["Start"][:10],
                "size_gb": round(p["ContentLength"] / (1024**3), 2),
                # 1SDV = dual VV+VH (standard IW dual-pol)
                "polarisation": "VV+VH" if "1SDV" in name else "unknown",
                "format": "COG" if "_COG" in name else "SAFE",
            }
        )
    return rows


def print_catalogue(rows: list[dict]) -> None:
    print(f"{'date':10} {'GB':>5} {'pol':7} {'fmt':5}  id  name")
    for r in rows:
        print(
            f"{r['date']:10} {r['size_gb']:5.2f} {r['polarisation']:7} {r['format']:5}  "
            f"{r['id']}  {r['name']}"
        )
    print(
        "\nVH note: every listed product is 1SDV → VV and VH are BOTH inside the same GRD.\n"
        "VH is also fetchable via Sentinel Hub Process API (no SAFE download) with bands "
        "['VV','VH'] — so dual-pol gains must not be credited to SNAP alone."
    )


def download_grd(product_id: str, dest_dir: Path = GRD_DIR, confirm: bool = True) -> Path:
    """
    Download a CDSE product ZIP. Requires CDSE_USER + CDSE_PASSWORD (or
    CDSE_ACCESS_TOKEN). Refuses to start unless confirm=True and user types YES.
    """
    rows = catalogue_zirku_grd()
    meta = next((r for r in rows if r["id"] == product_id), None)
    if meta is None:
        # still allow download if user passes an id from a wider search
        meta = {"name": product_id, "size_gb": "?", "date": "?", "polarisation": "?"}
    print("About to download:")
    print(f"  id:   {product_id}")
    print(f"  name: {meta.get('name')}")
    print(f"  date: {meta.get('date')}  size≈{meta.get('size_gb')} GB  pol={meta.get('polarisation')}")
    if confirm:
        ans = input("Type YES to download this file: ").strip()
        if ans != "YES":
            print("Aborted.")
            sys.exit(1)

    token = os.environ.get("CDSE_ACCESS_TOKEN")
    if not token:
        user = os.environ.get("CDSE_USER") or os.environ.get("CDSE_USERNAME")
        password = os.environ.get("CDSE_PASSWORD")
        if not user or not password:
            raise RuntimeError(
                "Set CDSE_ACCESS_TOKEN or CDSE_USER/CDSE_PASSWORD before download."
            )
        token = _cdse_token(user, password)

    dest_dir.mkdir(parents=True, exist_ok=True)
    out_zip = dest_dir / f"{meta.get('name', product_id)}.zip"
    url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    print(f"Downloading to {out_zip} ...")
    with urllib.request.urlopen(req, timeout=600) as resp, open(out_zip, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    print(f"Done: {out_zip} ({out_zip.stat().st_size / 1e9:.2f} GB)")
    return out_zip


def _cdse_token(user: str, password: str) -> str:
    body = urllib.parse.urlencode(
        {
            "client_id": "cdse-public",
            "username": user,
            "password": password,
            "grant_type": "password",
        }
    ).encode()
    req = urllib.request.Request(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())["access_token"]


# ---------------------------------------------------------------------------
# Side-by-side table
# ---------------------------------------------------------------------------

def side_by_side_table(
    results: dict[str, dict[str, Any]],
) -> str:
    """results: method_name -> score dict (and optional 'detections')."""
    headers = ["method", "n_pred", "n_truth", "tp", "fp", "fn", "precision", "recall", "count_err"]
    lines = [" | ".join(headers), "-|-".join("-" * len(h) for h in headers)]
    for method, sc in results.items():
        row = [
            method,
            str(sc.get("n_pred", "")),
            str(sc.get("n_truth", "")),
            str(sc.get("tp", "")),
            str(sc.get("fp", "")),
            str(sc.get("fn", "")),
            str(sc.get("precision", "")),
            str(sc.get("recall", "")),
            str(sc.get("count_error", "")),
        ]
        lines.append(" | ".join(row))
    return "\n".join(lines)


def comparison_caveats() -> str:
    return """
COMPARISON CAVEATS (read before trusting a 'SNAP win'):
1. Coord frames: existing/local_cfar use Process API AOI pixels; SNAP uses its own
   subset grid (and optionally geo vectors). Do not match centroids across methods
   until one side is reprojected into the other frame.
2. Same calendar date ≠ same acquisition: S2 scene time ≠ S1 pass (Zirku S1 is
   ~02:22 or ~14:32 UTC). Vessel may move between them; GT must be sensor-specific.
3. Process API VV is an orthorectified/mosaicked rendering; SNAP GRD is a single
   SAFE with full metadata. Radiometry/geolocation will not be pixel-identical.
4. VH: bundled in every 1SDV GRD AND available via Process API — test VH on the
   existing path before attributing dual-pol gains to SNAP.
5. Local CFAR here keeps medianBlur despeckle on purpose (isolate #2). SNAP uses
   Refined Lee (#1) + CFAR (#2) + optional VH (#3) together unless ablated.
6. build_timeseries currently never passes s1_vv into detect_for_aoi for s2_s1,
   so production may not be exercising detect_s1_vessels as a fallback at all.
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_catalogue(args: argparse.Namespace) -> None:
    rows = catalogue_zirku_grd(start=args.start, end=args.end, cog=args.cog)
    print_catalogue(rows)
    if not args.cog:
        print("\nSmaller COG twins exist (~0.7–0.9 GB). Re-run with --cog if you want those IDs.")
        print("Prefer classic SAFE for SNAP unless you've confirmed COG readability.")


def cmd_download(args: argparse.Namespace) -> None:
    download_grd(args.product_id, confirm=not args.yes)


def cmd_write_graph(_: argparse.Namespace) -> None:
    p = write_snap_graph(use_vh=False)
    print(f"Wrote {p}")
    p2 = write_snap_graph(HERE / "snap_ship_detect_graph_vvvh.xml", use_vh=True)
    print(f"Wrote {p2} (VV+VH calibration)")


def cmd_verify_snappy(_: argparse.Namespace) -> None:
    import esa_snappy
    from esa_snappy import ProductIO

    print("esa_snappy OK:", esa_snappy)
    print("ProductIO OK:", ProductIO)


def cmd_run_local(args: argparse.Namespace) -> None:
    """Run A (existing) + B (local CFAR) on a cached or provided VV .npy."""
    aoi, date = ZIRKU["aoi"], args.date
    npy = Path(args.vv) if args.vv else CACHE_DIR / "s1" / aoi / f"{date}.npy"
    if not npy.exists():
        print(
            f"Missing {npy}\n"
            "Fetch VV first from the notebook (fetch_s1_vv) or pass --vv path.npy.\n"
            "Not fetching here to avoid silent Process API charges."
        )
        sys.exit(2)
    vv = np.load(npy)
    existing = run_existing_s1(vv)
    # stride=2 for a faster first look on ~240^2 AOI; set --full for stride=1
    cfar_params = {"cfar_stride": 1 if args.full else 2}
    local = detect_local_cfar(vv, detect_params=cfar_params)
    print(f"existing_global_pct: {len(existing)} dets")
    for d in existing:
        print(" ", d)
    print(f"local_cfar_vv:       {len(local)} dets")
    for d in local:
        print(" ", d)
    scores = {
        "existing_global_pct": score_method(aoi, date, existing),
        "local_cfar_vv": score_method(aoi, date, local),
    }
    print("\n" + side_by_side_table(scores))
    if not GROUND_TRUTH.get((aoi, date)):
        print(f"\nNo GROUND_TRUTH entry for ({aoi!r}, {date!r}) — fill GROUND_TRUTH in this file.")
    print(comparison_caveats())


def cmd_run_snap(args: argparse.Namespace) -> None:
    safe = Path(args.safe)
    out = Path(args.out) if args.out else HERE / ".cache" / "snap_out" / f"{safe.stem}_ships.tif"
    run_snap_gpt(safe, out, use_vh=args.vh)
    # Load with tifffile/rasterio/gdal if available; else esa_snappy
    try:
        import tifffile

        arr = tifffile.imread(out)
    except Exception:
        from esa_snappy import ProductIO

        prod = ProductIO.readProduct(str(out))
        band = prod.getBandAt(0)
        w, h = band.getRasterWidth(), band.getRasterHeight()
        arr = np.zeros(w * h, np.float32)
        band.readPixels(0, 0, w, h, arr)
        arr = arr.reshape(h, w)
        prod.dispose()
    dets = detections_from_snap_mask(arr)
    print(f"snap_cfar: {len(dets)} dets  (coord_frame=snap_subset_pixels)")
    for d in dets:
        print(" ", d)
    print(comparison_caveats())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("catalogue", help="List IW GRD products over Zirku (public OData)")
    c.add_argument("--start", default="2026-07-01")
    c.add_argument("--end", default="2026-08-06")
    c.add_argument("--cog", action="store_true", help="List COG products (~half size)")
    c.set_defaults(func=cmd_catalogue)

    d = sub.add_parser("download", help="Download one GRD (requires YES + CDSE creds)")
    d.add_argument("--product-id", required=True)
    d.add_argument("--yes", action="store_true", help="Skip interactive YES (still needs creds)")
    d.set_defaults(func=cmd_download)

    g = sub.add_parser("write-graph", help="Write snap_ship_detect_graph.xml")
    g.set_defaults(func=cmd_write_graph)

    v = sub.add_parser("verify-snappy", help="Import esa_snappy / ProductIO")
    v.set_defaults(func=cmd_verify_snappy)

    r = sub.add_parser("run-local", help="A+B on cached VV .npy")
    r.add_argument("--date", required=True)
    r.add_argument("--vv", default=None, help="Path to VV dB .npy (default: .cache/sentinel/s1/...)")
    r.add_argument("--full", action="store_true", help="CFAR stride=1 (slower, exact)")
    r.set_defaults(func=cmd_run_local)

    s = sub.add_parser("run-snap", help="Run SNAP gpt graph on a local SAFE")
    s.add_argument("--safe", required=True, help="Path to .SAFE folder or zip SNAP can read")
    s.add_argument("--out", default=None)
    s.add_argument("--vh", action="store_true", help="Calibrate VV+VH")
    s.set_defaults(func=cmd_run_snap)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
