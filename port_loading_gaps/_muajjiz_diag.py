"""Diagnose Muajjiz 2026-08-01 detection."""
import os
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from sentinelhub import CRS, DataCollection, Geometry, MimeType, SHConfig, SentinelHubRequest, bbox_to_dimensions

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
config = SHConfig()
config.sh_client_id = os.getenv("SH_CLIENT_ID")
config.sh_client_secret = os.getenv("SH_CLIENT_SECRET")
config.sh_base_url = "https://sh.dataspace.copernicus.eu"
config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
S2 = DataCollection.SENTINEL2_L2A.define_from("s2l2a", service_url=config.sh_base_url)
RES = 10

MUAJJIZ = {
    "type": "Polygon",
    "coordinates": [
        [
            [38.395264, 23.810083],
            [38.398311, 23.812399],
            [38.410499, 23.801935],
            [38.407516, 23.799462],
            [38.395264, 23.810083],
        ]
    ],
}
geom = Geometry(MUAJJIZ, crs=CRS.WGS84)
size = bbox_to_dimensions(geom.bbox, RES)
EV = """
//VERSION=3
function setup(){return{input:[{bands:["B04","B03","B02","SCL"]}],output:{bands:3,sampleType:"UINT8"}};}
function evaluatePixel(s){
  if([3,8,9,10].includes(s.SCL))return[0,0,0];
  return[Math.min(255,s.B04*255*3.5),Math.min(255,s.B03*255*3.5),Math.min(255,s.B02*255*3.5)];
}
"""
PARAMS = {
    "diff_thresh": 25,
    "use_adaptive_diff": True,
    "diff_percentile": 90,
    "min_aspect": 1.65,
    "min_length_m": 110,
    "max_length_m": 350,
    "min_area_px": 35,
    "use_saturation": False,
    "use_hull_colors": True,
    "max_blob_frac": 0.15,
    "water_gray_lo": 8,
    "water_gray_hi": 85,
    "dedupe_px": 50,
    "hull_min_sep_px": 60,
}


def fetch(d):
    req = SentinelHubRequest(
        evalscript=EV,
        input_data=[SentinelHubRequest.input_data(S2, time_interval=(d, d), mosaicking_order="leastCC")],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        geometry=geom,
        size=size,
        config=config,
    )
    return req.get_data()[0]


def water_mask(rgb, lo, hi):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return ((g > lo) & (g < hi)).astype(np.uint8)


def hull_mask(scene, wm):
    r, g, b = scene[:, :, 0], scene[:, :, 1], scene[:, :, 2]
    gs = cv2.cvtColor(scene, cv2.COLOR_RGB2GRAY)
    red = (r > 85) & (r > g * 1.15) & (r > b * 1.15) & wm.astype(bool)
    olive = (g > 28) & (g > r * 1.02) & (gs > 10) & (gs < 95) & wm.astype(bool)
    m = (red | olive).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return m


def list_blobs(mask, label, min_area=10):
    print(f"\n{label} (nz={np.count_nonzero(mask)}):")
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows = []
    for cnt in cs:
        a = cv2.contourArea(cnt)
        if a < min_area:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
        lm = max(rw, rh) * RES
        asp = max(rw, rh) / max(min(rw, rh), 1)
        m = cv2.moments(cnt)
        rows.append((a, lm, asp, m["m10"] / m["m00"], m["m01"] / m["m00"]))
    for r in sorted(rows, reverse=True)[:10]:
        print(f"  area={r[0]:.0f} len={r[1]:.0f}m asp={r[2]:.2f} @({r[3]:.0f},{r[4]:.0f})")


scene = fetch("2026-08-01")
bg = np.median(np.stack([fetch(d).astype(np.float32) for d in ["2026-06-01", "2026-06-06", "2026-06-11", "2026-06-16"]], 0), 0).astype(np.uint8)
print("scene", scene.shape, "unique gray", len(np.unique(cv2.cvtColor(scene, cv2.COLOR_RGB2GRAY))))
gs, gb = cv2.cvtColor(scene, cv2.COLOR_RGB2GRAY), cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
print("black px (SCL/mask)", np.sum(gs == 0), "bright px", np.sum(gs > 85))

for lo, hi in [(8, 85), (8, 120), (1, 120), (5, 100)]:
    wm = water_mask(scene, lo, hi)
    diff = cv2.absdiff(gs, gb)
    diff[~wm.astype(bool)] = 0
    vals = diff[diff > 0]
    if len(vals) == 0:
        print(f"wm {lo}-{hi}: no diff")
        continue
    t = np.percentile(vals, 90)
    m = (diff >= t).astype(np.uint8) * 255
    print(f"wm {lo}-{hi}: water={wm.sum()} diff p90 t={t:.1f} mask_nz={np.count_nonzero(m)}")
    list_blobs(m, "  adaptive diff", 5)

list_blobs(hull_mask(scene, water_mask(scene, 8, 120)), "hull colors wm 8-120", 5)

# orange/red specific
r, g, b = scene[:, :, 0], scene[:, :, 1], scene[:, :, 2]
wm = water_mask(scene, 8, 120)
orange = ((r > 70) & (r > g * 1.1) & (g > 20) & wm.astype(bool)).astype(np.uint8) * 255
list_blobs(orange, "orange hull", 5)
