"""
Standalone berth slot picker — use when %matplotlib widget/qt fail in the notebook.

    .venv\\Scripts\\python pick_berths_standalone.py --aoi muajjiz --date 2026-08-01
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentinelhub import (
    BBox,
    CRS,
    DataCollection,
    Geometry,
    MimeType,
    SHConfig,
    SentinelHubRequest,
    bbox_to_dimensions,
)

NOTEBOOK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NOTEBOOK_DIR))
load_dotenv(NOTEBOOK_DIR.parent / ".env")

from berth_slot_picker import format_berth_slots_python, pick_berth_slots_tk  # noqa: E402

CACHE_DIR = NOTEBOOK_DIR / ".cache" / "sentinel"

MUAJJIZ_POLYGON = {
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

YANBU_POLYGON = {
    "type": "Polygon",
    "coordinates": [
        [
            [38.221478, 23.945802],
            [38.239846, 23.936839],
            [38.237851, 23.933348],
            [38.219182, 23.942566],
            [38.221478, 23.945802],
        ]
    ],
}

DEFAULT_SLOTS = {
    "muajjiz": [
        {"name": "berth_1", "box": [16, 16, 61, 50]},
        {"name": "berth_2", "box": [51, 55, 95, 86]},
        {"name": "berth_3", "box": [78, 90, 145, 124]},
    ],
    "yanbu_north_crude_terminal": [
        {"name": "berth_1", "box": [15, 8, 65, 42]},
        {"name": "berth_2", "box": [45, 28, 95, 62]},
        {"name": "berth_3", "box": [85, 52, 135, 86]},
        {"name": "berth_4", "box": [125, 78, 205, 130]},
    ],
}

VAR_NAMES = {
    "muajjiz": "MUAJJIZ_BERTH_SLOTS",
    "yanbu_north_crude_terminal": "YANBU_BERTH_SLOTS",
}

AOI_GEOM = {
    "muajjiz": Geometry(MUAJJIZ_POLYGON, crs=CRS.WGS84),
    "yanbu_north_crude_terminal": Geometry(YANBU_POLYGON, crs=CRS.WGS84),
}


def _zirku_bbox(lat, lon, buffer_m=1200):
    dlat = buffer_m / 111_000
    dlon = buffer_m / (111_000 * np.cos(np.radians(lat)))
    return BBox(bbox=(lon - dlon, lat - dlat, lon + dlon, lat + dlat), crs=CRS.WGS84)


def fetch_scene(aoi: str, date: str) -> np.ndarray:
    cache = CACHE_DIR / "s2" / aoi / f"{date}.npy"
    if cache.exists():
        print(f"Loaded cached scene: {cache}")
        return np.load(cache)

    config = SHConfig()
    config.sh_client_id = os.getenv("SH_CLIENT_ID")
    config.sh_client_secret = os.getenv("SH_CLIENT_SECRET")
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    config.sh_token_url = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    )
    s2 = DataCollection.SENTINEL2_L2A.define_from("s2l2a", service_url=config.sh_base_url)

    if aoi in AOI_GEOM:
        geom = AOI_GEOM[aoi]
        kwargs = {"geometry": geom}
        size = bbox_to_dimensions(geom.bbox, 10)
    elif aoi == "zirku_spm_a":
        bbox = _zirku_bbox(25.00833, 52.98333)
        kwargs = {"bbox": bbox}
        size = bbox_to_dimensions(bbox, 10)
    else:
        raise ValueError(f"Unknown AOI: {aoi}")

    ev = """
//VERSION=3
function setup(){return{input:[{bands:["B04","B03","B02","SCL"]}],output:{bands:3,sampleType:"UINT8"}}}
function evaluatePixel(s){if([3,8,9,10].includes(s.SCL))return[0,0,0];
return[Math.min(255,s.B04*255*3.5),Math.min(255,s.B03*255*3.5),Math.min(255,s.B02*255*3.5)];}
"""
    req = SentinelHubRequest(
        evalscript=ev,
        input_data=[
            SentinelHubRequest.input_data(s2, time_interval=(date, date), mosaicking_order="leastCC")
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        size=size,
        config=config,
        **kwargs,
    )
    img = req.get_data()[0]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, img)
    print(f"Fetched and cached: {cache}")
    return img


def main():
    parser = argparse.ArgumentParser(description="Pick berth slot boxes (Tkinter window)")
    parser.add_argument("--aoi", default="yanbu_north_crude_terminal", choices=list(AOI_GEOM) + ["zirku_spm_a"])
    parser.add_argument("--date", default="2026-08-01")
    parser.add_argument("--var-name", default=None)
    args = parser.parse_args()

    var_name = args.var_name or VAR_NAMES.get(args.aoi, "BERTH_SLOTS")
    existing = DEFAULT_SLOTS.get(args.aoi, [])
    slot_names = [s["name"] for s in existing] or None

    scene = fetch_scene(args.aoi, args.date)
    slots = pick_berth_slots_tk(
        scene,
        existing_slots=existing,
        slot_names=slot_names,
        var_name=var_name,
    )
    if slots:
        print("\nFinal — paste into AOI config:\n")
        print(format_berth_slots_python(slots, var_name=var_name))


if __name__ == "__main__":
    main()
