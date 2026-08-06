"""Sentinel-2 ship detection via mayrajeo YOLO weights (Ultralytics).

Weights: https://huggingface.co/mayrajeo/marine-vessel-yolo  (yolo11s_tci.pt)
Trained on Baltic L1C-TCI patches (320→640). Our CDSE L2A RGB is a domain
shift — treat results as an A/B trial, not production truth.

If Hugging Face downloads fail (corporate proxy / login HTML), download
``yolo11s_tci.pt`` in a browser and set env ``YOLO_WEIGHTS`` to that path, or pass
``weight_path=...`` / detect param ``yolo_weight_path``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vessel_detect import (
    build_berth_slot_masks,
    classify_length,
    scene_slot_quality,
)

DEFAULT_REPO = "mayrajeo/marine-vessel-yolo"
# Single ~19MB YOLO11s checkpoint trained on L1C-TCI (Baltic). Domain-shift vs our L2A RGB.
DEFAULT_WEIGHT = "yolo11s_tci.pt"


@lru_cache(maxsize=4)
def load_yolo_model(
    repo_id: str = DEFAULT_REPO,
    filename: str = DEFAULT_WEIGHT,
    weight_path: str | None = None,
    cache_dir: str | None = None,
):
    """Load Ultralytics YOLO weights (local path or Hugging Face download)."""
    from ultralytics import YOLO

    path = weight_path or os.environ.get("YOLO_WEIGHTS")
    # Prefer weights dropped next to this module (e.g. port_loading_gaps/yolo11s_tci.pt).
    if not path:
        sibling = Path(__file__).resolve().parent / filename
        if sibling.exists() and sibling.stat().st_size >= 1_000_000:
            path = str(sibling)
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"YOLO_WEIGHTS / weight_path not found: {p}")
        return YOLO(str(p))

    try:
        from huggingface_hub import hf_hub_download

        kwargs: dict[str, Any] = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
        # Guard against HF login HTML saved as the weight file.
        if Path(downloaded).stat().st_size < 1_000_000:
            raise RuntimeError(
                f"Downloaded file looks too small ({Path(downloaded).stat().st_size} bytes) — "
                "likely a login/error page, not the model."
            )
        return YOLO(downloaded)
    except Exception as exc:  # noqa: BLE001 — surface a clear install hint
        raise RuntimeError(
            "Could not load mayrajeo YOLO weights.\n"
            "1) Open https://huggingface.co/mayrajeo/marine-vessel-yolo and download "
            f"{DEFAULT_WEIGHT}\n"
            "2) Set env YOLO_WEIGHTS to that .pt path, or pass yolo_weight_path in detect params.\n"
            f"Original error: {exc}"
        ) from exc


def _xyxy_to_det(xyxy: np.ndarray, conf: float, resolution: float) -> dict:
    x1, y1, x2, y2 = map(float, xyxy)
    w_px, h_px = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    length_m = max(w_px, h_px) * resolution
    width_m = min(w_px, h_px) * resolution
    return {
        "centroid_xy": (round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)),
        "xyxy": (round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)),
        "length_m": round(length_m, 1),
        "width_m": round(width_m, 1),
        "size_class": classify_length(length_m),
        "angle_deg": None,
        "yolo_conf": round(float(conf), 3),
        "sensor": "yolo",
        "quality": "ok",
    }


def _predict_full(
    model,
    rgb: np.ndarray,
    *,
    conf: float,
    imgsz: int,
) -> list[dict]:
    """Run YOLO on one RGB uint8 image (BGR conversion for Ultralytics)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    results = model.predict(bgr, conf=conf, imgsz=imgsz, verbose=False)
    out: list[dict] = []
    if not results:
        return out
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return out
    xyxy = r0.boxes.xyxy.cpu().numpy()
    scores = r0.boxes.conf.cpu().numpy()
    for box, score in zip(xyxy, scores):
        out.append({"_xyxy": box, "_conf": float(score)})
    return out


def predict_ships_rgb(
    scene: np.ndarray,
    *,
    resolution: float = 10.0,
    conf: float = 0.25,
    imgsz: int = 640,
    tile: int = 320,
    overlap: float = 0.2,
    model=None,
    repo_id: str = DEFAULT_REPO,
    weight_file: str = DEFAULT_WEIGHT,
    weight_path: str | None = None,
) -> list[dict]:
    """
    Detect ships on an RGB uint8 AOI array.

    Small AOIs run as one shot; larger scenes use overlapping tiles (simple NMS).
    """
    if scene is None or scene.size == 0:
        return []
    model = model or load_yolo_model(repo_id, weight_file, weight_path=weight_path)
    h, w = scene.shape[:2]
    raw: list[dict] = []

    if max(h, w) <= tile * 1.25:
        raw = _predict_full(model, scene, conf=conf, imgsz=imgsz)
        for d in raw:
            d.update(_xyxy_to_det(d.pop("_xyxy"), d.pop("_conf"), resolution))
        return raw

    step = max(1, int(tile * (1 - overlap)))
    ys = list(range(0, max(h - tile, 0) + 1, step))
    xs = list(range(0, max(w - tile, 0) + 1, step))
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]
    if h > tile and ys[-1] != h - tile:
        ys.append(h - tile)
    if w > tile and xs[-1] != w - tile:
        xs.append(w - tile)

    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            patch = scene[y0:y1, x0:x1]
            if patch.shape[0] < tile or patch.shape[1] < tile:
                pad = np.zeros((tile, tile, 3), dtype=np.uint8)
                pad[: patch.shape[0], : patch.shape[1]] = patch
                patch = pad
            for d in _predict_full(model, patch, conf=conf, imgsz=imgsz):
                box = d["_xyxy"].copy()
                box[0] += x0
                box[2] += x0
                box[1] += y0
                box[3] += y0
                raw.append({"_xyxy": box, "_conf": d["_conf"]})

    dets = [_xyxy_to_det(d["_xyxy"], d["_conf"], resolution) for d in raw]
    return _nms_centroids(dets, min_sep_px=max(8, int(25 / max(resolution, 1))))


def _nms_centroids(dets: list[dict], min_sep_px: float = 12) -> list[dict]:
    kept: list[dict] = []
    for det in sorted(dets, key=lambda d: -d.get("yolo_conf", 0)):
        cx, cy = det["centroid_xy"]
        if any(
            np.hypot(cx - k["centroid_xy"][0], cy - k["centroid_xy"][1]) < min_sep_px
            for k in kept
        ):
            continue
        kept.append(det)
    return kept


def assign_yolo_to_slots(
    yolo_dets: list[dict],
    shape: tuple[int, ...],
    slots_px: list[dict],
    *,
    min_conf: float = 0.25,
    min_length_m: float = 80.0,
) -> list[dict]:
    """Keep the strongest in-slot YOLO hit per berth."""
    out: list[dict] = []
    for name, slot_mask in build_berth_slot_masks(shape, slots_px):
        best = None
        for det in yolo_dets:
            if det.get("yolo_conf", 0) < min_conf:
                continue
            if det.get("length_m", 0) < min_length_m:
                continue
            cx, cy = det["centroid_xy"]
            x, y = int(round(cx)), int(round(cy))
            if not (0 <= y < slot_mask.shape[0] and 0 <= x < slot_mask.shape[1]):
                continue
            if not slot_mask[y, x]:
                continue
            if best is None or det["yolo_conf"] > best["yolo_conf"]:
                best = {**det, "berth": name}
        if best is not None:
            out.append(best)
    return out


def detect_berth_slots_yolo(
    scene: np.ndarray,
    slots_px: list[dict],
    resolution: float = 10.0,
    detect_params: dict | None = None,
) -> list[dict]:
    """YOLO occupancy per berth slot (same skip semantics as classical path)."""
    params = detect_params or {}
    quality = scene_slot_quality(scene, slots_px, params)
    if quality["quality"] == "skip":
        return [
            {
                "berth": None,
                "length_m": None,
                "width_m": None,
                "size_class": "scene_skip",
                "centroid_xy": None,
                "angle_deg": None,
                "sensor": "yolo",
                "quality": "skip",
                **quality,
            }
        ]

    yolo_dets = predict_ships_rgb(
        scene,
        resolution=resolution,
        conf=params.get("yolo_conf", 0.25),
        imgsz=params.get("yolo_imgsz", 640),
        tile=params.get("yolo_tile", 320),
        overlap=params.get("yolo_overlap", 0.2),
        repo_id=params.get("yolo_repo", DEFAULT_REPO),
        weight_file=params.get("yolo_weight", DEFAULT_WEIGHT),
        weight_path=params.get("yolo_weight_path") or os.environ.get("YOLO_WEIGHTS"),
    )
    return assign_yolo_to_slots(
        yolo_dets,
        scene.shape,
        slots_px,
        min_conf=params.get("yolo_conf", 0.25),
        min_length_m=params.get("yolo_min_length_m", params.get("slot_min_length_m", 80)),
    )
