"""Verify dual-model isolation on one local image; no camera credentials needed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import hashlib
import cv2
import numpy as np
import torch
from ultralytics import YOLOWorld
from ultralytics.cfg import DEFAULT_CFG_DICT

from tools.live_detect import CLASSES
from ipcam.vision.traffic import VEHICLE_PROMPTS
from tools.live_runtime import ResponsiveInference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()
    image = cv2.imread(args.image)
    if image is None:
        parser.error("Image could not be read")
    path = Path("yolov8m-worldv2.pt")
    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    device = 0 if torch.cuda.is_available() else "cpu"
    precision = ({"quantize": 16 if device == 0 else 32} if "quantize" in DEFAULT_CFG_DICT
                 else {"half": device == 0})
    desk = YOLOWorld(str(path))
    desk.set_classes([c[0] for c in CLASSES])
    kwargs = dict(imgsz=args.imgsz, conf=min(c[2] for c in CLASSES), device=device,
                  rect=False, verbose=False, **precision)
    first = desk.predict(image, **kwargs)[0].boxes.data.cpu().numpy().copy()
    vehicle = YOLOWorld(str(path))
    vehicle.set_classes(list(VEHICLE_PROMPTS))
    pumps = []
    def predict_in_worker():
        result = desk.predict(image, **kwargs)[0].boxes.data.cpu().numpy().copy()
        vehicle.predict(image, **{**kwargs, "conf": .30})
        return result
    def pump():
        pumps.append(1)
        return True
    runtime = ResponsiveInference()
    try:
        second = runtime.run(predict_in_worker, pump)
    finally:
        runtime.close()
    assert tuple(desk.names.values()) == tuple(c[0] for c in CLASSES)
    assert tuple(vehicle.names.values()) == VEHICLE_PROMPTS
    assert first.shape == second.shape and np.allclose(first, second, atol=1e-5, rtol=1e-5)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_hash
    print(json.dumps({"device": str(device), "desk_detections": len(first),
        "desk_outputs_preserved": True, "weights_unchanged": True, "ui_pumps": len(pumps),
        "peak_gpu_allocated_mb": round(torch.cuda.max_memory_allocated()/2**20, 1)
        if device == 0 else None}))


if __name__ == "__main__":
    main()
