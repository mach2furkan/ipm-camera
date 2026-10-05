"""Generate reproducible prediction/latency records on reviewed held-out camera images."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ipcam.vision.desk import NAMES, prepare_detections
from run_pipeline import validate_model_classes
from tools.evaluate_camera import evaluate
from tools.label_editor import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Reviewed camera JSON with frames/truth/session')
    parser.add_argument('--weights', type=Path, required=True, help='Existing local weights only')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--world', action='store_true', help='Current YOLO-World baseline')
    parser.add_argument('--imgsz', type=int, default=960)
    args = parser.parse_args()
    if not args.weights.is_file() or args.imgsz < 32 or args.output.resolve() == args.input.resolve():
        parser.error('Local weights, valid resolution and separate output required')
    data = json.loads(args.input.read_text(encoding='utf-8'))
    for frame in data['frames']:
        frame['predictions'] = []
    evaluate(data)  # Validate taxonomy, review, held-out flags and boxes before loading the GPU.
    root = args.input.resolve().parent
    for frame in data['frames']:
        path = (root / frame['image']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Camera image must exist inside evaluation folder')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if frame.get('image_sha256') and frame['image_sha256'] != actual:
            raise ValueError('Camera image changed since annotation')
        frame['image_sha256'] = actual
    import torch
    import cv2
    from ultralytics import YOLO, YOLOWorld
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from tools.live_detect import CLASSES
    device = 0 if torch.cuda.is_available() else 'cpu'
    precision = ({'quantize': 16 if device == 0 else 32} if 'quantize' in DEFAULT_CFG_DICT
                 else {'half': device == 0})
    model = (YOLOWorld if args.world else YOLO)(str(args.weights))
    if args.world:
        model.set_classes([c[0] for c in CLASSES])
    mapping = (0, 1, 2, 3, 4, 9, 9, 5, 6, 7, 8) if args.world else tuple(range(10))
    floors = np.array([CLASSES[c][2] for c in (0, 1, 2, 3, 4, 7, 8, 9, 10, 5)]
                      if args.world else [.30]*9 + [.65])
    for _ in range(3):
        model.predict(np.zeros((args.imgsz, args.imgsz, 3), np.uint8), imgsz=args.imgsz,
                      device=device, **precision, verbose=False, rect=False)
    if not args.world:
        validate_model_classes(model.names)
    for index, frame in enumerate(data['frames']):
        image = cv2.imread(str(root / frame['image']))
        if image is None:
            raise ValueError('Camera image cannot be decoded')
        h, w = image.shape[:2]
        if device == 0:
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = model.predict(image, imgsz=args.imgsz, conf=float(floors.min()),
                               device=device, **precision, verbose=False, rect=False)[0]
        boxes = result.boxes
        detections = prepare_detections(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(),
                                        boxes.cls.cpu().numpy(), w, h, floors, mapping)
        if device == 0:
            torch.cuda.synchronize()
        frame['latency_ms'] = (time.perf_counter()-start)*1000
        frame['predictions'] = [dict(box=[int(c), (x1+x2)/2/w, (y1+y2)/2/h, (x2-x1)/w, (y2-y1)/h],
                                     confidence=float(score)) for x1, y1, x2, y2, score, c in detections]
        print(f'{index+1}/{len(data["frames"])}', flush=True)
    data['measurement'] = dict(weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                               imgsz=args.imgsz, device=str(device),
                               latency_scope='Prediction and postprocessing; excludes RTSP, decode and display')
    atomic_json(args.output, data)


if __name__ == '__main__':
    main()
