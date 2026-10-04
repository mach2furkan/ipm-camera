"""Validate a curated YOLO dataset, train, export and measure a batch-one engine.

No images are inferred to be negatives from missing annotation files.
Use --check to validate without starting training or downloading model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import importlib.util
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from ipcam.vision.desk import NAMES

ROOT = Path(__file__).resolve().parent


def validate_model_classes(names):
    if isinstance(names, dict):
        if set(names) != set(range(len(NAMES))):
            raise ValueError('Model class IDs must match the ten dataset classes')
        names = [names[i] for i in range(len(NAMES))]
    if list(names) != list(NAMES):
        raise ValueError('Model class names/order changed during training or export')


def validate_dataset(config, *, require_review=False):
    cfg = yaml.safe_load(config.read_text(encoding='utf-8'))
    names = cfg['names']
    if isinstance(names, dict) and set(names) != set(range(10)):
        raise ValueError('Dataset class IDs must be exactly 0..9')
    ordered = [names[i] for i in range(10)] if isinstance(names, dict) else names
    if list(ordered) != list(NAMES):
        raise ValueError('Dataset class order must match the ten fixed classes')
    root = Path(cfg['path'])
    if not root.is_absolute():
        root = config.parent / root
    counts, seen, pixels_seen, review_sessions = {}, {}, {}, {}
    review_path = root / 'review.json'
    review = json.loads(review_path.read_text(encoding='utf-8')) if require_review and review_path.exists() else {}
    if require_review and review.get('names') != list(NAMES):
        raise ValueError('Human review manifest missing or class definitions changed; run tools.review_dataset')
    findings_path = root / 'review_findings.json'
    if require_review and findings_path.exists():
        findings = json.loads(findings_path.read_text(encoding='utf-8'))
        if any(item.get('status') == 'open' for item in findings.values()):
            raise ValueError('Unresolved visual annotation findings; correct labels before training')
    for split in ('train', 'val'):
        folder = root / cfg[split]
        images = sorted(p for p in folder.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        if not images:
            raise ValueError(f'No annotated dataset images: {folder}')
        classes, negatives = [0] * 10, 0
        expected_labels = set()
        for image in images:
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in seen:
                raise ValueError(f'Duplicate image / split leakage: {image}, {seen[digest]}')
            seen[digest] = str(image)
            label = root / 'labels' / split / image.relative_to(folder).with_suffix('.txt')
            expected_labels.add(label.resolve())
            if not label.exists():
                raise ValueError(f'Missing annotation (explicit empty file required for negatives): {label}')
            with Image.open(image) as im:
                width, height = im.size
                im.verify()
            with Image.open(image) as im:
                rgb = im.convert('RGB')
                pixel_digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            if pixel_digest in pixels_seen:
                raise ValueError(f'Duplicate decoded image / split leakage: {image}, {pixels_seen[pixel_digest]}')
            pixels_seen[pixel_digest] = str(image)
            if require_review:
                key = image.relative_to(root).as_posix()
                entry = review.get('images', {}).get(key, {})
                label_digest = hashlib.sha256(label.read_bytes()).hexdigest()
                if (entry.get('status') != 'approved' or not entry.get('reviewer', '').strip()
                        or not entry.get('session', '').strip() or entry.get('image_sha256') != digest
                        or entry.get('label_sha256') != label_digest
                        or entry.get('all_target_objects_checked') is not True):
                    raise ValueError(f'Unreviewed or changed image/labels: {image}')
                session = entry['session']
                previous = review_sessions.setdefault(session, split)
                if previous != split:
                    raise ValueError(f'Camera/source session split leakage: {session}')
            rows = label.read_text(encoding='utf-8').splitlines()
            negatives += not rows
            unique_rows = set()
            for row in rows:
                values = row.split()
                if len(values) != 5:
                    raise ValueError(f'Invalid YOLO annotation: {label}')
                cls, x, y, w, h = map(float, values)
                box = (cls, x, y, w, h)
                if box in unique_rows:
                    raise ValueError(f'Duplicate annotation: {label}')
                unique_rows.add(box)
                if not np.isfinite([cls, x, y, w, h]).all() or cls != int(cls) or not 0 <= cls < 10:
                    raise ValueError(f'Invalid class or coordinates: {label}')
                if not (w > 0 and h > 0 and x-w/2 >= -1e-6 and y-h/2 >= -1e-6
                        and x+w/2 <= 1+1e-6 and y+h/2 <= 1+1e-6):
                    raise ValueError(f'Out of bounds annotation: {label}')
                if w * width * h * height <= 16:
                    raise ValueError(f'Annotation area <=16 pixels; review instead of silently deleting: {label}')
                classes[int(cls)] += 1
        orphan_labels = {p.resolve() for p in (root / 'labels' / split).rglob('*.txt')} - expected_labels
        if orphan_labels:
            raise ValueError(f'Orphan annotation without image: {sorted(map(str, orphan_labels))[0]}')
        if min(classes) == 0 or negatives == 0:
            raise ValueError(f'{split} must contain all ten classes and explicit hard negatives: {classes}, {negatives}')
        counts[split] = dict(images=len(images), boxes=classes, negatives=negatives)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'cctv_desk_data.yaml')
    parser.add_argument('--config', type=Path, default=ROOT / 'cctv_desk_hyperparams.yaml')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--benchmark-image', type=Path, help='Representative 2K camera image')
    parser.add_argument('--resume', type=Path, help='Resume an interrupted last.pt checkpoint')
    args = parser.parse_args()
    args.data = args.data.resolve()
    report = dict(dataset=validate_dataset(args.data, require_review=True))
    print(json.dumps(report, indent=2), flush=True)
    if args.check:
        return
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT
    import tensorrt  # Fail before spending hours training if export dependency is missing.

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    if importlib.util.find_spec('onnx') is None:
        raise RuntimeError('ONNX export dependency missing')
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    unknown = set(cfg) - set(DEFAULT_CFG_DICT)
    if unknown:
        raise ValueError(f'Unsupported training arguments: {unknown}')
    model = YOLO(args.resume or cfg.pop('model'))
    start = time.monotonic()
    if args.resume:
        model.train(resume=True, device=0)
    else:
        model.train(data=str(args.data), project=str(ROOT / 'cctv_desk_project'),
                    name='yolo11m_cctv_run', exist_ok=False, **cfg)
    best = Path(model.trainer.best)
    if not best.exists():
        raise RuntimeError('Training did not produce best.pt')
    trained = YOLO(best)
    validate_model_classes(trained.names)
    output = best.parent.parent / 'deployment_report.json'
    report.update(best_weights=str(best), train_hours=(time.monotonic() - start) / 3600,
                  status='trained; validation and export pending',
                  dataset_caveat='Public annotations: COCO tv is monitor proxy; pen not labelled in COCO; '
                                 'negative candidates need camera-domain review')
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    metrics = trained.val(data=str(args.data), imgsz=cfg['imgsz'], device=0)
    class_metrics = {}
    for index, class_id in enumerate(metrics.box.ap_class_index):
        precision, recall, ap50, ap = metrics.box.class_result(index)
        class_metrics[NAMES[int(class_id)]] = dict(class_id=int(class_id), precision=float(precision),
                                                 recall=float(recall), map50=float(ap50), map50_95=float(ap))
    report.update(gpu=torch.cuda.get_device_name(0), tensorrt=tensorrt.__version__,
                  map50=float(metrics.box.map50), map50_95=float(metrics.box.map),
                  per_class_metrics=class_metrics,
                  train_hours=(time.monotonic() - start) / 3600)
    engine = trained.export(format='engine', imgsz=cfg['imgsz'], half=True, device=0,
                            batch=1, dynamic=False, nms=True, simplify=True, workspace=2)
    detector = YOLO(engine, task='detect')
    sample = str(args.benchmark_image) if args.benchmark_image else np.zeros((1440, 2560, 3), np.uint8)
    timings, inference = [], []
    for i in range(120):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = detector.predict(sample, imgsz=cfg['imgsz'], device=0, verbose=False, rect=False)[0]
        if i == 0:
            validate_model_classes(result.names)
        torch.cuda.synchronize()
        if i >= 20:
            timings.append((time.perf_counter()-t0)*1000)
            inference.append(result.speed['inference'])
    report.update(engine=str(engine), benchmark_input='camera' if args.benchmark_image else 'synthetic',
                  predict_p50_ms=float(np.median(timings)), predict_p95_ms=float(np.percentile(timings, 95)),
                  inference_p50_ms=float(np.median(inference)),
                  status='exported; camera acceptance pending',
                  camera_acceptance='UNVERIFIED: held-out empty scenes and labelled pen at 2.5m required')
    output = best.parent.parent / 'deployment_report.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(output, flush=True)


if __name__ == '__main__':
    main()
