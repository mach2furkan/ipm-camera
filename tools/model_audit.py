"""Resume-safe, local model-assisted label triage. Predictions are review suggestions, never truth."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from ipcam.vision.desk import NAMES
from tools.evaluate_camera import iou

PROMPTS = ('person', 'book', 'laptop', 'computer monitor', 'computer keyboard',
           'computer mouse', 'mobile phone', 'cup', 'bottle', 'pen', 'pencil')
CLASS_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_annotations(truth, predictions):
    suggestions = []
    for prediction in predictions:
        box = prediction['box']
        same = max((iou(box, target) for target in truth if target[0] == box[0]), default=0)
        if same >= .5:
            continue
        other = max(((iou(box, target), int(target[0])) for target in truth if target[0] != box[0]),
                    default=(0, -1))
        if other[0] >= .7:
            kind = 'class_conflict_candidate'
        elif max((iou(box, target) for target in truth), default=0) < .2:
            kind = 'missing_annotation_candidate'
        else:
            kind = 'box_extent_candidate'
        # Pen and pencil prompts can both detect the same object.
        if any(s['box'][0] == box[0] and iou(s['box'], box) >= .7 for s in suggestions):
            continue
        suggestions.append(dict(**prediction, kind=kind, source_class=other[1] if other[0] >= .7 else None))
    undetected = [target for target in truth if not any(p['box'][0] == target[0] and iou(p['box'], target) >= .5
                                                       for p in predictions)]
    priority = sum(s['confidence'] * (2 if s['kind'] == 'class_conflict_candidate' else 1)
                   for s in suggestions) + .05*len(undetected)
    return dict(suggestions=suggestions, model_unconfirmed_source_boxes=undetected, priority=priority)


def read_records(path):
    records = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                record = json.loads(line)
                records[record['image']] = record
            except (ValueError, KeyError):
                # An interrupted final line is ignored; its image is reprocessed.
                continue
    return records


def run(root, weights, imgsz=960, batch=4, confidence=.45, limit=None):
    import torch
    from ultralytics import YOLOWorld
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from ultralytics.utils import WEIGHTS_DIR
    from tools.label_editor import atomic_json, build_editor
    root, weights = root.resolve(), weights.resolve()
    if not weights.is_file() or not (Path(WEIGHTS_DIR) / 'clip/ViT-B-32.pt').is_file():
        raise ValueError('Local YOLO-World and CLIP weights required; audit does not download weights')
    if not torch.cuda.is_available():
        raise ValueError('CUDA required for full-dataset audit')
    if batch < 1 or imgsz < 32 or not 0 < confidence <= 1:
        raise ValueError('Invalid audit configuration')
    signature = hashlib.sha256(json.dumps(dict(weights_sha256=sha(weights), prompts=PROMPTS, imgsz=imgsz,
                                               confidence=confidence, compare_version=1), sort_keys=True).encode()).hexdigest()
    output = root / 'model_audit.jsonl'
    old = read_records(output)
    images = sorted(p for p in (root / 'images').rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if limit is not None:
        images = images[:limit]
    jobs, completed = [], []
    for image in images:
        key = image.relative_to(root).as_posix()
        label = root / 'labels' / image.relative_to(root / 'images').with_suffix('.txt')
        hashes = dict(image_sha256=sha(image), label_sha256=sha(label))
        record = old.get(key, {})
        if record.get('signature') == signature and all(record.get(k) == v for k, v in hashes.items()):
            completed.append(record)
        else:
            jobs.append(dict(image=key, path=image, label=label, **hashes))
    print(f'Audit: {len(images)} images, {len(completed)} cached, {len(jobs)} to infer', flush=True)
    progress = root / 'model_audit_status.json'
    atomic_json(progress, dict(stage='auditing', total=len(images), cached=len(completed), inferred=0,
                               signature=signature))
    model = None
    if jobs:
        model = YOLOWorld(str(weights)).to('cuda')
        model.set_classes(list(PROMPTS))
    started = time.monotonic()
    precision = {'quantize': 16} if 'quantize' in DEFAULT_CFG_DICT else {'half': True}
    with output.open('a', encoding='utf-8') as stream:
        # Separate incomplete prior JSON from the first new complete record.
        stream.write('\n')
        for offset in range(0, len(jobs), batch):
            chunk = jobs[offset:offset+batch]
            results = model.predict([str(job['path']) for job in chunk], imgsz=imgsz, device=0, **precision,
                                    conf=confidence, iou=.5, rect=False, verbose=False)
            if len(results) != len(chunk):
                raise RuntimeError('Model output count differs from input batch')
            for job, result in zip(chunk, results):
                boxes = result.boxes.xywhn.cpu().tolist()
                classes = result.boxes.cls.cpu().tolist()
                scores = result.boxes.conf.cpu().tolist()
                predictions = [dict(box=[CLASS_IDS[int(c)], *box], confidence=float(score))
                               for box, c, score in zip(boxes, classes, scores)]
                truth = [list(map(float, line.split())) for line in job['label'].read_text(encoding='utf-8').splitlines()]
                record = dict(image=job['image'], image_sha256=job['image_sha256'],
                              label_sha256=job['label_sha256'], signature=signature,
                              **compare_annotations(truth, predictions))
                stream.write(json.dumps(record) + '\n')
                completed.append(record)
            stream.flush()
            if (offset+len(chunk)) % 100 < batch or offset+len(chunk) == len(jobs):
                atomic_json(progress, dict(stage='auditing', total=len(images), inferred=offset+len(chunk),
                                           cached=len(images)-len(jobs), signature=signature))
                print(f'{offset+len(chunk)}/{len(jobs)} inferred; elapsed={time.monotonic()-started:.1f}s', flush=True)
    counts = Counter(s['kind'] for record in completed for s in record['suggestions'])
    report = dict(status='model_assisted_triage_only', images=len(completed), names=list(NAMES),
                  teacher_weights=str(weights), signature=signature, confidence=confidence, imgsz=imgsz,
                  hypotheses=dict(counts), images_with_suggestions=sum(bool(r['suggestions']) for r in completed),
                  ranked_images=[dict(image=r['image'], priority=r['priority'], suggestions=len(r['suggestions']))
                                 for r in sorted(completed, key=lambda r: r['priority'], reverse=True)],
                  elapsed_seconds=time.monotonic()-started,
                  limitation='Teacher can miss objects or hallucinate; no labels or approvals were changed')
    atomic_json(root / 'model_audit_summary.json', report)
    build_editor(root)
    atomic_json(progress, dict(stage='complete', total=len(images), inferred=len(jobs),
                               cached=len(images)-len(jobs), signature=signature))
    print(json.dumps({k: report[k] for k in ('images', 'hypotheses', 'images_with_suggestions')}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('dataset_cctv_desk'))
    parser.add_argument('--weights', type=Path, default=Path('yolov8m-worldv2.pt'))
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    try:
        run(args.root, args.weights, args.imgsz, args.batch, limit=args.limit)
    except Exception as exc:
        from tools.label_editor import atomic_json
        atomic_json(args.root / 'model_audit_status.json', dict(stage='failed', error=str(exc)))
        raise
