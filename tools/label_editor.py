"""Build a local box editor; import explicit review decisions with stale-data checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import functools
import os
import time
from contextlib import contextmanager
from pathlib import Path

from ipcam.vision.desk import NAMES
from PIL import Image


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review_revision(entry):
    return hashlib.sha256(json.dumps(entry, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode('utf-8')).hexdigest()


def paths(root, key):
    relative = Path(key)
    if (relative.is_absolute() or len(relative.parts) < 3 or relative.parts[0] != 'images'
            or relative.parts[1] not in ('train', 'val') or '..' in relative.parts):
        raise ValueError('Invalid dataset image path')
    image = (root / relative).resolve()
    label = (root / 'labels' / Path(*relative.parts[1:]).with_suffix('.txt')).resolve()
    if not image.is_relative_to(root) or not label.is_relative_to(root):
        raise ValueError('Path escapes dataset root')
    return image, label


def validate_boxes(boxes, width, height):
    if not isinstance(boxes, list):
        raise ValueError('Boxes must be a list')
    seen = set()
    for box in boxes:
        if not isinstance(box, list) or len(box) != 5:
            raise ValueError('Each box must contain class, x, y, width, height')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
            raise ValueError('Non-finite or invalid box value')
        cls, x, y, w, h = box
        if cls != int(cls) or not 0 <= cls < len(NAMES):
            raise ValueError('Invalid class ID')
        if not (w > 0 and h > 0 and x-w/2 >= -1e-6 and y-h/2 >= -1e-6
                and x+w/2 <= 1+1e-6 and y+h/2 <= 1+1e-6):
            raise ValueError('Box outside image')
        if w*width*h*height <= 16:
            raise ValueError('Box area <=16 pixels needs further review')
        if tuple(box) in seen:
            raise ValueError('Duplicate box')
        seen.add(tuple(box))


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


@contextmanager
def dataset_lock(root):
    """Serialize review writes across the browser service and CLI processes."""
    with (root / '.review.lock').open('a+b') as handle:
        handle.seek(0, os.SEEK_END)
        if not handle.tell():
            handle.write(b'0')
            handle.flush()
        started = time.monotonic()
        while True:
            handle.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic()-started > 5:
                    raise ValueError('Dataset is being updated; retry after the other save finishes')
                time.sleep(.05)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def serialized(operation):
    @functools.wraps(operation)
    def wrapped(root, *args, **kwargs):
        root = Path(root).resolve()
        with dataset_lock(root):
            return operation(root, *args, **kwargs)
    return wrapped


@serialized
def stage_correction(root: Path, key: str, boxes: list, reason: str, reviewer: str,
                     image_sha256=None, label_sha256=None, draft=False, expected_revision=None):
    """Save a visually checked partial correction without claiming exhaustive review."""
    root = root.resolve()
    image, label = paths(root, key)
    if ((image_sha256 is not None and digest(image) != image_sha256)
            or (label_sha256 is not None and digest(label) != label_sha256)):
        raise ValueError('Stale editor data: image or labels changed; reload this image')
    with Image.open(image) as source:
        validate_boxes(boxes, *source.size)
    review_path = root / 'review.json'
    review = json.loads(review_path.read_text(encoding='utf-8'))
    if review.get('names') != list(NAMES):
        raise ValueError('Class definitions changed')
    if key not in review.get('images', {}) or not reason.strip() or not reviewer.strip():
        raise ValueError('Known image, reviewer and correction reason required')
    if expected_revision is not None and review_revision(review['images'][key]) != expected_revision:
        raise ValueError('Stale review state: another reviewer saved this image; reload')
    history = root / 'review_history' / Path(key).with_suffix('.json')
    history.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(history.read_text(encoding='utf-8')) if history.exists() else []
    records.append(dict(previous_label=label.read_text(encoding='utf-8'), previous_review=review['images'][key],
                        correction=dict(boxes=boxes, reason=reason, reviewer=reviewer), approval=False))
    atomic_json(history, records)
    temporary = label.with_suffix('.txt.tmp')
    temporary.write_text('\n'.join(f'{int(b[0])} ' + ' '.join(format(v, '.12g') for v in b[1:]) for b in boxes),
                         encoding='utf-8')
    temporary.replace(label)
    finding = review['images'][key].get('finding', '') if draft else reason
    findings_path = root / 'review_findings.json'
    if findings_path.exists():
        findings = json.loads(findings_path.read_text(encoding='utf-8'))
        if findings.get(key, {}).get('status') == 'open':
            finding = findings[key]['reason']
    review['images'][key].update(status='pending' if draft and not finding else 'needs_correction',
                                all_target_objects_checked=False,
                                image_sha256=digest(image), label_sha256=digest(label),
                                finding=finding, correction_note=reason, correction_reviewer=reviewer)
    atomic_json(review_path, review)
    return review['images'][key]


@serialized
def import_patch(root: Path, patch: dict):
    root = root.resolve()
    if patch.get('names') != list(NAMES):
        raise ValueError('Class definitions changed')
    if patch.get('all_target_objects_checked') is not True:
        raise ValueError('Check every target class before approval')
    if not isinstance(patch.get('reviewer'), str) or not patch['reviewer'].strip():
        raise ValueError('Reviewer identity required')
    if not isinstance(patch.get('session'), str) or not patch['session'].strip():
        raise ValueError('Camera/source session required')
    key = patch.get('image', '')
    image, label = paths(root, key)
    review_path = root / 'review.json'
    review = json.loads(review_path.read_text(encoding='utf-8'))
    if review.get('names') != list(NAMES) or key not in review['images']:
        raise ValueError('Image not present in current review manifest')
    if patch.get('review_revision') is not None and review_revision(review['images'][key]) != patch['review_revision']:
        raise ValueError('Stale review state: another reviewer saved this image; reload')
    if patch.get('image_sha256') != digest(image) or patch.get('label_sha256') != digest(label):
        raise ValueError('Stale editor data: image or labels changed; rebuild editor')
    with Image.open(image) as source:
        width, height = source.size
    boxes = patch.get('boxes')
    validate_boxes(boxes, width, height)
    for other_key, entry in review['images'].items():
        if (other_key != key and entry.get('status') == 'approved'
                and entry.get('session') == patch['session'].strip()
                and Path(other_key).parts[1] != Path(key).parts[1]):
            raise ValueError('Session already approved in another split')
    finding_path = root / 'review_findings.json'
    findings = json.loads(finding_path.read_text(encoding='utf-8')) if finding_path.exists() else {}
    if findings.get(key, {}).get('status') == 'open' and patch.get('finding_resolved') is not True:
        raise ValueError('Explicit resolution of existing finding required')
    # Save a recoverable history before changing labels. A crash leaves hash validation closed.
    history = root / 'review_history' / Path(key).with_suffix('.json')
    history.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(history.read_text(encoding='utf-8')) if history.exists() else []
    records.append(dict(previous_label=label.read_text(encoding='utf-8'), previous_review=review['images'][key],
                        patch=patch))
    atomic_json(history, records)
    text = '\n'.join(f'{int(b[0])} ' + ' '.join(format(v, '.12g') for v in b[1:]) for b in boxes)
    temporary = label.with_suffix('.txt.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(label)
    review['images'][key] = dict(status='approved', reviewer=patch['reviewer'].strip(),
                                session=patch['session'].strip(), all_target_objects_checked=True,
                                image_sha256=digest(image), label_sha256=digest(label))
    atomic_json(review_path, review)
    if findings.get(key, {}).get('status') == 'open':
        findings[key].update(status='resolved', reviewer=patch['reviewer'].strip(), label_sha256=digest(label))
        atomic_json(finding_path, findings)
    return review['images'][key]


@serialized
def restore_last_save(root: Path, key: str, image_sha256: str, label_sha256: str, expected_revision=None):
    image, label = paths(root, key)
    if digest(image) != image_sha256 or digest(label) != label_sha256:
        raise ValueError('Stale editor data: reload before restoring labels')
    history = root / 'review_history' / Path(key).with_suffix('.json')
    records = json.loads(history.read_text(encoding='utf-8')) if history.exists() else []
    if not records:
        raise ValueError('No saved label history for this image')
    previous_text = records[-1]['previous_label']
    boxes = [list(map(float, line.split())) for line in previous_text.splitlines()]
    with Image.open(image) as source:
        validate_boxes(boxes, *source.size)
    review_path = root / 'review.json'
    review = json.loads(review_path.read_text(encoding='utf-8'))
    records.append(dict(previous_label=label.read_text(encoding='utf-8'), previous_review=review['images'][key],
                        restore=True, approval=False))
    if review.get('names') != list(NAMES):
        raise ValueError('Class definitions changed')
    if expected_revision is not None and review_revision(review['images'][key]) != expected_revision:
        raise ValueError('Stale review state: reload before restoring labels')
    atomic_json(history, records)
    temporary = label.with_suffix('.txt.tmp')
    temporary.write_text(previous_text, encoding='utf-8')
    temporary.replace(label)
    review['images'][key].update(status='pending', all_target_objects_checked=False,
                                image_sha256=digest(image), label_sha256=digest(label))
    findings_path = root / 'review_findings.json'
    if findings_path.exists():
        findings = json.loads(findings_path.read_text(encoding='utf-8'))
        if key in findings:
            findings[key]['status'] = 'open'
            review['images'][key].update(status='needs_correction', finding=findings[key]['reason'])
            atomic_json(findings_path, findings)
    atomic_json(review_path, review)
    return review['images'][key]


def build_editor(root: Path):
    root = root.resolve()
    review = json.loads((root / 'review.json').read_text(encoding='utf-8'))
    from tools.model_audit import read_records
    model_records = read_records(root / 'model_audit.jsonl')
    items = []
    for key, entry in review['images'].items():
        image, label = paths(root, key)
        rows = [list(map(float, row.split())) for row in label.read_text(encoding='utf-8').splitlines()]
        hashes = dict(image_sha256=digest(image), label_sha256=digest(label))
        status = entry.get('status', 'pending')
        if status == 'approved' and not all(entry.get(k) == v for k, v in hashes.items()):
            status = 'pending'
        audit = model_records.get(key, {})
        if not all(audit.get(k) == v for k, v in hashes.items()):
            audit = {}
        items.append(dict(image=key, boxes=rows, status=status,
                          finding=entry.get('finding', ''), suggestions=audit.get('suggestions', []),
                          priority=audit.get('priority', 0), review_revision=review_revision(entry), **hashes))
    items.sort(key=lambda item: (item['status'] != 'needs_correction', item['status'] == 'approved',
                                -item['priority'], item['image']))
    template = Path(__file__).with_suffix('.html').read_text(encoding='utf-8')
    payload = json.dumps(dict(names=list(NAMES), items=items), ensure_ascii=False).replace('<', '\\u003c')
    output = root / 'label-editor.html'
    output.write_text(template.replace('/*DATA*/', payload).replace('/*API*/', 'null'), encoding='utf-8')
    print(output, flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('dataset_cctv_desk'))
    parser.add_argument('--import-patch', type=Path)
    args = parser.parse_args()
    if args.import_patch:
        import_patch(args.root, json.loads(args.import_patch.read_text(encoding='utf-8')))
    build_editor(args.root)
