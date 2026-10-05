"""Build a hash-verified review queue; never modify annotations or approve samples."""
import argparse
import json
import math
from pathlib import Path

from ipcam.vision.desk import NAMES
from tools.label_editor import atomic_json, paths, review_revision
from tools.model_audit import read_records, sha


def build_queue(root, limit=100, per_session=10):
    root = Path(root).resolve()
    if limit < 1 or per_session < 1:
        raise ValueError('Queue limits must be positive')
    review = json.loads((root / 'review.json').read_text(encoding='utf-8'))
    if review.get('names') != list(NAMES):
        raise ValueError('Review taxonomy mismatch')
    records = read_records(root / 'model_audit.jsonl')
    candidates, stale = [], 0
    for key, entry in review['images'].items():
        image, label = paths(root, key)
        image_hash, label_hash = sha(image), sha(label)
        approved = (entry.get('status') == 'approved' and entry.get('image_sha256') == image_hash
                    and entry.get('label_sha256') == label_hash)
        if approved:
            continue
        record = records.get(key, {})
        fresh = (record.get('image_sha256') == image_hash and record.get('label_sha256') == label_hash)
        stale += bool(record) and not fresh
        suggestions = record.get('suggestions', []) if fresh else []
        score = record.get('priority', 0) if fresh else 0
        if not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0:
            raise ValueError('Invalid audit priority')
        classes = sorted({int(s['box'][0]) for s in suggestions})
        if any(c not in range(len(NAMES)) for c in classes):
            raise ValueError('Invalid suggestion class')
        candidates.append(dict(image=key, image_sha256=image_hash, label_sha256=label_hash,
                               review_revision=review_revision(entry), priority=float(score),
                               session=entry.get('session') or key.split('/')[1],
                               suggested_classes=classes, suggestions=len(suggestions)))
    candidates.sort(key=lambda item: (-item['priority'], item['image']))
    selected, sessions, classes_seen = [], {}, set()
    # First cover suggested classes, then fill with difficult cases; cap correlated source/session samples.
    for diversity_pass in (True, False):
        for item in candidates:
            if len(selected) >= limit:
                break
            if item in selected or sessions.get(item['session'], 0) >= per_session:
                continue
            if diversity_pass and not (set(item['suggested_classes']) - classes_seen):
                continue
            selected.append(item)
            classes_seen.update(item['suggested_classes'])
            sessions[item['session']] = sessions.get(item['session'], 0) + 1
    return dict(names=list(NAMES), selected=selected, pending=len(candidates), stale_audits=stale,
                policy='Class coverage then model disagreement; session cap; suggestions are not truth',
                approvals_written=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('dataset_cctv_desk'))
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--per-session', type=int, default=10)
    args = parser.parse_args()
    report = build_queue(args.root, args.limit, args.per_session)
    atomic_json(args.root / 'active_learning_queue.json', report)
    print(f"Selected {len(report['selected'])}; pending {report['pending']}; stale audits {report['stale_audits']}")
