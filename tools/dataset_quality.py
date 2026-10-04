"""Produce auditable class, box-size and near-duplicate reports without approving labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from ipcam.vision.desk import NAMES
from run_pipeline import validate_dataset


def audit_quality(config: Path):
    import yaml
    cfg = yaml.safe_load(config.read_text(encoding='utf-8'))
    root = Path(cfg['path'])
    if not root.is_absolute():
        root = config.parent / root
    report = dict(names=list(NAMES), splits={}, issues=[], near_duplicate_candidates=[],
                  semantic_review='pending; automated checks cannot establish exhaustive labels')
    try:
        report['technical_validation'] = validate_dataset(config)
    except (ValueError, OSError) as exc:
        report['issues'].append(str(exc))
    fingerprints = defaultdict(list)
    exact = {}
    for split in ('train', 'val'):
        images = sorted(p for p in (root / cfg[split]).rglob('*')
                        if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        boxes, positives, sizes, sources = Counter(), Counter(), defaultdict(Counter), Counter()
        negatives = 0
        for image in images:
            try:
                label = root / 'labels' / split / image.relative_to(root / cfg[split]).with_suffix('.txt')
                with Image.open(image) as source:
                    width, height = source.size
                    rgb = source.convert('RGB')
                    digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                    pixels = list(rgb.convert('L').resize((9, 8)).tobytes())
                fingerprint = sum((pixels[r*9+c] > pixels[r*9+c+1]) << (r*8+c)
                                  for r in range(8) for c in range(8))
                key = image.relative_to(root).as_posix()
                if digest in exact:
                    report['issues'].append(f'Duplicate pixels: {key} / {exact[digest]}')
                exact[digest] = key
                # Equal dHash is a candidate only, never grounds for automatic deletion.
                for previous_hash, previous_images in fingerprints.items():
                    distance = (fingerprint ^ previous_hash).bit_count()
                    if distance <= 4:
                        for previous in previous_images:
                            if previous['split'] != split:
                                report['near_duplicate_candidates'].append(dict(first=previous['image'], second=key,
                                                                                dhash_distance=distance))
                fingerprints[fingerprint].append(dict(split=split, image=key))
                rows = label.read_text(encoding='utf-8').splitlines()
                negatives += not rows
                present = set()
                sources[image.name.split('_')[0]] += 1
                for row in rows:
                    cls, x, y, w, h = map(float, row.split())
                    if cls != int(cls) or not 0 <= cls < len(NAMES):
                        raise ValueError(f'Invalid class in {label}')
                    name = NAMES[int(cls)]
                    area = w*width*h*height
                    sizes[name]['small_<32px' if area < 32**2 else
                                'medium_<96px' if area < 96**2 else 'large'] += 1
                    boxes[name] += 1
                    present.add(name)
                positives.update(present)
            except (ValueError, OSError) as exc:
                report['issues'].append(f'{image}: {exc}')
        report['splits'][split] = dict(images=len(images), negative_candidates=negatives,
                                       sources=dict(sources), classes={name: dict(boxes=boxes[name],
                                       images=positives[name], pixel_area_bins=dict(sizes[name])) for name in NAMES})
    review = root / 'review.json'
    entries = json.loads(review.read_text(encoding='utf-8')).get('images', {}) if review.exists() else {}
    report['review_status'] = dict(total=len(entries), pending=sum(e.get('status') != 'approved'
                                                                for e in entries.values()))
    decisions_path = root / 'duplicate_review.json'
    decisions = json.loads(decisions_path.read_text(encoding='utf-8')) if decisions_path.exists() else {}
    for candidate in report['near_duplicate_candidates']:
        decision = decisions.get(candidate['first'] + '|' + candidate['second'], {})
        first, second = root / candidate['first'], root / candidate['second']
        valid = (decision.get('decision') == 'distinct' and decision.get('reviewer')
                 and decision.get('first_sha256') == hashlib.sha256(first.read_bytes()).hexdigest()
                 and decision.get('second_sha256') == hashlib.sha256(second.read_bytes()).hexdigest())
        candidate['review'] = 'distinct' if valid else 'pending'
    report['unresolved_duplicate_candidates'] = sum(c['review'] == 'pending'
                                                  for c in report['near_duplicate_candidates'])
    report['training_ready'] = False
    try:
        validate_dataset(config, require_review=True)
        if not report['unresolved_duplicate_candidates'] and not report['issues']:
            report['training_ready'] = True
    except (ValueError, OSError) as exc:
        report['review_blocker'] = str(exc)
    output = root / 'quality_report.json'
    root.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Quality report: {output}; training_ready={report["training_ready"]}', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('cctv_desk_data.yaml'))
    audit_quality(parser.parse_args().data.resolve())
