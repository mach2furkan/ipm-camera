"""Prepare a reproducible COCO + Open Images baseline, retaining provenance.

Source annotations are not exhaustive for all ten target classes. The output is
a public-data baseline; camera acceptance remains a separate required step.
"""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import random
import time
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from tools.audit_openimages import IDS, main as audit

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / 'dataset_sources'
DEST = ROOT / 'dataset_cctv_desk'
COCO = {1: 0, 84: 1, 73: 2, 72: 3, 76: 4, 74: 5, 77: 6, 47: 7, 44: 8}


def download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            tmp = path.with_suffix(path.suffix + '.part')
            with urllib.request.urlopen(url, timeout=30) as source, tmp.open('wb') as out:
                while chunk := source.read(1024 * 1024):
                    out.write(chunk)
            tmp.replace(path)
            return
        except Exception as exc:
            print(f'Download attempt {attempt+1}/4 failed: {path.name}: {exc}', flush=True)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def coco_jobs():
    archive = CACHE / 'coco/annotations_trainval2017.zip'
    download('https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip', archive)
    rng = random.Random(42)
    jobs = []
    with zipfile.ZipFile(archive) as z:
        for source, split, positives, negatives in [('train2017', 'train', 8333, 1250),
                                                    ('val2017', 'val', 1667, 250)]:
            data = json.loads(z.read(f'annotations/instances_{source}.json'))
            expected = {1: 'person', 84: 'book', 73: 'laptop', 72: 'tv', 76: 'keyboard',
                        74: 'mouse', 77: 'cell phone', 47: 'cup', 44: 'bottle'}
            categories = {c['id']: c['name'] for c in data['categories']}
            if any(categories.get(k) != v for k, v in expected.items()):
                raise ValueError('COCO category definitions do not match the fixed mapping')
            boxes, unsuitable = defaultdict(list), set()
            info = {im['id']: im for im in data['images']}
            for ann in data['annotations']:
                if ann['category_id'] not in COCO:
                    continue
                iid = ann['image_id']
                x, y, w, h = ann['bbox']
                if ann['iscrowd'] or w * h <= 16 or w <= 0 or h <= 0:
                    unsuitable.add(iid)
                    continue
                im = info[iid]
                W, H = im['width'], im['height']
                x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x+w), min(H, y+h)
                boxes[iid].append([COCO[ann['category_id']], (x1+x2)/2/W,
                                   (y1+y2)/2/H, (x2-x1)/W, (y2-y1)/H])
            candidates = sorted(set(boxes) - unsuitable)
            rng.shuffle(candidates)
            # Balance selection: retain rare desktop classes before common people.
            chosen = set()
            for cls in (5, 4, 6, 2, 3, 1, 7, 8, 0):
                pool = [i for i in candidates if any(b[0] == cls for b in boxes[i])]
                chosen.update(pool[:min(len(pool), positives//9)])
            for iid in candidates:
                if len(chosen) >= positives:
                    break
                chosen.add(iid)
            empty = sorted(set(info) - set(boxes) - unsuitable)
            rng.shuffle(empty)
            for iid in sorted(chosen) + empty[:negatives]:
                im = info[iid]
                jobs.append(dict(id=f'coco_{source}_{iid}', split=split,
                                 url=im['coco_url'].replace('http://images.cocodataset.org/',
                                                          'https://s3.amazonaws.com/images.cocodataset.org/'), boxes=boxes[iid],
                                 source='COCO2017', negative_candidate=not boxes[iid]))
    return jobs


def oi_jobs():
    from tools.audit_openimages import verify_class_definitions
    verify_class_definitions()
    if not (CACHE / 'openimages/audit.json').exists():
        audit()
    jobs = []
    for source, split, limit in [('train', 'train', 2917), ('validation', 'val', 583)]:
        print(f'Planning Open Images {source}', flush=True)
        metadata = CACHE / 'openimages' / f'{source}-boxes.csv'
        pens = json.loads((CACHE / 'openimages' / f'{source}-pens.json').read_text())
        candidates = sorted({r['ImageID'] for r in pens})
        random.Random(42).shuffle(candidates)
        selected = set(candidates[:limit])
        # There are only ~1,011 training pen images. Supplement with desk classes;
        # never fabricate the requested 3,500 distinct pen samples.
        pools = defaultdict(set)
        with metadata.open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                if row['LabelName'] in IDS[1:]:
                    pools[row['LabelName']].add(row['ImageID'])
        rng = random.Random(42)
        for cls in IDS[1:-1]:
            pool = sorted(pools[cls] - selected)
            rng.shuffle(pool)
            selected.update(pool[:max(0, (limit-len(selected))//(len(IDS)-IDS.index(cls)-1))])
        boxes, unsuitable = defaultdict(list), set()
        with metadata.open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                iid = row['ImageID']
                if iid not in selected or row['LabelName'] not in IDS:
                    continue
                if any(row.get(flag) == '1' for flag in ('IsGroupOf', 'IsDepiction', 'IsInside')):
                    unsuitable.add(iid)
                    continue
                x1, x2, y1, y2 = [float(row[k]) for k in ('XMin', 'XMax', 'YMin', 'YMax')]
                boxes[iid].append([IDS.index(row['LabelName']), (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1])
        for iid in sorted(selected - unsuitable):
            jobs.append(dict(id=f'oi_{source}_{iid}', split=split,
                             url=f'https://open-images-dataset.s3.amazonaws.com/{source}/{iid}.jpg',
                             boxes=boxes[iid], source=f'OpenImages {"V6" if source == "train" else "V5"} boxes',
                             negative_candidate=False))
    return jobs


def materialize(job):
    staged = CACHE / 'images' / job['split'] / (job['id'] + '.jpg')
    image = DEST / 'images' / job['split'] / (job['id'] + '.jpg')
    label = DEST / 'labels' / job['split'] / (job['id'] + '.txt')
    download(job['url'], staged)
    with Image.open(staged) as im:
        W, H = im.size
        im.verify()
    for box in job['boxes']:
        cls, x, y, w, h = box
        if (not all(math.isfinite(v) for v in box) or cls != int(cls) or not 0 <= cls < 10
                or w <= 0 or h <= 0 or x-w/2 < -1e-6 or y-h/2 < -1e-6
                or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6):
            raise ValueError(f'Invalid source annotation: {job["id"]}')
    if any(b[3] * W * b[4] * H <= 16 for b in job['boxes']):
        # Reject the entire image rather than teach unlabelled tiny positives as background.
        return dict(id=job['id'], rejected='tiny box', image=str(staged))
    image.parent.mkdir(parents=True, exist_ok=True)
    if not image.exists():
        image.write_bytes(staged.read_bytes())
    label.parent.mkdir(parents=True, exist_ok=True)
    # Preserve corrections made during review on subsequent preparation runs.
    if not label.exists():
        label.write_text('\n'.join(' '.join(map(str, box)) for box in job['boxes']), encoding='utf-8')
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('openimages', 'mixed'), default='openimages',
                        help='Use real monitor annotations by default; mixed COCO requires TV/pen review')
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    from tools.audit_openimages import verify_class_definitions
    verify_class_definitions()
    # Plan Open Images first: fail before downloading thousands of photos if metadata is unavailable.
    oi_plan = CACHE / 'openimages/download_plan_v2.json'
    if oi_plan.exists():
        oi = json.loads(oi_plan.read_text(encoding='utf-8'))
    else:
        oi = oi_jobs()
        oi_plan.write_text(json.dumps(oi), encoding='utf-8')
    for job in oi:
        job['source'] = f'OpenImages {"V6" if job["split"] == "train" else "V5"} boxes'
    print(f'Open Images plan: {len(oi)} images; source={args.source}', flush=True)
    jobs = oi + coco_jobs() if args.source == 'mixed' else oi
    (CACHE / 'download_plan.json').write_text(json.dumps(jobs), encoding='utf-8')
    failures, completed = [], []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(materialize, job): job for job in jobs}
        for n, future in enumerate(as_completed(futures), 1):
            try:
                completed.append(future.result())
            except Exception as exc:
                failures.append(dict(id=futures[future]['id'], error=str(exc)))
            if n % 100 == 0:
                print(f'{n}/{len(jobs)} prepared; failures={len(failures)}', flush=True)
    (CACHE / 'provenance.json').write_text(json.dumps(completed), encoding='utf-8')
    (CACHE / 'failures.json').write_text(json.dumps(failures), encoding='utf-8')
    # Exact duplicate photos across sources/splits are quarantined before training.
    seen = {}
    for split in ('val', 'train'):
        for image in sorted((DEST / 'images' / split).glob('*.jpg')):
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in seen:
                quarantine = CACHE / 'duplicates' / split
                quarantine.mkdir(parents=True, exist_ok=True)
                image.replace(quarantine / image.name)
                label = DEST / 'labels' / split / image.with_suffix('.txt').name
                label.replace(quarantine / label.name)
            else:
                seen[digest] = str(image)
    print(f'Prepared {len(completed)}; failed {len(failures)}. Public baseline needs camera review.', flush=True)
    from tools.review_dataset import create_review
    create_review(DEST)
    from tools.dataset_quality import audit_quality
    audit_quality(ROOT / 'cctv_desk_data.yaml')
    if failures:
        raise RuntimeError('Image downloads failed; inspect dataset_sources/failures.json before proceeding')


if __name__ == '__main__':
    main()
