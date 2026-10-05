"""Build an isolated official COCO subset with all 80 source categories retained."""
import argparse
import hashlib
import json
import random
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from PIL import Image

from tools.label_editor import atomic_json
from tools.prepare_desk_data import download

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / 'dataset_sources/coco_world_candidate'
URL = 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip'
TARGETS = (76, 74, 73, 77, 84, 47, 44, 1)


def select_ids(data, limit, seed=42):
    if limit < 1:
        raise ValueError('Positive subset size required')
    images = {im['id']: im for im in data['images']}
    boxes, invalid = defaultdict(list), set()
    for annotation in data['annotations']:
        iid = annotation['image_id']
        if annotation.get('iscrowd', 0):
            invalid.add(iid)  # YOLO cannot represent COCO crowd ignore regions.
            continue
        boxes[iid].append(annotation)
    eligible = set(boxes) - invalid
    rng = random.Random(seed)
    chosen = set()
    # Rare desk categories first, then fill with diverse original COCO categories.
    for category in TARGETS:
        pool = sorted(i for i in eligible if any(a['category_id'] == category for a in boxes[i]))
        rng.shuffle(pool)
        chosen.update(pool[:min(len(pool), limit//(2*len(TARGETS)))])
    remaining = sorted(eligible - chosen)
    rng.shuffle(remaining)
    chosen.update(remaining[:max(0, limit-len(chosen))])
    if len(chosen) != limit:
        raise ValueError('Insufficient source images for requested subset')
    return [(images[i], boxes[i]) for i in sorted(chosen)]


def convert_boxes(annotations, width, height, category_map):
    rows = []
    for annotation in annotations:
        x, y, w, h = annotation['bbox']
        if w <= 0 or h <= 0:
            raise ValueError('Invalid official COCO box')
        left, top, right, bottom = max(0, x), max(0, y), min(width, x+w), min(height, y+h)
        if right <= left or bottom <= top:
            raise ValueError('Official COCO box outside image')
        rows.append([category_map[annotation['category_id']], (left+right)/2/width,
                     (top+bottom)/2/height, (right-left)/width, (bottom-top)/height])
    return rows


def prepare(train_count=5000, val_count=500):
    archive = ROOT / 'dataset_sources/coco/annotations_trainval2017.zip'
    download(URL, archive)
    with zipfile.ZipFile(archive) as source:
        datasets = {split: json.loads(source.read(f'annotations/instances_{split}2017.json'))
                    for split in ('train', 'val')}
    categories = sorted(datasets['train']['categories'], key=lambda item: item['id'])
    if categories != sorted(datasets['val']['categories'], key=lambda item: item['id']) or len(categories) != 80:
        raise ValueError('Official source taxonomy mismatch')
    mapping = {c['id']: i for i, c in enumerate(categories)}
    names = {i: c['name'] for i, c in enumerate(categories)}
    jobs = [(split, image, annotations) for split, count in (('train', train_count), ('val', val_count))
            for image, annotations in select_ids(datasets[split], count)]
    DEST.mkdir(parents=True, exist_ok=True)
    state = DEST / 'preparation_status.json'
    atomic_json(state, dict(stage='downloading_images', total=len(jobs), completed=0))

    def materialize(job):
        split, image, annotations = job
        path = DEST / 'images' / split / image['file_name']
        download('https://s3.amazonaws.com/images.cocodataset.org/' + split+'2017/'+image['file_name'], path)
        with Image.open(path) as im:
            if im.size != (image['width'], image['height']):
                raise ValueError('Source image size mismatch')
            im.verify()
        rows = convert_boxes(annotations, image['width'], image['height'], mapping)
        label = DEST / 'labels' / split / Path(image['file_name']).with_suffix('.txt')
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text('\n'.join(str(int(row[0]))+' '+' '.join(format(v, '.12g') for v in row[1:]) for row in rows),
                         encoding='utf-8')
        return dict(split=split, image=path.relative_to(DEST).as_posix(),
                    image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    label_sha256=hashlib.sha256(label.read_bytes()).hexdigest(),
                    source_image_id=image['id'], boxes=len(rows))

    records, failures = [], []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(materialize, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                records.append(future.result())
            except Exception as exc:
                failures.append(dict(image=futures[future][1]['file_name'], error=str(exc)))
            if (len(records)+len(failures)) % 100 == 0:
                atomic_json(state, dict(stage='downloading_images', total=len(jobs), completed=len(records),
                                        failed=len(failures)))
                print(f'{len(records)}/{len(jobs)} images; failures={len(failures)}', flush=True)
    atomic_json(DEST / 'provenance.json', dict(source=URL, archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                                              names=names, records=records, failures=failures))
    if failures:
        raise ValueError('Dataset download incomplete; no training started')
    seen = {}
    for record in records:
        digest = record['image_sha256']
        if digest in seen:
            raise ValueError('Duplicate source image across selected dataset')
        seen[digest] = record['image']
    # Lists pin exact subsets; stale cached images cannot silently enter training.
    for split in ('train', 'val'):
        (DEST / f'{split}.txt').write_text('\n'.join(str((DEST / r['image']).resolve())
                                                    for r in sorted(records, key=lambda r: r['image']) if r['split'] == split),
                                         encoding='utf-8')
    config = DEST / 'data.yaml'
    config.write_text(yaml.safe_dump(dict(path=DEST.as_posix(), train='train.txt', val='val.txt', names=names),
                                     sort_keys=False), encoding='utf-8')
    atomic_json(state, dict(stage='ready', completed=len(records), splits=dict(Counter(r['split'] for r in records)),
                            source_labels='official COCO annotations; not camera-reviewed',
                            camera_acceptance='unverified'))
    print(f'Official subset ready: {config}', flush=True)
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-count', type=int, default=5000)
    parser.add_argument('--val-count', type=int, default=500)
    args = parser.parse_args()
    prepare(args.train_count, args.val_count)
