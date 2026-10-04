"""Download official box metadata and audit pen availability; resumable on rerun."""
import csv
import json
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / 'dataset_sources' / 'openimages'
URLS = {
    'train': 'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv',
    'validation': 'https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv',
}
IDS = ['/m/01g317', '/m/0bt_c3', '/m/01c648', '/m/02522', '/m/01m2v',
       '/m/020lf', '/m/050k8', '/m/02p5f1q', '/m/04dr76w', '/m/0k1tl']

SOURCE_NAMES = ('Person', 'Book', 'Laptop', 'Computer monitor', 'Computer keyboard',
                'Computer mouse', 'Mobile phone', 'Coffee cup', 'Bottle', 'Pen')


def verify_class_definitions():
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / 'class-descriptions-boxable.csv'
    if not path.exists():
        with urllib.request.urlopen('https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv',
                                    timeout=120) as source:
            path.write_bytes(source.read())
    with path.open(encoding='utf-8', newline='') as stream:
        definitions = dict(row for row in csv.reader(stream) if row)
    for mid, expected in zip(IDS, SOURCE_NAMES):
        if definitions.get(mid) != expected:
            raise ValueError(f'Open Images class mismatch: {mid}: {definitions.get(mid)!r} != {expected!r}')
    return dict(zip(IDS, SOURCE_NAMES))


def main():
    verify_class_definitions()
    ROOT.mkdir(parents=True, exist_ok=True)
    report = {}
    for split, url in URLS.items():
        file = ROOT / f'{split}-boxes.csv'
        if not file.exists():
            tmp = file.with_suffix('.part')
            print(f'Downloading {url}', flush=True)
            with urllib.request.urlopen(url, timeout=120) as source, tmp.open('wb') as dest:
                while block := source.read(1024 * 1024):
                    dest.write(block)
            tmp.replace(file)
        counts, images, pens = Counter(), {k: set() for k in IDS}, []
        with file.open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                cls = row['LabelName']
                if cls not in images:
                    continue
                counts[cls] += 1
                images[cls].add(row['ImageID'])
                if cls == IDS[-1]:
                    pens.append(row)
        report[split] = {cls: dict(boxes=counts[cls], images=len(images[cls])) for cls in IDS}
        (ROOT / f'{split}-pens.json').write_text(json.dumps(pens), encoding='utf-8')
        print(json.dumps({split: report[split]}, indent=2), flush=True)
    (ROOT / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
