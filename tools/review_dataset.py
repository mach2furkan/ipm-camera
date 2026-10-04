"""Render every image with class-labelled boxes and build an unsigned review manifest.

Never approves annotations automatically. Reviewers must check every target object,
correct labels, regenerate previews, then sign matching image/label hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw

from ipcam.vision.desk import NAMES


def create_review(root: Path):
    root = root.resolve()
    path = root / 'review.json'
    previous = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    findings_path = root / 'review_findings.json'
    findings = json.loads(findings_path.read_text(encoding='utf-8')) if findings_path.exists() else {}
    entries, cards = {}, []
    for split in ('train', 'val'):
        for image in sorted((root / 'images' / split).rglob('*')):
            if image.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            relative = image.relative_to(root / 'images' / split)
            label = root / 'labels' / split / relative.with_suffix('.txt')
            if not label.exists():
                raise ValueError(f'Missing annotation: {label}')
            key = image.relative_to(root).as_posix()
            hashes = dict(image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                          label_sha256=hashlib.sha256(label.read_bytes()).hexdigest())
            old = previous.get('images', {}).get(key, {})
            unchanged = previous.get('names') == list(NAMES) and all(old.get(k) == v for k, v in hashes.items())
            entries[key] = old if unchanged else dict(**hashes, status='pending', reviewer='', session='',
                                                     all_target_objects_checked=False)
            if key in findings and findings[key].get('status') == 'open':
                entries[key] = dict(entries[key], status='needs_correction',
                                    all_target_objects_checked=False, finding=findings[key]['reason'])
            with Image.open(image) as source:
                canvas = source.convert('RGB')
            canvas.thumbnail((1600, 1600))
            draw = ImageDraw.Draw(canvas)
            width, height = canvas.size
            for row in label.read_text(encoding='utf-8').splitlines():
                cls, x, y, w, h = map(float, row.split())
                if cls != int(cls) or not 0 <= cls < len(NAMES):
                    raise ValueError(f'Invalid class: {label}')
                bounds = ((x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height)
                draw.rectangle(bounds, outline='red', width=3)
                draw.text((bounds[0], max(0, bounds[1]-12)), f'{int(cls)} {NAMES[int(cls)]}', fill='red',
                          stroke_width=1, stroke_fill='white')
            preview = root / 'review_previews' / split / relative.with_suffix('.jpg')
            preview.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(preview, quality=92)
            cards.append(f'<article><a href="{escape(preview.relative_to(root).as_posix(), quote=True)}">'
                         f'<img loading="lazy" src="{escape(preview.relative_to(root).as_posix(), quote=True)}" '
                         f'alt="{escape(key, quote=True)}"></a><p>{escape(key)}</p>'
                         f'<a href="{escape(key, quote=True)}">Original image</a> · '
                         f'<a href="{escape(label.relative_to(root).as_posix(), quote=True)}">YOLO labels</a>'
                         f'<p>Status: {escape(entries[key].get("status", "pending"))}</p></article>')
    manifest = dict(names=list(NAMES), images=entries)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    pages = max(1, (len(cards)+99)//100)
    navigation = ' '.join(f'<a href="review-{i+1}.html">{i+1}</a>' for i in range(pages))
    legend = ' | '.join(f'{i}: {escape(name)}' for i, name in enumerate(NAMES))
    for page in range(pages):
        html = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Dataset label review</title>'
                '<style>body{font:16px system-ui;background:#111;color:#eee;padding:24px}'
                'a{color:#7bd}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:20px}'
                'article{background:#222;padding:12px;overflow-wrap:anywhere}img{width:100%;height:auto}'
                'nav{line-height:2.5}</style><h1>Dataset label review</h1>'
                '<p>Check every target object in the original image. Previews do not approve labels.</p>'
                f'<p>{legend}</p><nav>{navigation}</nav><main>' + ''.join(cards[page*100:(page+1)*100])
                + f'</main><nav>{navigation}</nav></html>')
        (root / f'review-{page+1}.html').write_text(html, encoding='utf-8')
    print(f'Review previews: {root / "review_previews"}; pending: '
          f'{sum(e.get("status") != "approved" for e in entries.values())}/{len(entries)}', flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('dataset_cctv_desk'))
    create_review(parser.parse_args().root)
