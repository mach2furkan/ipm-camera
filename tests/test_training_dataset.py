from pathlib import Path

import pytest
import yaml
from PIL import Image

from ipcam.vision.desk import NAMES
from run_pipeline import validate_dataset, validate_model_classes


def fixture_dataset(root: Path):
    config = root / 'data.yaml'
    config.write_text(yaml.safe_dump(dict(path=str(root), train='images/train',
                                        val='images/val', names=list(NAMES))))
    for split, colour in [('train', 40), ('val', 180)]:
        (root / 'images' / split).mkdir(parents=True)
        (root / 'labels' / split).mkdir(parents=True)
        for i in range(11):
            Image.new('RGB', (100, 100), (colour, i, 0)).save(root / 'images' / split / f'{i}.png')
            text = f'{i} 0.5 0.5 0.2 0.2' if i < 10 else ''
            (root / 'labels' / split / f'{i}.txt').write_text(text)
    return config


def test_valid_dataset_counts(tmp_path):
    report = validate_dataset(fixture_dataset(tmp_path))
    assert report['train'] == dict(images=11, boxes=[1]*10, negatives=1)


def test_candidate_can_lack_negatives_without_weakening_default_checks(tmp_path):
    config = fixture_dataset(tmp_path)
    (tmp_path / 'images/train/10.png').unlink()
    (tmp_path / 'labels/train/10.txt').unlink()
    with pytest.raises(ValueError, match='hard negatives'):
        validate_dataset(config)
    assert validate_dataset(config, require_negatives=False)['train']['negatives'] == 0
    (tmp_path / 'labels/train/0.txt').unlink()
    with pytest.raises(ValueError, match='Missing annotation'):
        validate_dataset(config, require_negatives=False)


def test_missing_labels_are_not_negatives(tmp_path):
    config = fixture_dataset(tmp_path)
    (tmp_path / 'labels/train/10.txt').unlink()
    with pytest.raises(ValueError, match='Missing annotation'):
        validate_dataset(config)


def test_duplicate_split_rejected(tmp_path):
    config = fixture_dataset(tmp_path)
    (tmp_path / 'images/val/0.png').write_bytes((tmp_path / 'images/train/0.png').read_bytes())
    with pytest.raises(ValueError, match='split leakage'):
        validate_dataset(config)


def test_bad_box_rejected(tmp_path):
    config = fixture_dataset(tmp_path)
    (tmp_path / 'labels/train/0.txt').write_text('0 0.98 0.5 0.2 0.2')
    with pytest.raises(ValueError, match='Out of bounds'):
        validate_dataset(config)


def test_extra_class_id_rejected(tmp_path):
    config = fixture_dataset(tmp_path)
    cfg = yaml.safe_load(config.read_text())
    cfg['names'] = dict(enumerate(NAMES)) | {10: 'unexpected'}
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='exactly 0..9'):
        validate_dataset(config)


def test_duplicate_annotation_rejected(tmp_path):
    config = fixture_dataset(tmp_path)
    (tmp_path / 'labels/train/0.txt').write_text('0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2')
    with pytest.raises(ValueError, match='Duplicate annotation'):
        validate_dataset(config)


def test_orphan_annotation_rejected(tmp_path):
    config = fixture_dataset(tmp_path)
    (tmp_path / 'labels/train/orphan.txt').write_text('')
    with pytest.raises(ValueError, match='Orphan annotation'):
        validate_dataset(config)


def test_reencoded_duplicate_rejected(tmp_path):
    config = fixture_dataset(tmp_path)
    with Image.open(tmp_path / 'images/train/0.png') as im:
        im.save(tmp_path / 'images/val/0.png', compress_level=0)
    with pytest.raises(ValueError, match='split leakage'):
        validate_dataset(config)


def approve_fixture(root):
    import json
    from tools.review_dataset import create_review
    review = create_review(root)
    for key, entry in review['images'].items():
        entry.update(status='approved', reviewer='test reviewer', session=key.split('/')[1],
                     all_target_objects_checked=True)
    (root / 'review.json').write_text(json.dumps(review))


def test_review_required(tmp_path):
    config = fixture_dataset(tmp_path)
    with pytest.raises(ValueError, match='manifest missing'):
        validate_dataset(config, require_review=True)
    approve_fixture(tmp_path)
    assert validate_dataset(config, require_review=True)['train']['images'] == 11
    (tmp_path / 'labels/train/0.txt').write_text('0 0.5 0.5 0.3 0.3')
    with pytest.raises(ValueError, match='changed image/labels'):
        validate_dataset(config, require_review=True)


def test_session_leakage_rejected(tmp_path):
    import json
    config = fixture_dataset(tmp_path)
    approve_fixture(tmp_path)
    path = tmp_path / 'review.json'
    review = json.loads(path.read_text())
    for entry in review['images'].values():
        entry['session'] = 'same-camera-session'
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match='session split leakage'):
        validate_dataset(config, require_review=True)


def test_official_class_mismatch_rejected(tmp_path, monkeypatch):
    from tools import audit_openimages as audit
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    (tmp_path / 'class-descriptions-boxable.csv').write_text('/m/02522,Television\n')
    with pytest.raises(ValueError, match='class mismatch'):
        audit.verify_class_definitions()


def test_export_class_order_checked():
    validate_model_classes(dict(enumerate(NAMES)))
    swapped = list(NAMES)
    swapped[3], swapped[9] = swapped[9], swapped[3]
    with pytest.raises(ValueError, match='names/order changed'):
        validate_model_classes(swapped)
