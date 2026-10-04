import json

from PIL import Image

from tests.test_training_dataset import fixture_dataset
from tools.dataset_quality import audit_quality
from tools import prepare_desk_data as preparation


def test_quality_report_does_not_approve_fixture(tmp_path):
    config = fixture_dataset(tmp_path)
    report = audit_quality(config)
    assert report['training_ready'] is False
    assert report['splits']['train']['classes']['kalem']['boxes'] == 1
    assert report['splits']['train']['negative_candidates'] == 1
    assert report['near_duplicate_candidates']
    assert json.loads((tmp_path / 'quality_report.json').read_text())['training_ready'] is False


def test_visual_finding_blocks_review(tmp_path):
    import pytest
    from run_pipeline import validate_dataset
    from tools.review_dataset import create_review
    from tests.test_training_dataset import approve_fixture
    config = fixture_dataset(tmp_path)
    approve_fixture(tmp_path)
    (tmp_path / 'review_findings.json').write_text(json.dumps({
        'images/train/0.png': dict(status='open', reason='Missing target object')}))
    with pytest.raises(ValueError, match='Unresolved visual'):
        validate_dataset(config, require_review=True)
    review = create_review(tmp_path)
    assert review['images']['images/train/0.png']['status'] == 'needs_correction'
    assert (tmp_path / 'review-1.html').exists()


def test_preparation_preserves_corrected_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, 'CACHE', tmp_path / 'cache')
    monkeypatch.setattr(preparation, 'DEST', tmp_path / 'dataset')
    staged = tmp_path / 'cache/images/train/example.jpg'
    staged.parent.mkdir(parents=True)
    Image.new('RGB', (100, 100)).save(staged)
    job = dict(id='example', split='train', url='not-used', boxes=[[0, .5, .5, .2, .2]])
    preparation.materialize(job)
    label = tmp_path / 'dataset/labels/train/example.txt'
    label.write_text('9 0.5 0.5 0.3 0.3')
    preparation.materialize(job)
    assert label.read_text() == '9 0.5 0.5 0.3 0.3'


def test_duplicate_decisions_invalidated_when_images_change(tmp_path):
    import hashlib
    config = fixture_dataset(tmp_path)
    report = audit_quality(config)
    decisions = {}
    for pair in report['near_duplicate_candidates']:
        decisions[pair['first']+'|'+pair['second']] = dict(decision='distinct', reviewer='reviewer',
            first_sha256=hashlib.sha256((tmp_path / pair['first']).read_bytes()).hexdigest(),
            second_sha256=hashlib.sha256((tmp_path / pair['second']).read_bytes()).hexdigest())
    (tmp_path / 'duplicate_review.json').write_text(json.dumps(decisions))
    assert audit_quality(config)['unresolved_duplicate_candidates'] == 0
    first = report['near_duplicate_candidates'][0]['first']
    with Image.open(tmp_path / first) as source:
        source.save(tmp_path / first, compress_level=0)
    assert audit_quality(config)['unresolved_duplicate_candidates'] > 0
