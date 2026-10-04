import json

import pytest

from ipcam.vision.desk import NAMES
from tests.test_training_dataset import fixture_dataset
from tools.label_editor import build_editor, digest, import_patch, stage_correction
from tools.review_dataset import create_review


def patch_for(root):
    return dict(names=list(NAMES), image='images/train/0.png',
                image_sha256=digest(root / 'images/train/0.png'),
                label_sha256=digest(root / 'labels/train/0.txt'),
                boxes=[[0, .5, .5, .3, .3]], reviewer='Reviewer', session='session-a',
                all_target_objects_checked=True, finding_resolved=False)


def test_corrected_label_saved_with_recoverable_history(tmp_path):
    fixture_dataset(tmp_path)
    create_review(tmp_path)
    patch = patch_for(tmp_path)
    entry = import_patch(tmp_path, patch)
    assert entry['status'] == 'approved'
    label = tmp_path / 'labels/train/0.txt'
    assert entry['label_sha256'] == digest(label)
    assert label.read_text() == '0 0.5 0.5 0.3 0.3'
    history = json.loads((tmp_path / 'review_history/images/train/0.json').read_text())
    assert history[0]['previous_label'] == '0 0.5 0.5 0.2 0.2'
    with pytest.raises(ValueError, match='Stale'):
        import_patch(tmp_path, patch)


@pytest.mark.parametrize('change,message', [
    ({'all_target_objects_checked': False}, 'every target'),
    ({'reviewer': ''}, 'identity'),
    ({'session': ''}, 'session required'),
    ({'image': '../outside.jpg'}, 'Invalid dataset'),
    ({'boxes': [[10, .5, .5, .2, .2]]}, 'class ID'),
    ({'boxes': [[0, .99, .5, .2, .2]]}, 'outside'),
    ({'boxes': [[0, .5, .5, float('nan'), .2]]}, 'Non-finite'),
])
def test_invalid_patch_leaves_labels_unchanged(tmp_path, change, message):
    fixture_dataset(tmp_path)
    create_review(tmp_path)
    patch = patch_for(tmp_path)
    patch.update(change)
    before = (tmp_path / 'labels/train/0.txt').read_bytes()
    with pytest.raises(ValueError, match=message):
        import_patch(tmp_path, patch)
    assert (tmp_path / 'labels/train/0.txt').read_bytes() == before


def test_finding_requires_explicit_resolution(tmp_path):
    fixture_dataset(tmp_path)
    create_review(tmp_path)
    path = tmp_path / 'review_findings.json'
    path.write_text(json.dumps({'images/train/0.png': dict(status='open', reason='Missing object')}))
    patch = patch_for(tmp_path)
    with pytest.raises(ValueError, match='Explicit resolution'):
        import_patch(tmp_path, patch)
    patch['finding_resolved'] = True
    import_patch(tmp_path, patch)
    assert json.loads(path.read_text())['images/train/0.png']['status'] == 'resolved'


def test_editor_embeds_data_without_file_fetch(tmp_path):
    fixture_dataset(tmp_path)
    create_review(tmp_path)
    html = build_editor(tmp_path).read_text(encoding='utf-8')
    assert '/*DATA*/' not in html
    assert 'const api=null;' in html
    assert 'images/train/0.png' in html


def test_partial_correction_never_approves_image(tmp_path):
    fixture_dataset(tmp_path)
    create_review(tmp_path)
    stage_correction(tmp_path, 'images/train/0.png', [[0, .5, .5, .3, .3]], 'Partial visual correction', 'Codex')
    review = json.loads((tmp_path / 'review.json').read_text())
    entry = review['images']['images/train/0.png']
    assert entry['status'] == 'needs_correction'
    assert entry['all_target_objects_checked'] is False
    assert entry['label_sha256'] == digest(tmp_path / 'labels/train/0.txt')
