import json
from tools.active_learning import build_queue
from tools.model_audit import sha
from ipcam.vision.desk import NAMES


def test_changed_approval_and_stale_teacher_are_never_skipped(tmp_path):
    image = tmp_path / 'images/train/a.jpg'
    label = tmp_path / 'labels/train/a.txt'
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    image.write_bytes(b'current image')
    label.write_text('0 .5 .5 .2 .2')
    key = 'images/train/a.jpg'
    review = dict(names=list(NAMES), images={key: dict(status='approved', image_sha256='stale',
                                                      label_sha256=sha(label), session='capture-1')})
    (tmp_path / 'review.json').write_text(json.dumps(review))
    (tmp_path / 'model_audit.jsonl').write_text(json.dumps(dict(image=key, image_sha256='stale',
                                                              label_sha256=sha(label), priority=999))+'\n')
    before = label.read_bytes()
    result = build_queue(tmp_path)
    assert len(result['selected']) == 1 and result['stale_audits'] == 1
    assert result['selected'][0]['priority'] == 0 and result['approvals_written'] == 0
    assert label.read_bytes() == before
