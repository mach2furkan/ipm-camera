import json
from ipcam.vision.desk import NAMES
from tools.training_watch import readiness


def test_approvals_alone_do_not_resolve_open_findings(tmp_path):
    (tmp_path / 'review.json').write_text(json.dumps(dict(names=list(NAMES), images={'a': {'status': 'approved'}})))
    (tmp_path / 'review_findings.json').write_text(json.dumps({'a': {'status': 'open'}}))
    assert not readiness(tmp_path)['ready_for_validation']
    (tmp_path / 'review_findings.json').write_text(json.dumps({'a': {'status': 'resolved'}}))
    assert readiness(tmp_path)['ready_for_validation']


def test_empty_manifest_never_starts_training(tmp_path):
    (tmp_path / 'review.json').write_text(json.dumps(dict(names=list(NAMES), images={})))
    assert not readiness(tmp_path)['ready_for_validation']
