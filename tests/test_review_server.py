import concurrent.futures
import http.client
import json
import threading

import pytest

from tests.test_label_editor import patch_for
from tests.test_training_dataset import fixture_dataset
from tools.review_dataset import create_review
from tools.review_server import ReviewServer, ReviewStore


@pytest.fixture
def store(tmp_path):
    fixture_dataset(tmp_path)
    create_review(tmp_path)
    return ReviewStore(tmp_path)


def payload(store):
    patch = patch_for(store.root)
    patch['review_revision'] = store.state(patch['image'])['review_revision']
    return patch


def test_draft_does_not_approve_and_restore_is_recoverable(store):
    patch = payload(store)
    before = (store.root / 'labels/train/0.txt').read_text()
    state = store.apply('draft', patch)
    assert state['status'] == 'pending'
    assert state['progress']['approved'] == 0
    assert state['label_sha256'] != patch['label_sha256']
    restored = store.apply('restore', state)
    assert (store.root / 'labels/train/0.txt').read_text() == before
    assert restored['status'] == 'pending'


def test_concurrent_saves_cannot_overwrite_each_other(store):
    patch = payload(store)
    def save():
        try:
            store.apply('approve', patch)
            return 'saved'
        except ValueError:
            return 'stale'
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: save(), range(2)))
    assert sorted(results) == ['saved', 'stale']


def test_review_only_change_invalidates_other_editor(store):
    patch = payload(store)
    patch['boxes'] = [[0, .5, .5, .2, .2]]
    store.apply('approve', patch)
    assert store.state(patch['image'])['label_sha256'] == patch['label_sha256']
    with pytest.raises(ValueError, match='Stale review state'):
        store.apply('approve', patch)


def test_independent_store_instances_share_dataset_write_lock(store):
    other = ReviewStore(store.root)
    patch = payload(store)
    def save(target):
        try:
            target.apply('approve', patch)
            return 'saved'
        except ValueError:
            return 'stale'
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save, (store, other)))
    assert sorted(results) == ['saved', 'stale']


def test_draft_preserves_open_visual_finding(store):
    finding = store.root / 'review_findings.json'
    finding.write_text(json.dumps({'images/train/0.png': dict(status='open', reason='Wrong class')}))
    state = store.apply('draft', payload(store))
    assert state['finding'] == 'Wrong class'
    assert state['status'] == 'needs_correction'


def test_http_origin_token_path_and_save_round_trip(store):
    with ReviewServer(store.root, 0) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        def request(method, route, body=None, headers=None):
            connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
            connection.request(method, route, body=body, headers=headers or {})
            response = connection.getresponse()
            code, data = response.status, response.read()
            connection.close()
            return code, data
        try:
            assert request('GET', '/', headers={'Host': 'evil.example'})[0] == 403
            assert request('GET', '/images/../../review.json')[0] == 404
            assert request('GET', '/images/train/0.png')[0] == 200
            code, body = request('GET', '/')
            assert code == 200
            assert b'const api=null;' not in body
            assert request('POST', '/api/approve', '{}', {'Content-Type': 'application/json'})[0] == 403
            headers = {'Content-Type': 'application/json', 'X-Review-Token': server.token, 'Origin': server.origin}
            patch = payload(store)
            code, body = request('POST', '/api/approve', json.dumps(patch), headers)
            assert code == 200
            assert json.loads(body)['status'] == 'approved'
            assert request('POST', '/api/approve', json.dumps(patch), headers)[0] == 409
            headers['Origin'] = 'https://evil.example'
            assert request('POST', '/api/draft', json.dumps(patch), headers)[0] == 403
        finally:
            server.shutdown()
            worker.join(5)
