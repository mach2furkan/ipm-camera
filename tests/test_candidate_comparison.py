import copy
import pytest
from ipcam.vision.desk import NAMES
from tools.compare_candidates import compare


def inputs():
    frames = [dict(image=f'{c}.jpg', image_sha256=f'{c:064x}', session='test-session',
                   reviewed=True, held_out=True, truth=[[c, .5, .5, .2, .2]],
                   predictions=[], latency_ms=10) for c in range(10)]
    frames.append(dict(image='empty.jpg', image_sha256='f'*64, session='test-session',
                       reviewed=True, held_out=True, truth=[], predictions=[], latency_ms=10))
    old = dict(names=list(NAMES), frames=frames, measurement=dict(latency_scope='predict and postprocess'))
    new = copy.deepcopy(old)
    for frame in new['frames'][:-1]:
        frame['predictions'] = [dict(box=frame['truth'][0], confidence=.9)]
    return old, new


def test_improving_candidate_is_eligible_but_never_deployed():
    old, new = inputs()
    result = compare(old, new, min_support=1, min_negative_frames=1)
    assert result['eligible_for_review'] and not result['deployed']


def test_one_class_regression_blocks_other_class_improvements():
    old, new = inputs()
    old['frames'][0]['predictions'] = [dict(box=old['frames'][0]['truth'][0], confidence=.9)]
    new['frames'][0]['predictions'] = []
    assert not compare(old, new, min_support=1, min_negative_frames=1)['eligible_for_review']


@pytest.mark.parametrize('field,value', [('image_sha256', 'a'*64), ('session', 'different'),
                                        ('truth', [[0, .4, .5, .2, .2]])])
def test_changed_evaluation_refused(field, value):
    old, new = inputs()
    new['frames'][0][field] = value
    with pytest.raises(ValueError, match='identical'):
        compare(old, new)


def test_missing_support_latency_regression_and_identical_models_blocked():
    old, new = inputs()
    assert not compare(old, new)['eligible_for_review']
    new['frames'][0]['latency_ms'] = 100
    assert not compare(old, new, min_support=1, min_negative_frames=1)['eligible_for_review']
    assert not compare(old, old, min_support=1, min_negative_frames=1)['eligible_for_review']
