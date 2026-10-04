import pytest

from ipcam.vision.desk import NAMES
from tools.evaluate_camera import evaluate


def frame(truth, predictions):
    return dict(image='frame.jpg', session='held-out-1', reviewed=True, held_out=True,
                truth=truth, predictions=predictions)


def test_duplicate_predictions_and_wrong_class_are_errors():
    box = [9, .5, .5, .2, .2]
    data = dict(names=list(NAMES), frames=[frame([box], [dict(box=box, confidence=.9),
                                                            dict(box=box, confidence=.8)])])
    report = evaluate(data)
    assert report['per_class']['kalem']['tp'] == 1
    assert report['per_class']['kalem']['fp'] == 1
    data['frames'][0]['predictions'] = [dict(box=[5, .5, .5, .2, .2], confidence=.9)]
    report = evaluate(data)
    assert report['per_class']['kalem']['fn'] == 1
    assert report['per_class']['fare']['fp'] == 1
    assert report['confusion_matrix'][9][5] == 1


def test_empty_frames_and_missing_class_are_not_fake_success():
    data = dict(names=list(NAMES), frames=[frame([], [dict(box=[9, .5, .5, .2, .2], confidence=.9)])])
    report = evaluate(data)
    assert report['negative_frame_false_alarm_rate'] == 1
    assert report['per_class']['kalem']['recall'] is None
    assert report['per_class']['insan']['precision'] is None


def test_unreviewed_frame_refused():
    data = dict(names=list(NAMES), frames=[frame([], [])])
    data['frames'][0]['reviewed'] = False
    with pytest.raises(ValueError, match='Reviewed held-out'):
        evaluate(data)
