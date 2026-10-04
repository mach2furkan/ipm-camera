import numpy as np
import pytest

from ipcam.analytics import ByteTrackConfig, ByteTracker
from ipcam.vision.desk import prepare_detections


def test_pen_aliases_merge_without_merging_distinct_classes():
    boxes = [[10, 10, 12, 80], [10, 10, 12, 80], [10, 10, 12, 80]]
    output = prepare_detections(boxes, [.7, .8, .9], [5, 6, 7], 200, 200, [.1]*10,
                                (0, 1, 2, 3, 4, 9, 9, 5, 6, 7, 8))
    assert output[:, 5].tolist() == [5, 9]
    assert output[:, 4].tolist() == [.9, .8]


def test_invalid_detector_values_do_not_index_class_arrays():
    output = prepare_detections([[10, 10, 30, 30]]*5,
                                [.9, .9, float('nan'), .9, .9],
                                [float('nan'), 999, 0, 1.5, 0], 100, 100, [.3]*10)
    assert len(output) == 1
    assert output[0, 5] == 0


def test_low_confidence_pen_confirms_without_score_fusion_identity_loss():
    tracker = ByteTracker(ByteTrackConfig(high_thresh=.036, low_thresh=.018,
                                          new_track_thresh=.036, fuse_score=False,
                                          min_hits=2, confirm_first_frame=False, mahalanobis_gate=None))
    detection = np.array([[10, 10, 12, 80, .13, 9]])
    assert tracker.update(detection, 0).tracks == []
    assert [t.cls for t in tracker.update(detection, .04).tracks] == [9]
    tracker.reset()
    assert tracker.frame_count == 0
    assert tracker.update(detection, .08).tracks == []


def test_expired_identity_not_resurrected_after_gap():
    tracker = ByteTracker(ByteTrackConfig(min_hits=1, lost_ttl_s=.5))
    detection = np.array([[10, 10, 30, 30, .9, 0]])
    first = tracker.update(detection, 0).tracks[0].track_id
    output = tracker.update(detection, 2)
    assert output.tracks[0].track_id != first
    assert first in output.removed_ids


def test_overlapping_different_class_does_not_delete_lost_track():
    tracker = ByteTracker(ByteTrackConfig(min_hits=1))
    person = np.array([[10, 10, 30, 30, .9, 0]])
    book = np.array([[10, 10, 30, 30, .9, 1]])
    first = tracker.update(person, 0).tracks[0].track_id
    output = tracker.update(book, .04)
    assert first in [track.track_id for track in output.lost]


def test_bad_input_does_not_change_tracker_state():
    tracker = ByteTracker()
    with pytest.raises(ValueError, match='shaped'):
        tracker.update(np.zeros((1, 7)), 10)
    assert tracker.frame_count == 0
    assert tracker._t is None
    with pytest.raises(ValueError, match='timestamp'):
        tracker.update(np.zeros((0, 6)), float('nan'))


def test_non_finite_detections_do_not_poison_kalman_state():
    tracker = ByteTracker(ByteTrackConfig(min_hits=1))
    tracker.update(np.array([[10, 10, 30, 30, .9, 0], [float('nan'), 0, 30, 30, .9, 0]]), 0)
    assert len(tracker.tracked) == 1
    assert np.isfinite(tracker.tracked[0].mean).all()
