"""Score explicit held-out camera annotations against predictions; never invent measurements."""
import argparse
import json
import math
from pathlib import Path

from ipcam.vision.desk import NAMES


def iou(a, b):
    def corners(box):
        x, y, w, h = box[1:]
        return x-w/2, y-h/2, x+w/2, y+h/2
    ax, ay, ar, ab = corners(a)
    bx, by, br, bb = corners(b)
    overlap = max(0, min(ar, br)-max(ax, bx))*max(0, min(ab, bb)-max(ay, by))
    return overlap / (a[3]*a[4]+b[3]*b[4]-overlap) if overlap else 0.0


def evaluate(data, confidence=.3, match_iou=.5):
    if data.get('names') != list(NAMES) or not data.get('frames'):
        raise ValueError('Fixed classes and nonempty camera frames required')
    if not (0 <= confidence <= 1 and 0 < match_iou <= 1):
        raise ValueError('Invalid evaluation thresholds')
    tp, fp, fn = [0]*10, [0]*10, [0]*10
    confusion = [[0]*11 for _ in range(11)]
    negative_frames = false_alarm_frames = 0
    sessions, ids = set(), set()
    for frame in data['frames']:
        if (frame.get('reviewed') is not True or frame.get('held_out') is not True
                or not frame.get('session') or not frame.get('image')):
            raise ValueError('Reviewed held-out camera frames with image/session identity required')
        if frame['image'] in ids:
            raise ValueError('Duplicate evaluation frame')
        ids.add(frame['image'])
        sessions.add(frame['session'])
        truth = frame['truth']
        predictions = frame['predictions']
        for box in truth + [p['box'] for p in predictions]:
            if len(box) != 5 or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in box):
                raise ValueError('Invalid evaluation box')
            c, x, y, w, h = box
            if c != int(c) or not 0 <= c < 10 or not (w > 0 and h > 0
                    and x-w/2 >= -1e-6 and y-h/2 >= -1e-6 and x+w/2 <= 1+1e-6 and y+h/2 <= 1+1e-6):
                raise ValueError('Invalid evaluation class or bounds')
        if any(not math.isfinite(p['confidence']) or not 0 <= p['confidence'] <= 1 for p in predictions):
            raise ValueError('Invalid prediction confidence')
        predictions = sorted((p for p in predictions if p['confidence'] >= confidence),
                             key=lambda p: p['confidence'], reverse=True)
        if not truth:
            negative_frames += 1
            false_alarm_frames += bool(predictions)
        # Standard class-aware matching for precision/recall; duplicates remain false positives.
        used = set()
        for prediction in predictions:
            box = prediction['box']
            c = int(box[0])
            candidates = [(iou(box, target), i) for i, target in enumerate(truth)
                          if i not in used and target[0] == c]
            score, index = max(candidates, default=(0, -1))
            if score >= match_iou:
                used.add(index)
                tp[c] += 1
            else:
                fp[c] += 1
        for i, target in enumerate(truth):
            if i not in used:
                fn[int(target[0])] += 1
        # Independent spatial matching exposes wrong-class predictions in the confusion matrix.
        used = set()
        for prediction in predictions:
            box = prediction['box']
            candidates = [(iou(box, target), i) for i, target in enumerate(truth) if i not in used]
            score, index = max(candidates, default=(0, -1))
            actual = int(truth[index][0]) if score >= match_iou else 10
            if score >= match_iou:
                used.add(index)
            confusion[actual][int(box[0])] += 1
        for i, target in enumerate(truth):
            if i not in used:
                confusion[int(target[0])][10] += 1
    per_class = {name: dict(tp=tp[i], fp=fp[i], fn=fn[i], support=tp[i]+fn[i],
                           precision=tp[i]/(tp[i]+fp[i]) if tp[i]+fp[i] else None,
                           recall=tp[i]/(tp[i]+fn[i]) if tp[i]+fn[i] else None) for i, name in enumerate(NAMES)}
    return dict(frames=len(ids), sessions=len(sessions), confidence=confidence, iou_threshold=match_iou,
                per_class=per_class, confusion_matrix=confusion, confusion_labels=[*NAMES, 'background'],
                negative_frames=negative_frames, false_alarm_frames=false_alarm_frames,
                negative_frame_false_alarm_rate=false_alarm_frames/negative_frames if negative_frames else None,
                acceptance='unverified; compare measurements with agreed camera requirements',
                note='Frame false-alarm rate is not an event/hour measurement; AP is not computed here')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--confidence', type=float, default=.3)
    args = parser.parse_args()
    report = evaluate(json.loads(args.input.read_text(encoding='utf-8')), args.confidence)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
