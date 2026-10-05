"""Fail-closed comparison on identical reviewed camera frames; does not deploy models."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from tools.evaluate_camera import evaluate
from tools.label_editor import atomic_json


def fingerprint(data):
    records = []
    for frame in data['frames']:
        digest = frame.get('image_sha256', '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Evaluation requires SHA256 of each actual image')
        latency = frame.get('latency_ms')
        if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0:
            raise ValueError('Measured positive finite per-frame latency required')
        records.append(dict(image=frame['image'], image_sha256=digest, session=frame['session'],
                            truth=sorted(frame['truth']), reviewed=frame['reviewed'], held_out=frame['held_out']))
    return hashlib.sha256(json.dumps(sorted(records, key=lambda r: r['image']),
                                    sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def compare(baseline, candidate, min_support=20, min_negative_frames=20, latency_ratio=1.05):
    if min_support < 1 or min_negative_frames < 1 or not math.isfinite(latency_ratio) or latency_ratio < 1:
        raise ValueError('Invalid comparison policy')
    old, new = evaluate(baseline, confidence=0), evaluate(candidate, confidence=0)
    scope = baseline.get('measurement', {}).get('latency_scope')
    if not scope or scope != candidate.get('measurement', {}).get('latency_scope'):
        raise ValueError('Identical declared latency measurement scope required')
    identity = fingerprint(baseline)
    if identity != fingerprint(candidate):
        raise ValueError('Baseline and candidate must use identical images, sessions and ground truth')
    def p95(data):
        values = sorted(frame['latency_ms'] for frame in data['frames'])
        return values[math.ceil(.95*len(values))-1]
    reasons, improvements = [], []
    for name, previous in old['per_class'].items():
        current = new['per_class'][name]
        if previous['support'] < min_support:
            reasons.append(f'{name}: insufficient held-out support')
        # Count-based comparison avoids declaring an undefined precision successful.
        for metric in ('tp',):
            if current[metric] < previous[metric]:
                reasons.append(f'{name}: true positives decreased')
            elif current[metric] > previous[metric]:
                improvements.append(f'{name}: true positives increased')
        if current['fp'] > previous['fp']:
            reasons.append(f'{name}: false positives increased')
        elif current['fp'] < previous['fp']:
            improvements.append(f'{name}: false positives decreased')
    if old['negative_frames'] < min_negative_frames:
        reasons.append('Insufficient held-out negative frames')
    if new['false_alarm_frames'] > old['false_alarm_frames']:
        reasons.append('Negative-frame false alarms increased')
    before, after = p95(baseline), p95(candidate)
    if after > before*latency_ratio:
        reasons.append('p95 latency exceeds allowed ratio')
    if not improvements:
        reasons.append('No detection improvement demonstrated')
    return dict(eligible_for_review=not reasons, deployed=False, reasons=reasons, improvements=improvements,
                evaluation_fingerprint=identity, baseline=old, candidate=new,
                latency_ms=dict(baseline_p95=before, candidate_p95=after, scope=scope),
                policy=dict(min_support=min_support, min_negative_frames=min_negative_frames,
                            max_latency_ratio=latency_ratio),
                limitation='Observed non-regression on these frames is not a universal accuracy guarantee')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = compare(json.loads(args.baseline.read_text(encoding='utf-8')),
                     json.loads(args.candidate.read_text(encoding='utf-8')))
    atomic_json(args.output, report)
    print(json.dumps(dict(eligible_for_review=report['eligible_for_review'], reasons=report['reasons'])))
    raise SystemExit(0 if report['eligible_for_review'] else 2)
