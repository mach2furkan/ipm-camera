"""Wait for explicit dataset approvals, then launch one isolated candidate training."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from ipcam.vision.desk import NAMES
from tools.label_editor import atomic_json

ROOT = Path(__file__).resolve().parent.parent


def readiness(root):
    manifest = json.loads((root / 'review.json').read_text(encoding='utf-8'))
    if manifest.get('names') != list(NAMES):
        raise ValueError('Review taxonomy mismatch')
    entries = manifest.get('images', {})
    findings_path = root / 'review_findings.json'
    findings = json.loads(findings_path.read_text(encoding='utf-8')) if findings_path.exists() else {}
    pending = sum(e.get('status') != 'approved' for e in entries.values())
    opened = sum(e.get('status') == 'open' for e in findings.values())
    return dict(total=len(entries), pending=pending, open_findings=opened,
                ready_for_validation=bool(entries) and pending == 0 and opened == 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'cctv_desk_data.yaml')
    parser.add_argument('--config', type=Path, default=ROOT / 'cctv_desk_hyperparams.yaml')
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.data.read_text(encoding='utf-8'))
    dataset = Path(config['path'])
    if not dataset.is_absolute():
        dataset = args.data.resolve().parent / dataset
    folder = ROOT / 'cctv_desk_project'
    folder.mkdir(exist_ok=True)
    status_path = folder / 'training_queue_status.json'
    # Exclude a second watcher/training process using a cross-process file lock.
    from tools.label_editor import dataset_lock
    lock_root = folder / 'training_queue_lock'
    lock_root.mkdir(exist_ok=True)
    with dataset_lock(lock_root):
        while True:
            try:
                state = readiness(dataset)
                atomic_json(status_path, dict(stage='waiting_for_review', training_started=False, **state))
                print(json.dumps(state), flush=True)
                if state['ready_for_validation']:
                    from tools.dataset_quality import audit_quality
                    report = audit_quality(args.data.resolve())
                    if not report['training_ready']:
                        atomic_json(status_path, dict(stage='validation_blocked', training_started=False,
                                                       blocker=report.get('review_blocker', 'Quality issues'),
                                                       **state))
                        if not args.wait:
                            return 2
                    else:
                        name = 'candidate_' + time.strftime('%Y%m%d_%H%M%S')
                        command = [sys.executable, '-u', str(ROOT / 'run_pipeline.py'), '--train-only',
                                   '--data', str(args.data.resolve()), '--config', str(args.config.resolve()),
                                   '--name', name]
                        atomic_json(status_path, dict(stage='launching_candidate', training_started=False,
                                                       candidate=name, config=str(args.config.resolve())))
                        with (folder / f'{name}.log').open('w', encoding='utf-8') as log:
                            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                        atomic_json(status_path, dict(stage='candidate_finished' if result.returncode == 0 else 'candidate_failed',
                                                       candidate=name, exit_code=result.returncode,
                                                       camera_acceptance='unverified', deployed=False,
                                                       log=str(folder / f'{name}.log')))
                        return result.returncode
                if not args.wait:
                    return 2
            except (OSError, ValueError) as exc:
                atomic_json(status_path, dict(stage='validation_blocked', training_started=False, error=str(exc)))
                print(str(exc), flush=True)
                if not args.wait:
                    return 2
            time.sleep(30)


if __name__ == '__main__':
    raise SystemExit(main())
