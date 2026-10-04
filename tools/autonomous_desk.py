"""Logged unattended preparation/training; keeps Windows awake only while running."""
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATUS = ROOT / 'autonomous_status.json'


def status(stage, **extra):
    record = dict(stage=stage, timestamp=time.strftime('%Y-%m-%d %H:%M:%S'), **extra)
    from tools.label_editor import atomic_json
    atomic_json(STATUS, record)
    print(json.dumps(record), flush=True)


def main():
    # Thread-scoped power request: restored on exit; no permanent power-plan changes.
    kernel = ctypes.windll.kernel32
    kernel.SetThreadExecutionState(0x80000001)
    try:
        stages = [('data_preparation', ['-m', 'tools.prepare_desk_data']),
                  ('model_assisted_label_audit', ['-m', 'tools.model_audit']),
                  ('data_validation', ['run_pipeline.py', '--check']),
                  ('training_export_benchmark', ['run_pipeline.py'])]
        for name, arguments in stages:
            status(name)
            with (ROOT / f'training-{name}.log').open('a', encoding='utf-8') as log:
                result = subprocess.run([sys.executable, '-u', *arguments], cwd=ROOT,
                                        stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                if name == 'data_validation':
                    status('awaiting_label_review', step=name, exit_code=result.returncode,
                           log=f'training-{name}.log',
                           review='dataset_cctv_desk/label-editor.html',
                           quality_report='dataset_cctv_desk/quality_report.json', training_started=False)
                    return result.returncode
                status('failed', step=name, exit_code=result.returncode,
                       log=f'training-{name}.log')
                return result.returncode
        status('finished', camera_acceptance='unverified')
        return 0
    except Exception as exc:
        status('failed', error=str(exc))
        return 1
    finally:
        kernel.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    raise SystemExit(main())
