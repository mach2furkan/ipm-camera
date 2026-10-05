"""Train a ten-class .pt candidate and produce a complete, honest handoff report.

The unreviewed-candidate option is separate from the reviewed deployment pipeline.
It never approves source labels, exports engines, or activates the trained model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import zipfile
from pathlib import Path

import yaml
from ipcam.vision.desk import NAMES
from run_pipeline import validate_dataset, validate_model_classes
from tools.label_editor import atomic_json, dataset_lock

ROOT = Path(__file__).resolve().parent.parent


def sha(path):
    with Path(path).open('rb') as handle:
        digest = hashlib.sha256()
        while chunk := handle.read(1024*1024):
            digest.update(chunk)
        return digest.hexdigest()


def review_summary(dataset):
    manifest = dataset / 'review.json'
    review = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else {}
    entries = review.get('images', {})
    findings_path = dataset / 'review_findings.json'
    findings = json.loads(findings_path.read_text(encoding='utf-8')) if findings_path.exists() else {}
    return dict(total=len(entries), pending=sum(v.get('status') != 'approved' for v in entries.values()),
                open_findings=sum(v.get('status') == 'open' for v in findings.values()),
                semantic_approval=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'cctv_desk_data.yaml')
    parser.add_argument('--config', type=Path, default=ROOT / 'cctv_desk_hyperparams.yaml')
    parser.add_argument('--allow-unreviewed-candidate', action='store_true')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()
    if args.batch < 1 or (args.epochs is not None and args.epochs < 1):
        parser.error('Epochs and batch must be positive')
    folder = ROOT / 'cctv_desk_project/yolo11m_cctv_run'
    if folder.exists():
        parser.error('Output directory already exists; existing training artifacts will not be overwritten')
    source = yaml.safe_load(args.data.read_text(encoding='utf-8'))
    dataset = Path(source['path'])
    if not dataset.is_absolute():
        dataset = args.data.resolve().parent / dataset
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    weights = ROOT / config.pop('model')
    if not weights.is_file():
        raise FileNotFoundError(f'Local pretrained weights required: {weights}')
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    unsupported = set(config)-set(DEFAULT_CFG_DICT)
    if unsupported:
        raise ValueError(f'Unsupported training options: {sorted(unsupported)}')
    state = ROOT / 'cctv_desk_project/yolo11m_cctv_training_status.json'
    state.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(state, dict(stage='validating_dataset', training_started=False, pid=os.getpid()))
    # Hold the same lock used by the label editor, so source labels stay stable.
    with dataset_lock(dataset):
        counts = validate_dataset(args.data.resolve(), require_review=not args.allow_unreviewed_candidate,
                                  require_negatives=not args.allow_unreviewed_candidate)
        folder.mkdir(parents=True)
        used_data = dict(source, path=str(dataset.resolve()))
        used_yaml = folder / 'cctv_desk_data.yaml'
        used_yaml.write_text(yaml.safe_dump(used_data, allow_unicode=True, sort_keys=False), encoding='utf-8')
        report_path = folder / 'deployment_report.json'
        report = dict(status='training', training_complete=False, deployed=False,
                      model_family='YOLO11m', names=list(NAMES), dataset=counts,
                      dataset_review=review_summary(dataset), candidate_only=args.allow_unreviewed_candidate,
                      camera_acceptance='unverified; optical and thermal held-out tests required',
                      original_data_yaml=str(args.data.resolve()), data_yaml=str(used_yaml.resolve()),
                      data_yaml_sha256=sha(used_yaml), base_weights=str(weights.resolve()), base_sha256=sha(weights),
                      ultralytics_version=ultralytics.__version__, torch_version=torch.__version__,
                      python_version=platform.python_version(), gpu=torch.cuda.get_device_name(0),
                      usage_permission=dict(unrestricted_permission_granted=False,
                          upstream_license='Ultralytics AGPL-3.0 or Enterprise',
                          enterprise_license_evidence='not provided',
                          proprietary_distribution='not cleared',
                          source='https://www.ultralytics.com/license'),
                      caveats=['Source labels have not been exhaustively reviewed for all ten target classes.',
                               'Public-data validation is not optical/thermal camera acceptance.',
                               'No training hard negatives are present.' if not counts['train']['negatives'] else ''])
        permission = folder / 'MODEL_USAGE.md'
        permission.write_text(
            '# Model usage and distribution\n\n'
            'No unrestricted proprietary usage or distribution permission is granted by this file.\n'
            'The model is fine-tuned from Ultralytics YOLO11m and remains subject to upstream licensing.\n'
            'Ultralytics states AGPL-3.0 or Enterprise licensing applies to fine-tuned models.\n'
            'For a proprietary application, verify an applicable Enterprise license or obtain legal confirmation '
            'of a compliant deployment before distribution. No Enterprise license evidence was supplied.\n\n'
            'Official terms: https://www.ultralytics.com/license\n'
            'Dataset annotations originate from Open Images. Source-image attribution/provenance obligations '
            'must also be checked; training does not itself establish distribution permission.\n\n'
            'This is an unreviewed public-data candidate. Optical/thermal camera acceptance is pending.\n',
            encoding='utf-8')
        atomic_json(report_path, report)
        if args.epochs is not None:
            config['epochs'] = args.epochs
        config['batch'] = args.batch
        config['workers'] = 2
        config['device'] = 0
        # The AMP checker uses the already available local weights/yolo26n.pt.
        config['amp'] = True
        started = time.monotonic()
        model = YOLO(str(weights), task='detect')
        def progress(trainer):
            data = dict(stage='training', training_started=True, pid=os.getpid(),
                        epoch=trainer.epoch+1, epochs=trainer.epochs,
                        metrics={k: float(v) for k, v in (trainer.metrics or {}).items()},
                        save_dir=str(trainer.save_dir), elapsed_s=time.monotonic()-started)
            atomic_json(state, data)
            report.update(epoch=data['epoch'], elapsed_s=data['elapsed_s'], latest_metrics=data['metrics'])
            atomic_json(report_path, report)
        model.add_callback('on_fit_epoch_end', progress)
        def training_started(trainer):
            atomic_json(state, dict(stage='training', training_started=True, pid=os.getpid(),
                                   epoch=0, epochs=trainer.epochs, save_dir=str(trainer.save_dir)))
        model.add_callback('on_train_start', training_started)
        sleep_guard = None
        if os.name == 'nt':
            import ctypes
            sleep_guard = ctypes.windll.kernel32
            sleep_guard.SetThreadExecutionState(0x80000001)
        try:
            atomic_json(state, dict(stage='initializing_training', training_started=False, pid=os.getpid()))
            model.train(data=str(used_yaml), project=str(folder.parent), name=folder.name,
                        exist_ok=True, **config)
            best = Path(model.trainer.best)
            trained = YOLO(str(best), task='detect')
            validate_model_classes(trained.names)
            with zipfile.ZipFile(best) as archive:
                if archive.testzip() is not None:
                    raise ValueError('Trained checkpoint failed archive CRC validation')
            report.update(status='validating_trained_candidate', best_weights=str(best.resolve()),
                          sha256=sha(best), size_bytes=best.stat().st_size, class_order_verified=True,
                          archive_crc_verified=True)
            atomic_json(report_path, report)
            atomic_json(state, dict(stage='validating_trained_candidate', best=str(best.resolve()), pid=os.getpid()))
            metrics = trained.val(data=str(used_yaml), imgsz=config['imgsz'], device=0,
                                  batch=args.batch, workers=2, plots=True)
            per_class = {}
            for i, cls in enumerate(metrics.box.ap_class_index):
                precision, recall, ap50, ap = metrics.box.class_result(i)
                per_class[NAMES[int(cls)]] = dict(class_id=int(cls), precision=float(precision),
                    recall=float(recall), map50=float(ap50), map50_95=float(ap))
            report.update(status='candidate_trained; camera_acceptance_pending', training_complete=True,
                          train_hours=(time.monotonic()-started)/3600,
                          map50=float(metrics.box.map50), map50_95=float(metrics.box.map),
                          per_class_metrics=per_class, training_args=str(folder/'args.yaml'))
            atomic_json(report_path, report)
            atomic_json(state, dict(stage='completed', training_complete=True, deployed=False,
                                   best=str(best.resolve()), report=str(report_path.resolve()),
                                   data_yaml=str(used_yaml.resolve()), pid=os.getpid()))
            print(json.dumps(dict(best=str(best.resolve()), report=str(report_path.resolve())), indent=2), flush=True)
        except BaseException as exc:
            report.update(status='failed_or_interrupted', training_complete=False,
                          error=f'{type(exc).__name__}: {exc}')
            atomic_json(report_path, report)
            atomic_json(state, dict(stage='failed_or_interrupted', training_complete=False,
                                   error=report['error'], pid=os.getpid()))
            raise
        finally:
            if sleep_guard is not None:
                sleep_guard.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()
