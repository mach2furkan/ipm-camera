"""Fine-tune the current YOLO-World checkpoint in isolation; never deploy automatically."""
import argparse
import ctypes
import hashlib
import json
import time
from pathlib import Path

from tools.label_editor import atomic_json, dataset_lock
from tools.trusted_coco import ROOT, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=640)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32:
        parser.error('Positive epochs/batch and valid image size required')
    checkpoint = ROOT / 'yolov8m-worldv2.pt'
    baseline_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    folder = ROOT / 'cctv_desk_project/world_candidates'
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / 'status.json'
    power = ctypes.windll.kernel32
    if not power.SetThreadExecutionState(0x80000001):
        raise OSError('Cannot prevent system sleep during training')
    try:
        with dataset_lock(folder):
            atomic_json(state_path, dict(stage='preparing_official_dataset', training_started=False,
                                         base_weights=str(checkpoint), base_sha256=baseline_hash,
                                         automatic_deployment=False, sleep_prevented=True))
            data = prepare()
            import torch
            from ultralytics import YOLOWorld
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required')
            model = YOLOWorld(str(checkpoint))
            name = 'world_from_current_' + time.strftime('%Y%m%d_%H%M%S')

            def progress(trainer):
                atomic_json(state_path, dict(stage='training', training_started=True, candidate=name,
                                             epoch=trainer.epoch+1, epochs=trainer.epochs,
                                             metrics={k: float(v) for k,v in (trainer.metrics or {}).items()},
                                             base_sha256=baseline_hash, automatic_deployment=False,
                                             save_dir=str(trainer.save_dir), sleep_prevented=True))

            def started(trainer):
                atomic_json(state_path, dict(stage='training', training_started=True, candidate=name,
                                             epoch=0, epochs=trainer.epochs, base_sha256=baseline_hash,
                                             automatic_deployment=False, save_dir=str(trainer.save_dir),
                                             sleep_prevented=True))

            model.add_callback('on_train_start', started)
            model.add_callback('on_fit_epoch_end', progress)
            atomic_json(state_path, dict(stage='initializing_training', training_started=False, candidate=name,
                                         base_sha256=baseline_hash, automatic_deployment=False))
            model.train(data=str(data), project=str(folder), name=name, exist_ok=False,
                        epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, device=0, workers=4,
                        freeze=10, optimizer='AdamW', lr0=.0001, lrf=.1, warmup_epochs=2,
                        patience=7, amp=True, seed=42, deterministic=True,
                        mosaic=.3, close_mosaic=5, mixup=0, degrees=0, shear=0,
                        save=True, save_period=1, plots=True)
            best = Path(model.trainer.best)
            if not best.is_file():
                raise RuntimeError('No best candidate produced')
            atomic_json(state_path, dict(stage='candidate_trained', candidate=name, best=str(best),
                                         base_sha256=baseline_hash, automatic_deployment=False,
                                         camera_acceptance='unverified; original model remains active'))
            print(str(best), flush=True)
    except Exception as exc:
        atomic_json(state_path, dict(stage='failed', error=str(exc), automatic_deployment=False))
        raise
    finally:
        power.SetThreadExecutionState(0x80000000)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != baseline_hash:
            raise RuntimeError('Original model checksum changed')


if __name__ == '__main__':
    main()
