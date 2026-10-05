"""After candidate training, compare official validation AP against the unchanged base model."""
import ctypes
import hashlib
import json
import time
from pathlib import Path

from tools.label_editor import atomic_json, dataset_lock
from tools.trusted_coco import ROOT, DEST


def main():
    folder = ROOT / 'cctv_desk_project/world_candidates'
    state_path = folder / 'status.json'
    output = folder / 'source_comparison.json'
    guard = folder / 'validation_lock'
    guard.mkdir(parents=True, exist_ok=True)
    power = ctypes.windll.kernel32
    if not power.SetThreadExecutionState(0x80000001):
        raise OSError('Cannot prevent idle sleep during validation')
    try:
        with dataset_lock(guard):
            deadline = time.monotonic()+6*3600
            while time.monotonic() < deadline:
                state = json.loads(state_path.read_text(encoding='utf-8'))
                if state['stage'] == 'failed':
                    atomic_json(output, dict(stage='training_failed', error=state.get('error'), deployed=False))
                    return 2
                if state['stage'] == 'candidate_trained':
                    break
                time.sleep(30)
            else:
                atomic_json(output, dict(stage='timeout', deployed=False))
                return 2
            baseline = ROOT / 'yolov8m-worldv2.pt'
            if hashlib.sha256(baseline.read_bytes()).hexdigest() != state['base_sha256']:
                raise ValueError('Original checkpoint changed')
            provenance = json.loads((DEST / 'provenance.json').read_text(encoding='utf-8'))
            for record in provenance['records']:
                if record['split'] != 'val':
                    continue
                image = DEST / record['image']
                label = DEST / 'labels/val' / image.with_suffix('.txt').name
                if (hashlib.sha256(image.read_bytes()).hexdigest() != record['image_sha256']
                        or hashlib.sha256(label.read_bytes()).hexdigest() != record['label_sha256']):
                    raise ValueError('Validation image or annotation changed')
            from ultralytics import YOLOWorld
            report = dict(stage='validating', candidate=state['candidate'], deployed=False,
                          camera_acceptance='unverified; COCO has no pen class',
                          provenance_sha256=hashlib.sha256((DEST / 'provenance.json').read_bytes()).hexdigest())
            atomic_json(output, report)
            for role, weights in (('baseline', baseline), ('candidate', Path(state['best']))):
                model = YOLOWorld(str(weights))
                metrics = model.val(data=str(DEST / 'data.yaml'), imgsz=640, batch=8, device=0,
                                    workers=4, plots=False, project=str(folder),
                                    name=f'{state["candidate"]}_{role}_validation', exist_ok=False)
                classes = {}
                for index, class_id in enumerate(metrics.box.ap_class_index):
                    precision, recall, ap50, ap = metrics.box.class_result(index)
                    classes[metrics.names[int(class_id)]] = dict(precision=float(precision), recall=float(recall),
                                                                map50=float(ap50), map50_95=float(ap))
                report[role] = dict(map50=float(metrics.box.map50), map50_95=float(metrics.box.map), per_class=classes)
                atomic_json(output, report)
                del model
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()
            regressions = [name for name, previous in report['baseline']['per_class'].items()
                           if name not in report['candidate']['per_class']
                           or report['candidate']['per_class'][name]['map50_95'] < previous['map50_95']]
            report.update(stage='comparison_finished', source_class_regressions=regressions,
                          aggregate_improved=report['candidate']['map50_95'] > report['baseline']['map50_95'],
                          source_non_regression=not regressions, deployed=False,
                          limitation='Source AP comparison does not establish camera quality or latency')
            atomic_json(output, report)
            print(json.dumps({key:report[key] for key in ('stage','aggregate_improved','source_non_regression','deployed')}),
                  flush=True)
            return 0
    except Exception as exc:
        atomic_json(output, dict(stage='failed', error=str(exc), deployed=False))
        raise
    finally:
        power.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    raise SystemExit(main())
