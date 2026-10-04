"""Fixed class order and camera geometry checks for the closed vocabulary model."""
import math

NAMES = ('insan', 'kitap', 'dizustu_bilgisayar', 'monitor', 'klavye',
         'fare', 'telefon', 'bardak', 'sise', 'kalem')


def validate_detection(box, frame_width, frame_height, class_name):
    if frame_width <= 0 or frame_height <= 0 or class_name not in NAMES:
        return False
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        return False
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= frame_width and 0 <= y1 < y2 <= frame_height):
        return False
    w, h = x2 - x1, y2 - y1
    # Keep thin pens: an arbitrary four-pixel width floor loses valid objects.
    if w * h <= 16:
        return False
    ratio = w * h / (frame_width * frame_height)
    limit = 1.0 if class_name == 'insan' else 0.70
    if class_name in ('kalem', 'fare', 'telefon'):
        limit = 0.35
    return ratio <= limit


def prepare_detections(boxes, scores, classes, width, height, floors, class_map=None, duplicate_iou=.65):
    """Validate detector output, map aliases to fixed classes and suppress duplicate aliases."""
    import numpy as np
    boxes, scores, classes = np.asarray(boxes, float), np.asarray(scores, float), np.asarray(classes, float)
    floors = np.asarray(floors, float)
    mapping = np.asarray(class_map if class_map is not None else range(len(NAMES)), float)
    if floors.shape != (len(NAMES),) or not np.isfinite(floors).all() or (floors < 0).any():
        raise ValueError('Exactly ten finite nonnegative confidence floors required')
    if (mapping.ndim != 1 or not np.isfinite(mapping).all() or (mapping != np.floor(mapping)).any()
            or (mapping < 0).any() or (mapping >= len(NAMES)).any()):
        raise ValueError('Invalid source-to-fixed class mapping')
    if boxes.shape == (0,):
        boxes = boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.shape != (len(boxes),) or classes.shape != (len(boxes),):
        raise ValueError('Detector arrays have inconsistent shapes')
    if not 0 < duplicate_iou <= 1:
        raise ValueError('Invalid duplicate IoU threshold')
    valid = (np.isfinite(boxes).all(axis=1) & np.isfinite(scores) & np.isfinite(classes)
             & (scores >= 0) & (scores <= 1) & (classes == np.floor(classes))
             & (classes >= 0) & (classes < len(mapping)))
    boxes, scores, classes = boxes[valid], scores[valid], mapping[classes[valid].astype(int)].astype(int)
    valid = np.array([score >= floors[cls] and validate_detection(box, width, height, NAMES[cls])
                      for box, score, cls in zip(boxes, scores, classes)], dtype=bool)
    boxes, scores, classes = boxes[valid], scores[valid], classes[valid]
    keep = []
    for index in np.argsort(-scores, kind='stable'):
        duplicate = False
        for previous in keep:
            if classes[index] != classes[previous]:
                continue
            a, b = boxes[index], boxes[previous]
            overlap = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
            union = (a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-overlap
            if overlap/union >= duplicate_iou:
                duplicate = True
                break
        if not duplicate:
            keep.append(index)
    if not keep:
        return np.zeros((0, 6), dtype=float)
    return np.column_stack((boxes[keep], scores[keep], classes[keep]))
