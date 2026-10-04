"""Live object detection on the camera, full screen, with stable tracked labels.

    python -m tools.live_detect --ip 192.168.1.64 --user admin          (password: env HIK_PASS)
    python -m tools.live_detect --ip 192.168.1.64 --stream sub --imgsz 640

Keys:  F  full screen on/off    Q / Esc  quit    S  save snapshot    + / -  confidence

Pipeline: RTSP (main stream, TCP, low-latency options) -> GPU decode -> YOLO-World
(open vocabulary, so classes outside COCO such as "pen" work) on CUDA in FP16 ->
ByteTrack (stable ids, no label flicker) -> overlay. Labels are rendered with a TrueType
font because OpenCV's Hershey fonts cannot draw Turkish characters.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ipcam import CameraConfig, DecoderOptions, HWAccel, RTSPStreamReader, StreamRole, StreamWatchdog  # noqa: E402
from ipcam.analytics import ByteTrackConfig, ByteTracker  # noqa: E402
from ipcam.metrics import RateMeter, RollingWindow  # noqa: E402
from ipcam.vision.desk import NAMES, prepare_detections  # noqa: E402

# (prompt for the open-vocabulary model, Turkish label, per-class confidence floor, BGR colour)
CLASSES: list[tuple[str, str, float, tuple[int, int, int]]] = [
    ("person", "insan", 0.30, (58, 163, 224)),
    ("book", "kitap", 0.20, (90, 200, 120)),
    ("laptop", "dizüstü bilgisayar", 0.35, (220, 170, 60)),
    ("computer monitor", "monitör", 0.25, (220, 170, 60)),
    ("keyboard", "klavye", 0.20, (200, 140, 60)),
    ("pen", "kalem", 0.12, (80, 90, 240)),
    ("pencil", "kalem", 0.12, (80, 90, 240)),
    ("computer mouse", "fare", 0.15, (200, 90, 200)),
    ("cell phone", "telefon", 0.20, (160, 200, 240)),
    ("cup", "bardak", 0.20, (120, 200, 200)),
    ("bottle", "şişe", 0.20, (120, 200, 200)),
]


class LabelPainter:
    """Draws UTF-8 text via Pillow onto small patches only (cheap at 2560x1440)."""

    def __init__(self, size: int) -> None:
        from PIL import ImageFont

        self.font = None
        for path in (r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\segoeui.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
            if os.path.exists(path):
                self.font = ImageFont.truetype(path, size)
                break
        if self.font is None:
            self.font = ImageFont.load_default()
        self._cache: dict[tuple[str, tuple[int, int, int]], np.ndarray] = {}

    def patch(self, text: str, bg: tuple[int, int, int]) -> np.ndarray:
        key = (text, bg)
        p = self._cache.get(key)
        if p is None:
            from PIL import Image, ImageDraw

            x0, y0, x1, y1 = self.font.getbbox(text)
            pad = max(4, (y1 - y0) // 4)
            img = Image.new("RGB", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), bg[::-1])
            lum = 0.299 * bg[2] + 0.587 * bg[1] + 0.114 * bg[0]
            ImageDraw.Draw(img).text((pad - x0, pad - y0), text, font=self.font,
                                     fill=(20, 20, 20) if lum > 140 else (245, 245, 245))
            p = np.asarray(img)[:, :, ::-1].copy()
            if len(self._cache) > 512:
                self._cache.clear()
            self._cache[key] = p
        return p

    def put(self, frame: np.ndarray, text: str, x: int, y: int, bg: tuple[int, int, int]) -> None:
        p = self.patch(text, bg)
        h, w = p.shape[:2]
        H, W = frame.shape[:2]
        x = max(0, min(W - w, x))
        y = max(0, min(H - h, y))
        frame[y:y + h, x:x + w] = p


def main() -> int:
    ap = argparse.ArgumentParser(description="Live object detection, full screen")
    ap.add_argument("--ip", default=os.environ.get("HIK_IP", "192.168.1.64"))
    ap.add_argument("--user", default=os.environ.get("HIK_USER", "admin"))
    ap.add_argument("--password", default=None, help="or env HIK_PASS")
    ap.add_argument("--stream", choices=["main", "sub"], default="main")
    ap.add_argument("--model", default="yolov8m-worldv2.pt")
    ap.add_argument("--closed-model", action="store_true", help="fixed ten-class trained .pt or .engine")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf-scale", type=float, default=1.0, help="multiplies every per-class threshold")
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--max-age-ms", type=float, default=250, help="Reject frames older than this before inference")
    args = ap.parse_args()
    if args.max_age_ms <= 0 or not np.isfinite(args.max_age_ms):
        ap.error('--max-age-ms must be finite and positive')
    if not .3 <= args.conf_scale <= 3:
        ap.error('--conf-scale must be between 0.3 and 3')
    password = args.password or os.environ.get("HIK_PASS")
    if not password:
        import tkinter as tk
        from tkinter import simpledialog

        prompt = tk.Tk()
        prompt.withdraw()
        prompt.attributes("-topmost", True)
        try:
            password = simpledialog.askstring("Kamera bağlantısı", f"{args.ip} kamera parolası:",
                                              show="*", parent=prompt)
        finally:
            prompt.destroy()
        if not password:
            return 64

    import torch
    from ultralytics import YOLO, YOLOWorld
    from ultralytics.cfg import DEFAULT_CFG_DICT

    device = 0 if torch.cuda.is_available() else "cpu"
    precision = ({'quantize': 16 if device == 0 else 32} if 'quantize' in DEFAULT_CFG_DICT
                 else {'half': device == 0})
    model = YOLO(args.model, task="detect") if args.closed_model else YOLOWorld(args.model)
    if args.closed_model:
        model.predict(np.zeros((args.imgsz, args.imgsz, 3), np.uint8),
                      imgsz=args.imgsz, device=device, rect=False, verbose=False)
        if tuple(model.names[i] for i in range(10)) != NAMES or len(model.names) != 10:
            raise ValueError("Model class order does not match CCTV desk taxonomy")
    else:
        model.set_classes([c[0] for c in CLASSES])
    half = device == 0

    role = StreamRole.MAIN if args.stream == "main" else StreamRole.SUB
    cam = CameraConfig(args.ip, args.user, password)
    prof = cam.profile(role)
    cam = cam.with_profile(role, replace(prof, decoder=DecoderOptions(hwaccel=HWAccel.AUTO), stall_timeout_s=3.0))
    reader = RTSPStreamReader(cam, role, decode=True)
    watchdog = StreamWatchdog(reader)

    minimum_floor = (.30 if args.closed_model else min(c[2] for c in CLASSES))*.3
    tracker = ByteTracker(ByteTrackConfig(high_thresh=minimum_floor, low_thresh=minimum_floor/2,
                                          new_track_thresh=minimum_floor, fuse_score=False,
                                          min_hits=2, confirm_first_frame=False, lost_ttl_s=1.0,
                                          class_aware=True, mahalanobis_gate=None))
    win = "Canli tespit"   # ASCII: HighGUI window titles are not Unicode-safe on Windows
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    fullscreen = not args.windowed
    if fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(win, 1600, 900)

    painter: LabelPainter | None = None
    applied_fs = False
    hud_painter: LabelPainter | None = None
    fps = RateMeter(2.0)
    infer_ms = RollingWindow(120)
    lat_ms = RollingWindow(120)
    scale = args.conf_scale
    seq = 0
    canonical_sources = (0, 1, 2, 3, 4, 7, 8, 9, 10, 5)
    labels_tr = [CLASSES[i][1] for i in canonical_sources]
    colours = [CLASSES[i][3] for i in canonical_sources]
    floors = np.array([CLASSES[i][2] for i in canonical_sources])
    class_map = (0, 1, 2, 3, 4, 9, 9, 5, 6, 7, 8)
    if args.closed_model:
        labels_tr = list(NAMES)
        floors = np.array([0.30] * 9 + [0.65])
        class_map = tuple(range(len(NAMES)))
    generation = reader.generation
    print(f"bağlanıyor: {cam.redacted_rtsp_url(role)}  model: {args.model} ({'GPU' if half else 'CPU'})")
    try:
        watchdog.start()
        while True:
            generation_at_read = reader.generation
            res = reader.wait_frame(seq, timeout=1.0, max_age_ms=args.max_age_ms)
            if res is None:
                blank = np.zeros((540, 960, 3), np.uint8)
                cv2.putText(blank, "goruntu bekleniyor...", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
                cv2.imshow(win, blank)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
                continue
            seq = res.seq
            if reader.generation != generation_at_read:
                tracker.reset()
                generation = reader.generation
                continue
            if reader.generation != generation:
                tracker.reset()
                generation = reader.generation
            frame = res.image
            if not isinstance(frame, np.ndarray):
                frame = frame.permute(1, 2, 0).cpu().numpy()[:, :, ::-1].copy()
            else:
                frame = frame.copy()
            H, W = frame.shape[:2]
            if painter is None:
                painter = LabelPainter(max(14, H // 48))
                hud_painter = LabelPainter(max(13, H // 60))

            t0 = time.perf_counter()
            pred = model.predict(frame, imgsz=args.imgsz, conf=float(floors.min() * scale), device=device,
                                 **precision, verbose=False, rect=False)[0]
            if reader.generation != generation:
                tracker.reset()
                generation = reader.generation
                continue
            infer_ms.add((time.perf_counter() - t0) * 1000)
            b = pred.boxes
            if len(b):
                xyxy = b.xyxy.cpu().numpy()
                conf = b.conf.cpu().numpy()
                cls = b.cls.cpu().numpy()
                dets = prepare_detections(xyxy, conf, cls, W, H, floors*scale, class_map)
            else:
                dets = np.zeros((0, 6))
            out = tracker.update(dets, res.frame.arrival_ns / 1e9)

            counts: dict[str, int] = {}
            thick = max(2, H // 400)
            for tr in out.tracks:
                x1, y1, x2, y2 = (int(v) for v in tr.box)
                c = colours[tr.cls]
                name = labels_tr[tr.cls]
                counts[name] = counts.get(name, 0) + 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), c, thick, cv2.LINE_AA)
                text = f"{name} #{tr.track_id}  {tr.conf:.2f}"
                ph = painter.patch(text, c).shape[0]
                painter.put(frame, text, x1, y1 - ph if y1 - ph > 0 else y1, c)

            fps.tick()
            lat = res.frame.latency()
            lat_ms.add(lat.total_ms + (time.perf_counter() - t0) * 1000)
            summary = "   ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "nesne yok"
            hud = (f"{W}x{H}  {fps.rate():4.1f} fps   model {infer_ms.summary().p50:4.0f} ms   "
                   f"gecikme {lat_ms.summary().p50:4.0f} ms   eşik x{scale:.2f}   |   {summary}")
            hud_painter.put(frame, hud, 0, 0, (24, 24, 24))
            cv2.imshow(win, frame)
            if fullscreen and not applied_fs:   # must be set after the first imshow to take effect
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                applied_fs = True
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            elif key == ord("s"):
                path = f"tespit_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(path, frame)
                print("kaydedildi:", path)
            elif key in (ord("+"), ord("=")):
                scale = min(3.0, scale * 1.15)
            elif key == ord("-"):
                scale = max(0.3, scale / 1.15)
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        watchdog.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
