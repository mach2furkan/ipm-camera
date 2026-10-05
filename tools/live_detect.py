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
from collections import OrderedDict, deque

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ipcam import CameraConfig, DecoderOptions, HWAccel, RTSPStreamReader, StreamRole, StreamWatchdog  # noqa: E402
from ipcam.analytics import ByteTrackConfig, ByteTracker  # noqa: E402
from ipcam.metrics import RateMeter, RollingWindow  # noqa: E402
from ipcam.vision.desk import NAMES, prepare_detections  # noqa: E402
from tools.live_runtime import ResponsiveInference, InferenceCancelled, box_arrays  # noqa: E402

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
        self._cache = OrderedDict()

    def patch(self, text: str, bg: tuple[int, int, int]) -> np.ndarray:
        key = (text, bg)
        p = self._cache.get(key)
        if p is not None:
            self._cache.move_to_end(key)
        if p is None:
            from PIL import Image, ImageDraw

            x0, y0, x1, y1 = self.font.getbbox(text)
            pad = max(4, (y1 - y0) // 4)
            img = Image.new("RGB", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), bg[::-1])
            lum = 0.299 * bg[2] + 0.587 * bg[1] + 0.114 * bg[0]
            ImageDraw.Draw(img).text((pad - x0, pad - y0), text, font=self.font,
                                     fill=(20, 20, 20) if lum > 140 else (245, 245, 245))
            p = np.asarray(img)[:, :, ::-1].copy()
            self._cache[key] = p
            if len(self._cache) > 512:
                self._cache.popitem(last=False)
        return p

    def put(self, frame: np.ndarray, text: str, x: int, y: int, bg: tuple[int, int, int]) -> None:
        p = self.patch(text, bg)
        h, w = p.shape[:2]
        H, W = frame.shape[:2]
        x = max(0, min(W - w, x))
        y = max(0, min(H - h, y))
        h, w = min(h, H-y), min(w, W-x)
        frame[y:y + h, x:x + w] = p[:h, :w]


def main(argv=None, connection_defaults=None) -> int:
    ap = argparse.ArgumentParser(description="Live object detection, full screen")
    ap.add_argument("--ip", default=os.environ.get("HIK_IP", "192.168.1.64"))
    ap.add_argument("--user", default=os.environ.get("HIK_USER", "admin"))
    ap.add_argument("--password", default=None, help="or env HIK_PASS")
    ap.add_argument("--stream", choices=["main", "sub"], default="main")
    ap.add_argument("--setup", action="store_true", help="Open IP, username and password form")
    ap.add_argument("--rtsp-port", type=int, default=554)
    ap.add_argument("--channel", type=int, default=1)
    ap.add_argument("--rtsp-path", default="", help="Custom /path for non-Hikvision cameras")
    ap.add_argument("--traffic", action="store_true", help="Independent vehicle detection and line counting")
    ap.add_argument("--brand", choices=["auto", "dahua", "hikvision", "onvif", "custom"], default="auto")
    ap.add_argument("--onvif-port", type=int, default=80)
    ap.add_argument("--software-decode", action="store_true", help="Compatible CPU video decoding; inference stays on GPU")
    ap.add_argument("--model", default="yolov8m-worldv2.pt")
    ap.add_argument("--closed-model", action="store_true", help="fixed ten-class trained .pt or .engine")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf-scale", type=float, default=1.0, help="multiplies every per-class threshold")
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--max-age-ms", type=float, default=250, help="Reject frames older than this before inference")
    args = ap.parse_args(argv)
    if args.max_age_ms <= 0 or not np.isfinite(args.max_age_ms):
        ap.error('--max-age-ms must be finite and positive')
    if not .3 <= args.conf_scale <= 3:
        ap.error('--conf-scale must be between 0.3 and 3')
    cam = None
    password = args.password or os.environ.get("HIK_PASS")
    brands = dict(auto="Otomatik", dahua="Dahua", hikvision="Hikvision", onvif="ONVIF", custom="Özel RTSP")
    if args.setup:
        from tools.camera_setup import ask_camera
        defaults = dict(host=args.ip, username=args.user, stream=args.stream,
                        port=args.rtsp_port, channel=args.channel, path=args.rtsp_path,
                        traffic=args.traffic, brand=brands[args.brand], onvif_port=args.onvif_port,
                        compatible=args.software_decode)
        if connection_defaults:
            defaults.update(connection_defaults)
        selection = ask_camera(**defaults)
        if selection is None:
            return 64
        cam = selection.camera
        args.stream, args.traffic = selection.stream, selection.traffic
        args.software_decode = selection.compatible
        if connection_defaults is not None:
            connection_defaults.update(host=cam.host, username=cam.username, stream=args.stream,
                port=cam.rtsp_port, channel=cam.channel, brand=selection.brand,
                path=selection.custom_path, traffic=args.traffic, onvif_port=selection.onvif_port,
                compatible=args.software_decode)
    elif not password:
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

    if cam is None:
        from tools.camera_setup import selection_from_fields
        try:
            cam = selection_from_fields(args.ip, args.user, password, str(args.rtsp_port),
                str(args.channel), args.rtsp_path, args.stream, args.traffic,
                brand=brands[args.brand], onvif_port=str(args.onvif_port)).camera
            if args.brand != "hikvision":
                from ipcam.stream.connect import resolve_camera
                cam = resolve_camera(cam, args.stream, brands[args.brand], args.onvif_port)
        except ValueError as exc:
            ap.error(str(exc))

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
    traffic_model = traffic_tracker = traffic_overlay = None
    if args.traffic:
        from ipcam.vision.traffic import VEHICLE_PROMPTS, TrafficOverlay
        # Never mutate the desk detector vocabulary: it retains its original 11 prompts / 10 classes.
        traffic_model = YOLOWorld("yolov8m-worldv2.pt")
        traffic_model.set_classes(list(VEHICLE_PROMPTS))
        traffic_tracker = ByteTracker(ByteTrackConfig(high_thresh=.3, low_thresh=.15,
            new_track_thresh=.3, min_hits=3, confirm_first_frame=False, class_aware=True,
            lost_ttl_s=2.0, fuse_score=False, mahalanobis_gate=None))
        traffic_overlay = TrafficOverlay()
    half = device == 0

    role = StreamRole.MAIN if args.stream == "main" else StreamRole.SUB
    prof = cam.profile(role)
    cam = cam.with_profile(role, replace(prof, decoder=DecoderOptions(
        hwaccel=HWAccel.NONE if args.software_decode else HWAccel.AUTO), stall_timeout_s=3.0))
    reader = RTSPStreamReader(cam, role, decode=True)
    watchdog = StreamWatchdog(reader)

    minimum_floor = (.30 if args.closed_model else min(c[2] for c in CLASSES))*.3
    tracker = ByteTracker(ByteTrackConfig(high_thresh=minimum_floor, low_thresh=minimum_floor/2,
                                          new_track_thresh=minimum_floor, fuse_score=False,
                                          min_hits=2, confirm_first_frame=False, lost_ttl_s=1.0,
                                          class_aware=True, mahalanobis_gate=None))
    win = "Canli tespit"   # ASCII: HighGUI window titles are not Unicode-safe on Windows
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if traffic_overlay is not None:
        cv2.setMouseCallback(win, traffic_overlay.click)
    fullscreen = not args.windowed
    if fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(win, 1600, 900)

    painter: LabelPainter | None = None
    painter_height = None
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
    waiting_painter = LabelPainter(16)
    from tools.ptz_panel import PTZPanel
    ptz_panel = PTZPanel(cam)
    ptz_was_moving = False
    ptz_motion_token = None
    inference = ResponsiveInference()
    pending_keys = deque(maxlen=32)
    last_hud_time = -float("inf")
    hud = ""
    last_preview = None
    last_preview_at = 0.
    reconfigure = False
    def pump_controls():
        nonlocal reconfigure
        ptz_panel.pump()
        key = cv2.waitKey(1) & 0xFF
        if key == ord("p"):
            ptz_panel.show()
            return True
        if key == ord("c"):
            reconfigure = True
            return False
        if key in (27, ord("q")) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            return False
        if key != 255:
            pending_keys.append(key)
        return True
    def reset_tracking():
        tracker.reset()
        if traffic_tracker is not None:
            traffic_tracker.reset()
            if traffic_overlay.counter is not None:
                traffic_overlay.counter.reset_tracks()
    print(f"bağlanıyor: {cam.redacted_rtsp_url(role)}  model: {args.model} ({'GPU' if half else 'CPU'})")
    try:
        watchdog.start()
        while True:
            generation_at_read = reader.generation
            res = reader.wait_frame(seq, timeout=.03, max_age_ms=args.max_age_ms)
            if res is None:
                # A normal inter-frame gap is not a disconnection. Keep the matched
                # annotated frame briefly instead of flashing the waiting screen.
                if (last_preview is not None and reader.generation == generation
                        and time.perf_counter()-last_preview_at < .5):
                    cv2.imshow(win, last_preview)
                    if not pump_controls():
                        break
                    continue
                blank = np.zeros((540, 960, 3), np.uint8)
                cv2.putText(blank, "goruntu bekleniyor...", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
                status = watchdog.stats()
                waiting_painter.put(blank, f"Kamera: {cam.host}:{cam.rtsp_port} | {status.state.value}",
                                    30, 315, (24, 24, 24))
                if status.last_failure is not None:
                    waiting_painter.put(blank, f"Bağlantı sorunu: {status.last_failure.value}. IP, şifre ve RTSP yolunu kontrol et.",
                                        30, 355, (24, 24, 24))
                waiting_painter.put(blank, "C: bağlantı | P: PTZ / zoom | Q/Esc: çıkış", 30, 395, (24, 24, 24))
                cv2.imshow(win, blank)
                if not pump_controls():
                    break
                continue
            seq = res.seq
            if reader.generation != generation_at_read:
                reset_tracking()
                generation = reader.generation
                continue
            if reader.generation != generation:
                reset_tracking()
                generation = reader.generation
            frame = res.image
            if not isinstance(frame, np.ndarray):
                frame = frame.permute(1, 2, 0).cpu().numpy()[:, :, ::-1].copy()
            else:
                frame = frame.copy()
            H, W = frame.shape[:2]
            if painter is None or painter_height != H:
                painter = LabelPainter(max(14, H // 48))
                hud_painter = LabelPainter(max(13, H // 60))
                painter_height = H

            t0 = time.perf_counter()
            def predict_frame():
                pred = model.predict(frame, imgsz=args.imgsz, conf=float(floors.min() * scale), device=device,
                                     **precision, verbose=False, rect=False)[0]
                vehicle_pred = None
                if traffic_model is not None:
                    vehicle_pred = traffic_model.predict(frame, imgsz=args.imgsz, conf=.30, device=device,
                        **precision, verbose=False, rect=False)[0]
                return pred, vehicle_pred
            pred, traffic_pred = inference.run(predict_frame, pump_controls)
            if reader.generation != generation:
                reset_tracking()
                generation = reader.generation
                continue
            infer_ms.add((time.perf_counter() - t0) * 1000)
            b = pred.boxes
            if len(b):
                xyxy, conf, cls = box_arrays(b)
                dets = prepare_detections(xyxy, conf, cls, W, H, floors*scale, class_map)
                # User disabled the canonical pen class (both pen and pencil aliases).
                # Keep the detector vocabulary intact so other class scores stay unchanged.
                dets = dets[dets[:, 5] != NAMES.index("kalem")]
            else:
                dets = np.zeros((0, 6))
            out = tracker.update(dets, res.frame.arrival_ns / 1e9)
            traffic_out = None
            if traffic_pred is not None:
                from ipcam.vision.traffic import VEHICLE_LABELS, vehicle_detections
                vb = traffic_pred.boxes
                vd = vehicle_detections(*box_arrays(vb), W, H)
                traffic_out = traffic_tracker.update(vd, res.frame.arrival_ns / 1e9)
                moving = ptz_panel.moving
                token = ptz_panel.motion_token
                motion_changed = token != ptz_motion_token
                ptz_motion_token = token
                if moving or ptz_was_moving or motion_changed:
                    if traffic_overlay.counter is not None:
                        traffic_overlay.counter.reset_tracks()
                ptz_was_moving = moving
                # Camera motion must not be counted as a vehicle crossing.
                traffic_overlay.update(traffic_out, res.frame.arrival_ns / 1e9, H, W,
                                       count=not moving and not motion_changed)

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
            # latency() is sampled now: its queue age already includes inference.
            lat_ms.add(lat.total_ms)
            summary = "   ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "nesne yok"
            now = time.perf_counter()
            if now - last_hud_time >= .25:
                hud = (f"{W}x{H}  {fps.rate():4.1f} fps   model {infer_ms.summary().p50:4.0f} ms   "
                       f"gecikme {lat_ms.summary().p50:4.0f} ms   eşik x{scale:.2f}   |   {summary}")
                last_hud_time = now
            hud_painter.put(frame, hud, 0, 0, (24, 24, 24))
            if traffic_out is not None:
                for tr in traffic_out.tracks:
                    x1, y1, x2, y2 = (int(v) for v in tr.box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 255), thick)
                    painter.put(frame, f"{VEHICLE_LABELS[tr.cls]} T#{tr.track_id} {tr.conf:.2f}",
                                x1, y1, (32, 80, 80))
                traffic_overlay.draw(frame, hud_painter)
            hud_painter.put(frame, "P: PTZ / zoom | C: bağlantı | F: tam ekran | S: fotoğraf | Q/Esc: çıkış",
                            0, H-32, (24, 24, 24))
            cv2.imshow(win, frame)
            ptz_panel.pump()
            last_preview = frame
            last_preview_at = time.perf_counter()
            if fullscreen and not applied_fs:   # must be set after the first imshow to take effect
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                applied_fs = True
            current_key = cv2.waitKey(1) & 0xFF
            if current_key in (27, ord("q")):
                break
            if current_key != 255:
                pending_keys.append(current_key)
            key = pending_keys.popleft() if pending_keys else 255
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
            elif key == ord("l") and traffic_overlay is not None:
                traffic_overlay.points = []
            elif key == ord("r") and traffic_overlay is not None:
                if traffic_overlay.counter is not None:
                    traffic_overlay.counter.reset_counts()
            elif key == ord("c"):
                reconfigure = True
                break
            elif key == ord("p"):
                ptz_panel.show()
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    except (KeyboardInterrupt, InferenceCancelled):
        pass
    finally:
        try:
            inference.close()
        finally:
            try:
                watchdog.stop()
            finally:
                try:
                    ptz_panel.close()
                finally:
                    cv2.destroyAllWindows()
    return 75 if reconfigure else 0


if __name__ == "__main__":
    sys.exit(main())
