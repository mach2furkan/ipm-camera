"""End-to-end smoke test / benchmark for the Hikvision streaming + ISAPI layer.

* ISAPI: device info, sub-stream encoder settings, IR-cut state (polled), optional alerts.
* RTSP: sub stream (x02) through RTSPStreamReader + StreamWatchdog, consumed at a fixed
  rate (default 30 FPS) exactly like an inference loop would.
* Reports per-frame latency (network lag / decode / convert / queue), FPS in/out,
  dropped frames, reconnects, and asserts that no native handle leaked at shutdown.

Examples::

    python -m tools.main_pipeline_test --ip 192.168.1.64 --user admin --password '***' --display
    HIK_IP=192.168.1.64 HIK_PASS=*** python -m tools.main_pipeline_test --duration 120 --max-p95-ms 150

Exit codes: 0 ok, 1 no frames, 2 latency budget exceeded, 3 resource leak, 130 interrupted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ipcam import (
    CameraConfig,
    DecoderOptions,
    HikvisionISAPIClient,
    HWAccel,
    IPCamError,
    RTSPStreamReader,
    StreamRole,
    StreamWatchdog,
    live_handles,
)
from ipcam.bus import EventBus
from ipcam.isapi import AlertEvent, AlertStreamListener
from ipcam.isapi.alert_stream import AlertMessage
from ipcam.metrics import RateMeter, RollingWindow
from ipcam.stream.watchdog import WatchdogEvent
from ipcam.vision import IRStateResolver

log = logging.getLogger("smoke")


# --------------------------------------------------------------------------- async side-car

class AsyncSidecar:
    """Runs the asyncio control plane on a background thread so the main thread can own
    the OpenCV window (HighGUI must stay on one thread, and on macOS on the main one)."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="isapi-loop", daemon=True)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def start(self) -> AsyncSidecar:
        self._thread.start()
        return self

    def submit(self, coro: Any) -> Future[Any]:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def stop(self) -> None:
        async def shutdown() -> None:
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        try:
            self.submit(shutdown()).result(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(5)
        self.loop.close()


@dataclass
class ControlPlaneState:
    device: str = "?"
    encoder: str = "?"
    ir_mode: str = "?"
    ir_error: str | None = None
    alerts: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


async def control_plane(cfg: CameraConfig, state: ControlPlaneState, resolver: IRStateResolver,
                        ir_poll_s: float, alerts: bool) -> None:
    async with HikvisionISAPIClient.from_config(cfg) as cam:
        try:
            info = await cam.get_device_info()
            with state.lock:
                state.device = f"{info.model} fw {info.firmware_version} sn {info.serial_number[-9:]}"
        except IPCamError as exc:
            state.device = f"unavailable ({exc})"
        try:
            ch = await cam.get_streaming_channel(cfg.channel * 100 + 2)
            with state.lock:
                state.encoder = (f"{ch.codec} {ch.width}x{ch.height} @{ch.max_fps:g}fps GOP={ch.gop_frames} "
                                 f"{'SmartCodec ' if ch.smart_codec else ''}{ch.bitrate_kbps or '?'}kbps")
        except (IPCamError, ValueError) as exc:
            state.encoder = f"unavailable ({exc})"
        log.info("device: %s", state.device)
        log.info("sub-stream encoder: %s", state.encoder)

        listener: AlertStreamListener | None = None
        consumer: asyncio.Task[None] | None = None
        if alerts:
            bus: EventBus[AlertMessage] = EventBus()
            bus.bind_loop()
            sub = bus.subscribe(name="smoke")
            listener = AlertStreamListener(cam, bus)
            listener.start()

            async def consume() -> None:
                async for msg in sub:
                    if isinstance(msg, AlertEvent):
                        line = (f"{msg.timestamp:%H:%M:%S} {msg.event_type} {msg.phase.value} "
                                f"ch={msg.channel_id} regions={','.join(msg.region_ids) or '-'}")
                        log.info("ALERT %s", line)
                        with state.lock:
                            state.alerts = (state.alerts + [line])[-5:]

            consumer = asyncio.create_task(consume())

        try:
            while True:
                try:
                    ir = await cam.get_ircut_filter(cfg.channel)
                    resolver.update_isapi(ir)
                    with state.lock:
                        state.ir_mode, state.ir_error = ir.mode.value, None
                except IPCamError as exc:
                    with state.lock:
                        state.ir_error = str(exc)
                await asyncio.sleep(ir_poll_s)
        finally:
            if consumer is not None:
                consumer.cancel()
            if listener is not None:
                await listener.stop()


# --------------------------------------------------------------------------- benchmark loop

@dataclass
class Report:
    duration_s: float
    frames_consumed: int
    consumer_fps: float
    decoder_fps: float
    frames_overwritten: int
    gated_packets: int
    decode_errors: int
    reconnects: int
    stalls: int
    zombies: int
    decoder: str
    latency_total_ms: dict[str, float]
    latency_pipeline_ms: dict[str, float]
    network_lag_ms: dict[str, float]
    decode_ms: dict[str, float]
    convert_ms: dict[str, float]
    queue_ms: dict[str, float]
    ir_state: str
    leaked_handles: dict[str, int]


def _summ(w: RollingWindow) -> dict[str, float]:
    s = w.summary()
    return {"p50": round(s.p50, 2), "p95": round(s.p95, 2), "p99": round(s.p99, 2),
            "max": round(s.maximum, 2), "n": s.count}


def parse_size(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    w, _, h = text.lower().partition("x")
    return int(w), int(h)


def run(args: argparse.Namespace) -> int:
    cfg = CameraConfig(
        host=args.ip or os.environ.get("HIK_IP", ""),
        username=args.user or os.environ.get("HIK_USER", "admin"),
        password=args.password or os.environ.get("HIK_PASS", ""),
        channel=args.channel,
        rtsp_port=args.rtsp_port,
        http_port=args.http_port,
        https=args.https,
    )
    if not cfg.host or not cfg.password:
        log.error("camera address and password are required (--ip/--password or HIK_IP/HIK_PASS)")
        return 64
    decoder = DecoderOptions(hwaccel=HWAccel(args.hwaccel), output_size=parse_size(args.size),
                             decoder_name=args.decoder)
    cfg = cfg.with_profile(StreamRole.SUB, replace(cfg.sub, decoder=decoder, stall_timeout_s=args.stall_s))

    resolver = IRStateResolver()
    cp = ControlPlaneState()
    sidecar = AsyncSidecar().start()
    cp_future = sidecar.submit(control_plane(cfg, cp, resolver, args.ir_poll_s, args.alerts))

    def on_wd(ev: WatchdogEvent) -> None:
        extra = f" retry in {ev.next_retry_s:.1f}s" if ev.next_retry_s else ""
        log.info("link %s: %s%s", ev.state.value, ev.reason, extra)

    reader = RTSPStreamReader(cfg, StreamRole.SUB)

    async def request_keyframe() -> None:
        # Long GOPs (50 frames at 20 fps = 2.5 s) would otherwise delay the first decodable
        # frame after every (re)connect; an on-demand IDR makes start-up near instant.
        async with HikvisionISAPIClient.from_config(cfg) as cam:
            await cam.request_keyframe(cfg.channel * 100 + 2)

    def on_connected(_info: Any) -> None:
        fut = sidecar.submit(request_keyframe())
        fut.add_done_callback(lambda f: f.cancelled() or f.exception() is None
                              or log.debug("requestKeyFrame failed: %s", f.exception()))

    reader.on_connected(on_connected)
    watchdog = StreamWatchdog(reader, on_event=on_wd)
    log.info("connecting %s", cfg.redacted_rtsp_url(StreamRole.SUB))

    cv2: Any = None
    if args.display:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            log.warning("opencv-python not installed; running headless")

    w_total, w_pipe, w_net, w_dec, w_conv, w_queue = (RollingWindow(4096) for _ in range(6))
    consumer_rate = RateMeter(2.0)
    consumed = 0
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    period = 1.0 / args.fps
    t_start = time.monotonic()
    last_consume = 0.0
    last_status = t_start
    last_seq = 0
    interrupted = False

    watchdog.start()
    try:
        while not stop.is_set():
            now = time.monotonic()
            if args.duration and now - t_start >= args.duration:
                break
            # Event driven like a real inference loop: block until a frame newer than the last
            # one exists, process it at once; --fps only caps the rate (excess frames are dropped
            # by the reader's drop-oldest slot, never queued).
            res = reader.wait_frame(last_seq, timeout=0.5)
            if res is not None:
                last_seq = res.seq
                last_consume = time.perf_counter()
                img = res.image  # forces lazy conversion, like a model's pre-processing step
                lat = res.frame.latency()
                w_total.add(lat.total_ms)
                w_pipe.add(lat.pipeline_ms)
                w_net.add(lat.network_lag_ms)
                w_dec.add(lat.decode_ms)
                w_conv.add(lat.convert_ms)
                w_queue.add(lat.queue_ms)
                consumer_rate.tick()
                consumed += 1
                ir = resolver.observe_frame(img)

                if cv2 is not None and getattr(img, "ndim", 0) == 3:
                    view = img.copy()
                    lines = [
                        (f"latency {lat.total_ms:6.1f} ms (net {lat.network_lag_ms:5.1f} | dec {lat.decode_ms:5.1f}"
                         f" | cvt {lat.convert_ms:4.1f} | q {lat.queue_ms:4.1f})"),
                        (f"fps out {consumer_rate.rate():4.1f}  in {reader.stats().decode_rate:4.1f}  "
                         f"seq {res.seq} skipped {res.skipped}  gop+{res.frame.frames_since_keyframe}"),
                        f"IR {ir.mode.value} ({ir.source}, chroma {ir.chroma:.1f}, isapi {cp.ir_mode})",
                    ]
                    for i, text in enumerate(lines):
                        y = 24 + i * 24
                        cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.imshow("ipcam smoke test", view)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                        break

            if now - last_status >= 1.0:
                last_status = now
                st = reader.stats()
                wd = watchdog.stats()
                t = w_total.summary()
                print(
                    f"\r[{wd.state.value:9s}] out {consumer_rate.rate():5.1f} fps | in {st.decode_rate:5.1f} fps | "
                    f"lat p50 {t.p50:6.1f} p95 {t.p95:6.1f} ms | drop {st.frames_overwritten:5d} | "
                    f"reconn {wd.reconnects} | IR {resolver.state().mode.value:7s} | {st.decoder}   ",
                    end="", flush=True,
                )

            # Pace the consumer at the target rate, like a fixed-rate inference loop.
            spare = period - (time.perf_counter() - last_consume)
            if res is not None and spare > 0:
                time.sleep(spare)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        print()
        elapsed = time.monotonic() - t_start
        st = reader.stats()
        watchdog.stop()
        cp_future.cancel()
        sidecar.stop()
        if cv2 is not None:
            cv2.destroyAllWindows()
    interrupted = interrupted or stop.is_set()

    wd = watchdog.stats()
    report = Report(
        duration_s=round(elapsed, 2),
        frames_consumed=consumed,
        consumer_fps=round(consumed / elapsed, 2) if elapsed else 0.0,
        decoder_fps=round(st.frames_decoded / elapsed, 2) if elapsed else 0.0,
        frames_overwritten=st.frames_overwritten,
        gated_packets=st.gated_packets,
        decode_errors=st.decode_errors,
        reconnects=wd.reconnects,
        stalls=wd.stalls,
        zombies=wd.zombies,
        decoder=st.decoder,
        latency_total_ms=_summ(w_total),
        latency_pipeline_ms=_summ(w_pipe),
        network_lag_ms=_summ(w_net),
        decode_ms=_summ(w_dec),
        convert_ms=_summ(w_conv),
        queue_ms=_summ(w_queue),
        ir_state=f"{resolver.state().mode.value} (isapi={cp.ir_mode})",
        leaked_handles=dict(live_handles()),
    )
    print_report(report, cp)
    if args.json:
        Path(args.json).write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    if interrupted and args.duration and consumed == 0:
        code = 130
    elif consumed == 0:
        code = 1
    elif args.max_p95_ms and report.latency_total_ms["p95"] > args.max_p95_ms:
        code = 2
    elif report.leaked_handles:
        code = 3
    else:
        code = 0
    return code


def print_report(r: Report, cp: ControlPlaneState) -> None:
    def row(name: str, d: dict[str, float]) -> str:
        return f"  {name:<18} p50 {d['p50']:7.2f}  p95 {d['p95']:7.2f}  p99 {d['p99']:7.2f}  max {d['max']:7.2f}"

    print("=" * 78)
    print(f"device          : {cp.device}")
    print(f"encoder (sub)   : {cp.encoder}")
    print(f"decoder         : {r.decoder}")
    print(f"duration        : {r.duration_s:.1f}s   consumed {r.frames_consumed} frames")
    print(f"fps             : consumer {r.consumer_fps:.2f} / decoder {r.decoder_fps:.2f}")
    print(f"drop-oldest     : {r.frames_overwritten} overwritten, {r.gated_packets} gated, "
          f"{r.decode_errors} decode errors")
    print(f"resilience      : {r.reconnects} reconnects, {r.stalls} stalls, {r.zombies} zombie threads")
    print(f"IR state        : {r.ir_state}")
    print("latency (ms)")
    print(row("total", r.latency_total_ms))
    print(row("host pipeline", r.latency_pipeline_ms))
    print(row("network lag", r.network_lag_ms))
    print(row("decode", r.decode_ms))
    print(row("convert", r.convert_ms))
    print(row("queue", r.queue_ms))
    print(f"native handles  : {'none leaked' if not r.leaked_handles else r.leaked_handles}")
    print("=" * 78)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hikvision RTSP/ISAPI pipeline smoke test")
    p.add_argument("--ip", help="camera IP (env HIK_IP)")
    p.add_argument("--user", help="username (env HIK_USER, default admin)")
    p.add_argument("--password", help="password (env HIK_PASS)")
    p.add_argument("--channel", type=int, default=1)
    p.add_argument("--rtsp-port", type=int, default=554)
    p.add_argument("--http-port", type=int, default=80)
    p.add_argument("--https", action="store_true")
    p.add_argument("--fps", type=float, default=30.0, help="consumer rate (default 30)")
    p.add_argument("--duration", type=float, default=30.0, help="seconds; 0 = until Ctrl+C")
    p.add_argument("--hwaccel", default="auto", choices=[h.value for h in HWAccel])
    p.add_argument("--decoder", help="explicit FFmpeg decoder, e.g. h264_cuvid")
    p.add_argument("--size", help="resize output, e.g. 640x360")
    p.add_argument("--stall-s", type=float, default=3.0)
    p.add_argument("--ir-poll-s", type=float, default=2.0)
    p.add_argument("--alerts", action="store_true", help="also listen to the ISAPI alert stream")
    p.add_argument("--display", action="store_true", help="show an OpenCV window with overlay")
    p.add_argument("--max-p95-ms", type=float, default=0.0, help="fail (exit 2) above this p95 latency")
    p.add_argument("--json", help="write the report as JSON")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
