"""Phase 7 live demo without cameras: simulated site -> edge batches -> bus -> fusion -> PTZ -> C2 map.

    python -m tools.fusion_demo                 # then open http://127.0.0.1:8080
    python -m tools.fusion_demo --speed 3 --people 20 --port 9000

Everything downstream of the simulator is production code: protobuf wire format,
in-process bus (swap for NATS JetStream with --nats), FusionService, GlobalTrackManager,
SlewToCueEngagement and the C2 server. A restricted zone (x 36-52 m, y 21-32 m) raises
intrusion alerts from whichever camera sees the person, which cues the PTZ.

The thermal panel is fed by a simulated thermal PT channel over real RTSP (Digest auth,
interleaved RTP, PT 109 fragments) through ThermalStreamReader; a fire starts at t=40 s
and a transformer ROI heats up so fire, rate-of-rise and temperature alarms reach the log.
PTZ commands from the console (stop, home, Shift+click slew, track selected) drive the
simulated head.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ipcam.fusion import GlobalTrackManager  # noqa: E402
from ipcam.fusion.c2 import C2Server  # noqa: E402
from ipcam.fusion.service import FusionService, PTZUnit, SiteConfig  # noqa: E402
from ipcam.fusion.sim import SimPTZ, SimWorld, default_site  # noqa: E402
from ipcam.fusion.c2.thermal_panel import ThermalPanel  # noqa: E402
from ipcam.fusion.ptz import PTZPose, aim, fov_deg  # noqa: E402
from ipcam.fusion.transport import InProcessBus, SecurityAlert  # noqa: E402
from ipcam.thermal import RoiMonitor, ThermalStreamReader  # noqa: E402
from ipcam.thermal.sim import FakeThermalRtspServer, SimPerson, SyntheticThermalScene  # noqa: E402

log = logging.getLogger("fusion-demo")

ZONE = (36.0, 21.0, 52.0, 32.0)   # restricted area (world metres) inside cam-B's field of view


async def main(args: argparse.Namespace) -> None:
    cams, persons, topo, mount = default_site(args.seed, n_people=args.people, duration_s=args.duration)
    world = SimWorld(cams, persons, seed=args.seed)
    if args.nats:
        from ipcam.fusion.transport import NatsJetStreamBus
        bus = await NatsJetStreamBus(args.nats).connect()
    else:
        bus = InProcessBus()

    t_wall0 = time.time()
    sim_now = lambda: (time.time() - t_wall0) * args.speed  # noqa: E731
    epoch0 = t_wall0
    rng = np.random.default_rng(args.seed)
    ptz = SimPTZ(mount, sim_now)

    def ptz_detections() -> np.ndarray:
        t = sim_now() - 0.1                       # one frame of video latency
        dets = [ptz.detections(st[0], rng=rng) for p in persons if (st := p.state(t)) is not None]
        dets = [d for d in dets if len(d)]
        return np.concatenate(dets) if dets else np.zeros((0, 5))

    manager = GlobalTrackManager(topo)
    unit = PTZUnit(mount, ptz, ptz_detections, coverage_m=90.0, clock=sim_now, pose_provider=ptz.advance)
    svc = FusionService(bus, manager, site=SiteConfig("Demo tesisi", (0.0, 0.0, 130.0, 40.0), world.calibrations),
                        ptz_units=[unit], clock=lambda: epoch0 + sim_now())
    await svc.start()

    # --- simulated thermal PT channel over real RTSP --------------------------------------
    scene = SyntheticThermalScene(people=[SimPerson(70.0, -12.0, 1.3), SimPerson(140.0, 20.0, -0.9)],
                                  vehicle_at=(180.0, 4.0), fire_at=(120.0, -10.0), fire_start_s=40.0 / args.speed)
    th_srv = FakeThermalRtspServer(scene, fps=8.0)
    await th_srv.start()
    th_reader = ThermalStreamReader(th_srv.url(), username="admin", password="sim-pass", with_metadata=True)
    th_reader.start()
    roi = RoiMonitor("trafo-1", [[0.05, 0.55], [0.25, 0.55], [0.25, 0.75], [0.05, 0.75]], prealarm_c=45, alarm_c=60,
                     dwell_s=2.0, rise_c_per_min=4.0, rise_window_s=30.0)
    base_render = scene.render

    def render_with_hot_trafo(t: float) -> np.ndarray:      # a transformer slowly overheating
        img = base_render(t)
        h, w = img.shape
        img[int(h * .6):int(h * .7), int(w * .08):int(w * .2)] = 25.0 + 0.6 * t * args.speed
        return img

    scene.render = render_with_hot_trafo  # type: ignore[method-assign]
    panel = ThermalPanel(th_reader.latest, label="PT-termal", rois=[roi],
                         on_alarm=lambda kind, text: svc.push_event(kind, text, camera_id="pt-termal"),
                         status=lambda: {"connected": th_reader.connected, "fps": round(th_reader.rate.rate(), 1),
                                         "reconnects": th_reader.reconnects})
    th_reader.on_frame = panel.ingest          # analytics run on every frame, viewer or not
    last_cmd = {"range": None}

    async def ptz_control(cmd: dict) -> dict:
        action = cmd.get("action")
        if action in ("stop", "slew", "home") and unit.engagement is not None:
            await unit.engagement.stop()
            unit.target_gid = None
        if action == "stop":
            await ptz.stop()
            return {"message": "durduruldu"}
        if action == "home":
            await ptz.absolute(PTZPose(0.0, 10.0, 1.0))
            return {"message": "ev konumuna dönüyor"}
        if action == "focus":
            return {"message": "tek dokunuş odak gönderildi"}
        if action == "slew":
            x, y = float(cmd["x"]), float(cmd["y"])
            if not (np.isfinite(x) and np.isfinite(y)):
                raise ValueError("invalid target")
            pose = aim(mount, (x, y))
            last_cmd["range"] = pose.range_m
            await ptz.absolute(pose)
            return {"message": f"yöneliyor: az {pose.azimuth_deg:.1f}°, {pose.range_m:.0f} m"}
        if action == "track":
            gid = int(cmd["gid"])
            assigned = svc._assign_ptz(manager.resolve(gid))
            if assigned is None:
                raise ValueError("hedef PTZ kapsamı dışında veya PTZ meşgul")
            return {"message": f"G-{gid} izleniyor"}
        raise ValueError(f"bilinmeyen komut {action!r}")

    def extra_state() -> dict:
        p = ptz.advance()
        return {"ptz_pose": {"azimuth": p.azimuth_deg, "elevation": p.elevation_deg, "zoom": p.zoom,
                             "focal_mm": mount.focal_min_mm * p.zoom, "focus_m": last_cmd["range"],
                             "hfov": fov_deg(mount, p.zoom)[0], "online": True}}

    server = C2Server(svc, host=args.host, port=args.port, thermal=panel, ptz_control=ptz_control,
                      extra_state=extra_state)
    await server.start()
    print(f"C2 operator station: http://{args.host}:{server.port}  (Ctrl+C to stop)")

    stop = asyncio.Event()
    try:
        asyncio.get_running_loop().add_signal_handler(signal.SIGINT, stop.set)
    except (NotImplementedError, RuntimeError):
        pass  # Windows: KeyboardInterrupt ends asyncio.run instead

    seq = 0
    alerted: set[int] = set()
    period = 1.0 / args.fps
    next_t = 0.0
    try:
        while not stop.is_set() and sim_now() < args.duration + 60:
            t = sim_now()
            if t < next_t:
                await asyncio.sleep((next_t - t) / args.speed)
                continue
            next_t += period
            for cid, (dets, ended) in world.step(t).items():
                seq += 1
                batch = world.frame_batch(cid, t, dets, ended, seq, t_offset=epoch0)
                await bus.publish(f"events.camera.{cid}", batch.encode())
                for d in dets:
                    st = persons[d.pid].state(t)
                    if st is None or d.pid in alerted:
                        continue
                    x, y = st[0]
                    if ZONE[0] <= x <= ZONE[2] and ZONE[1] <= y <= ZONE[3]:
                        alerted.add(d.pid)
                        alert = SecurityAlert(cid, int((t + epoch0) * 1e9), d.local_id, "intrusion", "kuzey-depo")
                        await bus.publish(f"alerts.camera.{cid}", alert.encode())
    finally:
        await server.stop()
        await th_reader.stop()
        await th_srv.stop()
        await svc.stop()
        await bus.close()
        snap = svc.snapshot()
        eng = svc.ptz[0].engagement
        print(f"entities={snap['stats']['entities']} batches={snap['stats']['batches']} "
              f"e2e_p50={snap['stats']['e2e_p50_ms']} ms "
              f"ptz_locks={eng.stats.locks if eng else 0} "
              f"cue_to_lock_ms={[round(v) for v in eng.stats.cue_to_lock_ms] if eng else []}")


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7 fusion / C2 demo on a simulated site")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--speed", type=float, default=2.0, help="simulation speed factor")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--people", type=int, default=14)
    p.add_argument("--duration", type=float, default=180.0)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--nats", help="NATS server URL; default uses the in-process bus")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(main(parse()))
    except KeyboardInterrupt:
        pass
