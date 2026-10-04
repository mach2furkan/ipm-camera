"""Bring-up probe for a thermal PT camera: read-only, never moves the head.

    python -m tools.pt_thermal_probe --ip 192.168.1.64 --user admin --password '***'
    python -m tools.pt_thermal_probe --ip 192.168.1.64 --https --pin 3f:9a:...   (TLS certificate pinning)

Reports what the integration layer will rely on, so model/firmware differences surface on
day one instead of during an incident:

* device profile: namespace, optical/thermal channel ids, PTZ channel, capability flags,
  absoluteEx ranges (elevation, zoom, focal length);
* current head pose incl. inclinometer pitch, device clock offset against this host;
* real-time thermometry rules (if configured);
* raw thermal stream: connects over RTSP, reports the inferred payload layout, frame rate
  and temperature range, and writes ``thermal_probe.png`` (and the raw matrix as .npy).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ipcam.errors import IPCamError  # noqa: E402
from ipcam.isapi.client import HikvisionISAPIClient  # noqa: E402
from ipcam.isapi.pt_thermal import PTThermalCamera  # noqa: E402
from ipcam.thermal import PlateauAGC, ThermalStreamReader, colorize, encode_png  # noqa: E402


async def probe(args: argparse.Namespace) -> int:
    password = args.password or os.environ.get("HIK_PASS", "")
    client = HikvisionISAPIClient(args.ip, args.user, password, port=args.http_port, https=args.https,
                                  tls_fingerprint_sha256=args.pin)
    cam = PTThermalCamera(client)
    report: dict[str, object] = {}
    try:
        prof = await cam.discover()
        report["profile"] = prof.summary()
        print(json.dumps(report["profile"], indent=2, ensure_ascii=False))
        try:
            pose = await cam.get_pose()
            report["pose"] = dataclasses.asdict(pose)
            print(f"pose: az {pose.azimuth:.3f}  el {pose.elevation:.3f}  zoom {pose.zoom:.2f}  "
                  f"focal {pose.focal_len_mm}  pitch(sensor) {pose.pitch_sensor_deg}")
        except IPCamError as exc:
            print(f"pose: unavailable ({exc})")
        try:
            off = await cam.measure_clock_offset()
            report["clock_offset_s"] = off.offset_s
            print(f"device clock offset: {off.offset_s:+.3f} s (rtt {off.rtt_s * 1000:.0f} ms, +/- {off.uncertainty_s:.2f} s)")
        except IPCamError as exc:
            print(f"clock: unavailable ({exc})")
        if prof.supports_realtime_thermometry:
            try:
                for r in await cam.realtime_rules():
                    print(f"rule {r.rule_id} {r.name!r} [{r.kind}] max {r.max_c} min {r.min_c} avg {r.avg_c}")
            except IPCamError as exc:
                print(f"real-time thermometry: {exc}")

        if prof.thermal_channel is not None and not args.no_stream:
            path = cam.thermal_stream_path(prof.thermal_channel, args.stream_type)
            url = f"rtsp://{args.ip}:{args.rtsp_port}{path}"
            reader = ThermalStreamReader(url, username=args.user, password=password, with_metadata=True)
            reader.start()
            t0 = time.monotonic()
            while time.monotonic() - t0 < args.seconds:
                await asyncio.sleep(0.25)
            await reader.stop()
            f = reader.latest()
            if f is None:
                print(f"thermal stream: no frames ({reader.last_error}); decode errors {reader.decode_errors}")
            else:
                lay = reader.decoder.layout
                fps = reader.frames / max(time.monotonic() - t0, 1e-6)
                print(f"thermal stream: {reader.frames} frames ({fps:.1f} fps), layout {lay}, "
                      f"range {np.nanmin(f.temps):.1f} .. {np.nanmax(f.temps):.1f} degC, "
                      f"assembler {reader.assembler_stats}")
                out = Path(args.out)
                out.mkdir(parents=True, exist_ok=True)
                (out / "thermal_probe.png").write_bytes(encode_png(colorize(PlateauAGC(smoothing=1.0)(f.temps), "iron")))
                np.save(out / "thermal_probe.npy", f.temps)
                (out / "thermal_header.bin").write_bytes(f.header)
                print(f"saved {out / 'thermal_probe.png'} and the raw matrix")
        (Path(args.out) / "probe_report.json").write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False),
                                                           encoding="utf-8")
        return 0
    except IPCamError as exc:
        print(f"probe failed: {exc}")
        return 1
    finally:
        await client.aclose()


def main() -> int:
    p = argparse.ArgumentParser(description="Read-only bring-up probe for thermal PT cameras")
    p.add_argument("--ip", required=True)
    p.add_argument("--user", default="admin")
    p.add_argument("--password", help="or env HIK_PASS")
    p.add_argument("--http-port", type=int, default=80)
    p.add_argument("--rtsp-port", type=int, default=554)
    p.add_argument("--https", action="store_true")
    p.add_argument("--pin", help="SHA-256 fingerprint of the device certificate (hex, ':' allowed)")
    p.add_argument("--stream-type", default="pixel-to-pixel_thermometry_data",
                   choices=["pixel-to-pixel_thermometry_data", "thermal_raw_data", "real-time_raw_data"])
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--out", default="probe_out")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return asyncio.run(probe(a))


if __name__ == "__main__":
    sys.exit(main())
