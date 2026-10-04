"""Find Hikvision / ONVIF cameras on the local network.

    python -m tools.discover                 # all IPv4 interfaces, SADP + WS-Discovery + port sweep
    python -m tools.discover --no-sweep

Three independent methods, because each one fails in some network:

* **SADP** (UDP multicast 239.255.255.250:37020) -- Hikvision's own discovery; answers even
  when the device is *not activated* or sits in a different IP subnet than the PC, and
  reports model, serial, MAC, firmware, IP/mask/gateway and activation state.
* **WS-Discovery** (UDP 239.255.255.250:3702) -- ONVIF; vendor independent.
* **TCP sweep** of the interface's /24 on 80, 554, 8000 -- works when multicast is blocked;
  HTTP responses are checked for camera fingerprints (Digest realm, ISAPI paths).

Sends only discovery probes and TCP connects; no login attempts (wrong logins lock cameras).
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import re
import socket
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

SADP_GROUP, SADP_PORT = "239.255.255.250", 37020
WSD_GROUP, WSD_PORT = "239.255.255.250", 3702


@dataclass
class Found:
    ip: str
    methods: set[str] = field(default_factory=set)
    info: dict[str, str] = field(default_factory=dict)
    ports: set[int] = field(default_factory=set)


def local_ipv4() -> list[tuple[str, int]]:
    """(ip, prefix) of up interfaces, excluding loopback/link-local."""
    out: list[tuple[str, int]] = []
    try:
        import subprocess

        ps = ("Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and "
              "$_.IPAddress -notlike '169.254*' } | ForEach-Object { \"$($_.IPAddress)/$($_.PrefixLength)\" }")
        res = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=10)
        for line in res.stdout.split():
            ip, _, pre = line.partition("/")
            out.append((ip, int(pre or 24)))
    except Exception:  # noqa: BLE001 - non-Windows fallback
        pass
    if not out:
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                ip = info[4][0]
                if not ip.startswith(("127.", "169.254")):
                    out.append((ip, 24))
        except OSError:
            pass
    return sorted(set(out))


def _xml_local(root: ET.Element) -> dict[str, str]:
    return {el.tag.rsplit("}", 1)[-1]: (el.text or "").strip() for el in root.iter() if (el.text or "").strip()}


def multicast_probe(iface_ip: str, group: str, port: int, payload: bytes, timeout: float) -> list[tuple[str, bytes]]:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((iface_ip, 0 if port == WSD_PORT else port))
    except OSError:
        s.bind((iface_ip, 0))
    s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(iface_ip))
    s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    try:
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(group) + socket.inet_aton(iface_ip))
    except OSError:
        pass
    s.settimeout(0.3)
    replies: list[tuple[str, bytes]] = []
    end = time.monotonic() + timeout
    sent = 0
    while time.monotonic() < end:
        if sent < 3 and time.monotonic() > end - timeout + sent * 0.5:
            try:
                s.sendto(payload, (group, port))
            except OSError:
                pass
            sent += 1
        try:
            data, addr = s.recvfrom(65535)
        except (socket.timeout, OSError):
            continue
        if data != payload:
            replies.append((addr[0], data))
    s.close()
    return replies


def sadp(iface_ip: str, timeout: float) -> dict[str, dict[str, str]]:
    probe = (f'<?xml version="1.0" encoding="utf-8"?><Probe><Uuid>{str(uuid.uuid4()).upper()}</Uuid>'
             "<Types>inquiry</Types></Probe>").encode()
    out: dict[str, dict[str, str]] = {}
    for src, data in multicast_probe(iface_ip, SADP_GROUP, SADP_PORT, probe, timeout):
        try:
            info = _xml_local(ET.fromstring(data))
        except ET.ParseError:
            continue
        if "DeviceType" in info or "DeviceSN" in info or "IPv4Address" in info:
            out[info.get("IPv4Address") or src] = info | {"_reply_from": src}
    return out


def ws_discovery(iface_ip: str, timeout: float) -> dict[str, dict[str, str]]:
    msg = f"""<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
 xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
<e:Header><w:MessageID>uuid:{uuid.uuid4()}</w:MessageID><w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
<w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action></e:Header>
<e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body></e:Envelope>""".encode()
    out: dict[str, dict[str, str]] = {}
    for src, data in multicast_probe(iface_ip, WSD_GROUP, WSD_PORT, msg, timeout):
        text = data.decode("utf-8", "replace")
        xaddr = re.search(r"XAddrs>([^<]+)<", text)
        scopes = re.search(r"Scopes>([^<]+)<", text)
        info = {"XAddrs": xaddr.group(1) if xaddr else ""}
        if scopes:
            for sc in scopes.group(1).split():
                m = re.match(r"onvif://www\.onvif\.org/(name|hardware|location)/(.+)", sc)
                if m:
                    info[m.group(1)] = m.group(2).replace("%20", " ")
        out[src] = info
    return out


async def _connect(ip: str, port: int, timeout: float) -> bool:
    try:
        _, w = await asyncio.wait_for(asyncio.open_connection(ip, port), timeout)
        w.close()
        return True
    except (OSError, asyncio.TimeoutError):
        return False


async def _http_fingerprint(ip: str, port: int = 80) -> str | None:
    try:
        r, w = await asyncio.wait_for(asyncio.open_connection(ip, port), 1.5)
        w.write(f"GET /ISAPI/System/deviceInfo HTTP/1.1\r\nHost: {ip}\r\nConnection: close\r\n\r\n".encode())
        await w.drain()
        data = await asyncio.wait_for(r.read(4096), 2.0)
        w.close()
    except (OSError, asyncio.TimeoutError):
        return None
    text = data.decode("latin-1", "replace")
    realm = re.search(r'realm="([^"]+)"', text)
    server = re.search(r"(?im)^server:\s*(.+)$", text)
    status = text.split("\r\n", 1)[0]
    parts = [status]
    if realm:
        parts.append(f'realm="{realm.group(1)}"')
    if server:
        parts.append(f"server={server.group(1).strip()}")
    return " ".join(parts)


async def sweep(network: ipaddress.IPv4Network, ports: tuple[int, ...], exclude: set[str]) -> dict[str, set[int]]:
    sem = asyncio.Semaphore(256)
    found: dict[str, set[int]] = {}

    async def one(ip: str, port: int) -> None:
        async with sem:
            if await _connect(ip, port, 0.6):
                found.setdefault(ip, set()).add(port)

    await asyncio.gather(*(one(str(h), p) for h in network.hosts() if str(h) not in exclude for p in ports))
    return found


async def main_async(args: argparse.Namespace) -> int:
    ifaces = local_ipv4()
    if not ifaces:
        print("no IPv4 interface found")
        return 1
    print("interfaces:", ", ".join(f"{ip}/{p}" for ip, p in ifaces))
    devices: dict[str, Found] = {}
    own = {ip for ip, _ in ifaces}

    loop = asyncio.get_running_loop()
    jobs = []
    for ip, _ in ifaces:
        jobs.append(loop.run_in_executor(None, sadp, ip, args.timeout))
        jobs.append(loop.run_in_executor(None, ws_discovery, ip, args.timeout))
    results = await asyncio.gather(*jobs, return_exceptions=True)
    for i, res in enumerate(results):
        if isinstance(res, BaseException):
            continue
        method = "SADP" if i % 2 == 0 else "ONVIF"
        for ip, info in res.items():
            d = devices.setdefault(ip, Found(ip))
            d.methods.add(method)
            d.info.update(info)

    if not args.no_sweep:
        nets = {ipaddress.ip_interface(f"{ip}/{max(p, 24)}").network for ip, p in ifaces}
        for net in nets:
            print(f"sweeping {net} on ports 80, 554, 8000 ...")
            for ip, ports in (await sweep(net, (80, 554, 8000), own)).items():
                d = devices.setdefault(ip, Found(ip))
                d.ports |= ports
                d.methods.add("TCP")
    # Fingerprint HTTP on everything that answered.
    for d in devices.values():
        if 80 in d.ports or "SADP" in d.methods or "ONVIF" in d.methods:
            fp = await _http_fingerprint(d.ip)
            if fp:
                d.info["http"] = fp

    def is_camera(d: Found) -> bool:
        http = d.info.get("http", "")
        return ("SADP" in d.methods or "ONVIF" in d.methods or 554 in d.ports or 8000 in d.ports
                or "IP Camera" in http or "DVRDVS" in http or "/ISAPI" in http)

    cams = [d for d in devices.values() if is_camera(d)]
    others = [d for d in devices.values() if not is_camera(d)]
    print()
    if not cams:
        print("Kamera bulunamadı.")
    for d in sorted(cams, key=lambda x: tuple(int(p) for p in x.ip.split("."))):
        i = d.info
        print(f"KAMERA  {d.ip}   [{'+'.join(sorted(d.methods))}]  portlar: {sorted(d.ports) or '-'}")
        for key, label in (("DeviceDescription", "model"), ("DeviceType", "tip"), ("DeviceSN", "seri no"),
                           ("MAC", "MAC"), ("SoftwareVersion", "firmware"), ("IPv4SubnetMask", "maske"),
                           ("IPv4Gateway", "ağ geçidi"), ("Activated", "aktive"), ("DHCP", "DHCP"),
                           ("HttpPort", "HTTP port"), ("name", "ONVIF ad"), ("hardware", "ONVIF donanım"),
                           ("XAddrs", "ONVIF adres"), ("http", "HTTP")):
            if i.get(key):
                print(f"        {label:<13} {i[key]}")
        mask = i.get("IPv4SubnetMask")
        if mask:
            cam_net = ipaddress.ip_interface(f"{d.ip}/{mask}").network
            if not any(ipaddress.ip_address(ip) in cam_net for ip, _ in ifaces):
                print(f"        UYARI: kamera {cam_net} ağında; bu PC o ağda değil -> PC'ye o ağdan bir IP ekleyin "
                      f"veya kameranın IP'sini SADP ile değiştirin.")
        if i.get("Activated", "").lower() == "false":
            print("        UYARI: cihaz aktive edilmemiş -> önce şifre belirleyin (web arayüzü veya SADP).")
    if others and args.verbose:
        print("\ndiğer cihazlar:")
        for d in others:
            print(f"        {d.ip}  portlar {sorted(d.ports)}  {d.info.get('http', '')}")
    return 0 if cams else 2


def main() -> int:
    p = argparse.ArgumentParser(description="Discover Hikvision/ONVIF cameras on the LAN")
    p.add_argument("--timeout", type=float, default=3.0)
    p.add_argument("--no-sweep", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
