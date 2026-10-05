"""Vendor-independent connection preflight: common RTSP paths and ONVIF media discovery."""
from __future__ import annotations

import base64
import hashlib
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import replace
from datetime import datetime, timezone
from urllib.parse import urlsplit
from xml.sax.saxutils import escape

from ipcam.config import CameraConfig, StreamRole
from ipcam.errors import FailureKind, classify_failure

BRANDS = ("Otomatik", "Dahua", "Hikvision", "ONVIF", "Özel RTSP")


class ConnectionCheckError(ValueError):
    pass


def common_paths(brand, channel, stream):
    sub = stream == "sub"
    dahua = f"/cam/realmonitor?channel={channel}&subtype={int(sub)}"
    hik = f"/Streaming/Channels/{channel}{'02' if sub else '01'}"
    if brand == "Dahua":
        return [dahua]
    if brand == "Hikvision":
        return [hik]
    if brand == "Otomatik":
        return [dahua, hik]
    return []


def _same_device(url, camera, schemes):
    parts = urlsplit(url)
    if parts.scheme not in schemes or parts.hostname != camera.host or parts.username or parts.password:
        raise ConnectionCheckError("Kamera farklı adres veya kimlik bilgisi içeren bir yayın bildirdi. Özel RTSP yolunu girin.")
    return parts


def onvif_candidates(camera, stream, port=80):
    """Media1 GetCapabilities / GetProfiles / GetStreamUri, HTTP Digest + WSSE digest."""
    import httpx
    host = f"[{camera.host}]" if ":" in camera.host else camera.host
    device_url = f"http://{host}:{port}/onvif/device_service"
    with httpx.Client(auth=httpx.DigestAuth(camera.username, camera.password), timeout=4,
                      trust_env=False, follow_redirects=False) as client:
        def request(endpoint, action, body):
            _same_device(endpoint, camera, ("http", "https"))
            nonce = os.urandom(16)
            created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            digest = base64.b64encode(hashlib.sha1(nonce+created.encode()+camera.password.encode()).digest()).decode()
            envelope = ('<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" '
                'xmlns:tds="http://www.onvif.org/ver10/device/wsdl" '
                'xmlns:trt="http://www.onvif.org/ver10/media/wsdl" '
                'xmlns:tt="http://www.onvif.org/ver10/schema" '
                'xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd" '
                'xmlns:wsu="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">'
                '<s:Header><wsse:Security><wsse:UsernameToken>'
                f'<wsse:Username>{escape(camera.username)}</wsse:Username>'
                '<wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">'
                f'{digest}</wsse:Password><wsse:Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">'
                f'{base64.b64encode(nonce).decode()}</wsse:Nonce><wsu:Created>{created}</wsu:Created>'
                f'</wsse:UsernameToken></wsse:Security></s:Header><s:Body>{body}</s:Body></s:Envelope>')
            response = client.post(endpoint, content=envelope.encode(), headers={
                "Content-Type": f'application/soap+xml; charset=utf-8; action="{action}"'})
            if response.status_code in (401, 403):
                raise ConnectionCheckError("ONVIF kullanıcı adı/şifre reddedildi. ONVIF hesabını ve yetkisini kontrol edin.")
            response.raise_for_status()
            if len(response.content) > 2_000_000:
                raise ConnectionCheckError("ONVIF yanıtı beklenenden büyük.")
            root = ET.fromstring(response.content)
            if root.find(".//{*}Fault") is not None:
                raise ConnectionCheckError("ONVIF isteği reddedildi; ONVIF etkinliğini, hesabını ve kamera saatini kontrol edin.")
            return root
        capabilities = request(device_url, "http://www.onvif.org/ver10/device/wsdl/GetCapabilities",
                               "<tds:GetCapabilities><tds:Category>Media</tds:Category></tds:GetCapabilities>")
        media = capabilities.find(".//{*}Media/{*}XAddr")
        if media is None or not media.text:
            return []
        endpoint = media.text.strip()
        profiles = request(endpoint, "http://www.onvif.org/ver10/media/wsdl/GetProfiles", "<trt:GetProfiles/>")
        items = profiles.findall(".//{*}Profiles")
        sources = {}
        for profile in items:
            source = profile.find(".//{*}VideoSourceConfiguration/{*}SourceToken")
            key = source.text if source is not None and source.text else "default"
            sources.setdefault(key, []).append(profile)
        if camera.channel > len(sources):
            raise ConnectionCheckError("ONVIF profilinde seçilen kanal bulunamadı; kanal veya özel RTSP yolunu kontrol edin.")
        if sources:
            items = list(sources.values())[camera.channel-1]
        def area(profile):
            width, height = profile.find(".//{*}Resolution/{*}Width"), profile.find(".//{*}Resolution/{*}Height")
            try:
                return int(width.text)*int(height.text)
            except (AttributeError, ValueError, TypeError):
                return 0
        items.sort(key=area, reverse=stream == "main")
        candidates = []
        for profile in items[:4]:
            token = profile.get("token")
            if not token:
                continue
            uri = request(endpoint, "http://www.onvif.org/ver10/media/wsdl/GetStreamUri",
                '<trt:GetStreamUri><trt:StreamSetup><tt:Stream>RTP-Unicast</tt:Stream>'
                '<tt:Transport><tt:Protocol>RTSP</tt:Protocol></tt:Transport></trt:StreamSetup>'
                f'<trt:ProfileToken>{escape(token)}</trt:ProfileToken></trt:GetStreamUri>').find(".//{*}Uri")
            if uri is not None and uri.text:
                parts = _same_device(uri.text.strip(), camera, ("rtsp",))
                path = parts.path + ("?"+parts.query if parts.query else "")
                candidates.append(replace(camera, rtsp_port=parts.port or 554, rtsp_path=path))
        return candidates


def probe_video(camera, role):
    """Require one decoded frame, close native resources, never expose raw exception URLs."""
    import av
    started = time.monotonic()
    with av.open(camera.rtsp_url(role), options={"rtsp_transport": "tcp", "allowed_media_types": "video"},
                 timeout=(4., 4.)) as container:
        for packet in container.demux(video=0):
            if time.monotonic()-started > 10:
                break
            for frame in packet.decode():
                if frame.width > 0 and frame.height > 0:
                    return
    raise ConnectionCheckError("Bağlantı açıldı ancak çözülebilen görüntü gelmedi.")


def resolve_camera(camera, stream, brand="Otomatik", onvif_port=80, probe=probe_video, discover=onvif_candidates):
    role = StreamRole(stream)
    if brand not in BRANDS:
        raise ConnectionCheckError("Geçersiz kamera markası.")
    candidates = ([camera] if camera.rtsp_path else
                  [replace(camera, rtsp_path=p) for p in common_paths(brand, camera.channel, stream)])
    if brand == "Özel RTSP" and not camera.rtsp_path:
        raise ConnectionCheckError("Özel RTSP için kameranın yayın yolunu girin.")
    failures = []
    def try_candidates(items):
        for candidate in items:
            try:
                probe(candidate, role)
                return candidate
            except Exception as exc:
                kind = classify_failure(exc)
                if kind is FailureKind.AUTH:
                    raise ConnectionCheckError("Kamera kullanıcı adı/şifreyi reddetti (401/403). Bilgileri ve izleme yetkisini kontrol edin.") from None
                failures.append(kind)
        return None
    result = try_candidates(candidates)
    if result is not None:
        return result
    onvif_hint = ""
    if not camera.rtsp_path and brand in ("Otomatik", "ONVIF"):
        try:
            discovered = discover(camera, stream, onvif_port)
        except ConnectionCheckError as exc:
            onvif_hint = str(exc)
            discovered = []
        except Exception:
            onvif_hint = "ONVIF yayını bulunamadı; ONVIF etkinliğini ve portunu kontrol edin."
            discovered = []
        result = try_candidates(discovered)
        if result is not None:
            return result
    hint = {FailureKind.NETWORK: "IP/RTSP portuna erişilemiyor.", FailureKind.TIMEOUT: "Kameradan zamanında görüntü gelmedi.",
            FailureKind.NOT_FOUND: "RTSP yolu veya kanal bulunamadı (404).", FailureKind.DECODE: "Video çözülemedi; H.264/H.265 yayın ayarını kontrol edin."}
    detail = hint.get(failures[-1], "Çözülebilen RTSP görüntüsü alınamadı.") if failures else "Yayın bulunamadı."
    raise ConnectionCheckError(f"{detail} {onvif_hint} Marka, kanal ve özel RTSP yolunu kontrol edin.") from None
