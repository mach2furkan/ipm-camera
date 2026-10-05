"""Typed configuration for cameras, RTSP transport and decoder selection."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from enum import Enum
from urllib.parse import quote


class StreamRole(str, Enum):
    """Hikvision channel suffix: <channel>01 = main stream, <channel>02 = sub stream."""

    MAIN = "main"
    SUB = "sub"

    @property
    def suffix(self) -> str:
        return "01" if self is StreamRole.MAIN else "02"


class HWAccel(str, Enum):
    """Decoder backend selection.

    AUTO   probe CUDA -> VA-API -> QSV -> D3D11VA -> VideoToolbox -> software.
    NVDEC  PyAV demux + PyNvVideoCodec decode; frames stay in GPU memory (torch CUDA tensors).
    The rest pin a specific FFmpeg hwaccel device type via PyAV.
    """

    AUTO = "auto"
    NONE = "none"
    CUDA = "cuda"
    NVDEC = "nvdec"
    VAAPI = "vaapi"
    QSV = "qsv"
    D3D11VA = "d3d11va"
    VIDEOTOOLBOX = "videotoolbox"


@dataclass(frozen=True, slots=True)
class LowLatencyOptions:
    """FFmpeg demuxer/codec options that collapse buffering to the minimum.

    Defaults are the mandated values: TCP interleaved transport (no UDP loss ->
    no macroblocking), ``nobuffer+discardcorrupt``, ``low_delay``, max_delay 500 ms,
    reorder_queue_size 0, probesize 32 KiB, analyzeduration 500 ms.
    """

    rtsp_transport: str = "tcp"
    fflags: str = "nobuffer+discardcorrupt"
    flags: str = "low_delay"
    max_delay_us: int = 500_000
    reorder_queue_size: int = 0
    probesize: int = 32_768
    analyzeduration_us: int = 500_000
    socket_timeout_us: int = 3_000_000
    video_only: bool = True
    user_agent: str = "ipcam/0.1"
    extra: tuple[tuple[str, str], ...] = ()

    def to_ffmpeg(self, *, libavformat_major: int | None = None) -> dict[str, str]:
        opts: dict[str, str] = {
            "rtsp_transport": self.rtsp_transport,
            "fflags": self.fflags,
            "flags": self.flags,
            "max_delay": str(self.max_delay_us),
            "reorder_queue_size": str(self.reorder_queue_size),
            "probesize": str(self.probesize),
            "analyzeduration": str(self.analyzeduration_us),
            "user_agent": self.user_agent,
        }
        # FFmpeg >= 5 (libavformat 59) renamed the RTSP socket I/O timeout to "timeout".
        # On FFmpeg 4 "timeout" means *listen* timeout and silently turns the demuxer into
        # an RTSP server, so the legacy "stimeout" key must be used there.
        if libavformat_major is None or libavformat_major >= 59:
            opts["timeout"] = str(self.socket_timeout_us)
        else:
            opts["stimeout"] = str(self.socket_timeout_us)
        if self.video_only:
            # Skip SETUP for the audio track: one fewer round trip and less TCP payload.
            opts["allowed_media_types"] = "video"
        opts.update(dict(self.extra))
        return opts


@dataclass(frozen=True, slots=True)
class DecoderOptions:
    hwaccel: HWAccel = HWAccel.AUTO
    device_index: int = 0
    # Explicit FFmpeg decoder name (h264_cuvid, hevc_qsv, h264_v4l2m2m, h264_nvmpi, ...).
    decoder_name: str | None = None
    # Output pixel layout for CPU frames; GPU frames are always RGB planar CHW uint8.
    output_format: str = "bgr24"
    # Optional (width, height): resize is fused with the colour conversion in swscale/CUDA.
    output_size: tuple[int, int] | None = None
    # Slice threading adds no frame latency; frame threading delays output by N-1 frames.
    thread_type: str = "SLICE"
    thread_count: int = 0
    allow_software_fallback: bool = True


@dataclass(frozen=True, slots=True)
class StreamProfile:
    role: StreamRole
    decoder: DecoderOptions = field(default_factory=DecoderOptions)
    latency: LowLatencyOptions = field(default_factory=LowLatencyOptions)
    open_timeout_s: float = 5.0
    read_timeout_s: float = 3.0
    stall_timeout_s: float = 3.0
    # Grace for the first decodable frame after (re)connect: Hikvision GOPs with H.265+ /
    # Smart Codec can stretch to 4-8 s, which would otherwise trip the 3 s heartbeat.
    keyframe_wait_s: float = 10.0
    # Seconds of compressed packets retained for pre-event evidence (main stream only).
    preroll_s: float = 0.0


@dataclass(frozen=True, slots=True)
class CameraConfig:
    host: str
    username: str
    password: str = field(repr=False)
    channel: int = 1
    rtsp_port: int = 554
    http_port: int = 80
    https: bool = False
    verify_tls: bool = False
    name: str | None = None
    sub: StreamProfile = field(default_factory=lambda: StreamProfile(StreamRole.SUB))
    main: StreamProfile = field(
        default_factory=lambda: StreamProfile(
            StreamRole.MAIN,
            decoder=DecoderOptions(),
            stall_timeout_s=3.0,
        )
    )
    # Main stream is consumed on demand; it is stopped this long after the last lease ends.
    main_idle_linger_s: float = 10.0
    rtsp_path: str | None = None

    @property
    def label(self) -> str:
        return self.name or self.host

    def profile(self, role: StreamRole) -> StreamProfile:
        return self.main if role is StreamRole.MAIN else self.sub

    def rtsp_url(self, role: StreamRole, *, with_credentials: bool = True) -> str:
        """``rtsp://user:pass@ip:554/Streaming/Channels/<ch><01|02>`` with RFC 3986 escaping.

        Hikvision passwords frequently contain ``@``, ``:`` or ``#``; unescaped they corrupt
        the URL authority and FFmpeg fails with an opaque 401 or "Invalid argument".
        """
        path = self.rtsp_path or f"/Streaming/Channels/{self.channel}{role.suffix}"
        host = f"[{self.host}]" if ":" in self.host and not self.host.startswith("[") else self.host
        if with_credentials:
            auth = f"{quote(self.username, safe='')}:{quote(self.password, safe='')}@"
        else:
            auth = ""
        return f"rtsp://{auth}{host}:{self.rtsp_port}{path}"

    def redacted_rtsp_url(self, role: StreamRole) -> str:
        return self.rtsp_url(role, with_credentials=False).replace("rtsp://", f"rtsp://{self.username}:***@")

    def with_profile(self, role: StreamRole, profile: StreamProfile) -> CameraConfig:
        return replace(self, main=profile) if role is StreamRole.MAIN else replace(self, sub=profile)

    @classmethod
    def from_env(cls, prefix: str = "HIK_") -> CameraConfig:
        """Build from ``HIK_IP``, ``HIK_USER``, ``HIK_PASS`` (+ optional ``HIK_CHANNEL`` ...)."""

        def env(key: str, default: str | None = None) -> str:
            value = os.environ.get(prefix + key, default)
            if value is None:
                raise KeyError(f"missing environment variable {prefix + key}")
            return value

        return cls(
            host=env("IP"),
            username=env("USER", "admin"),
            password=env("PASS"),
            channel=int(env("CHANNEL", "1")),
            rtsp_port=int(env("RTSP_PORT", "554")),
            http_port=int(env("HTTP_PORT", "80")),
            https=env("HTTPS", "0") in {"1", "true", "yes"},
        )
