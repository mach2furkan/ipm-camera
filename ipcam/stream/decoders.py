"""Decoder backends: FFmpeg hwaccel through PyAV, and NVDEC zero-copy through PyNvVideoCodec.

Both consume compressed ``av.Packet`` objects from the same low-latency PyAV demuxer, so
transport, keyframe gating, packet taps (evidence recording) and the watchdog behave
identically regardless of where pixels are decoded.

=================  ==========================  =====================================
backend            pixels end up in            notes
=================  ==========================  =====================================
PyAV + hwaccel     host RAM (numpy, lazy)      CUDA/NVDEC, VA-API, QSV, D3D11VA,
                                               VideoToolbox; software fallback
PyAV + decoder     host RAM (numpy, lazy)      explicit decoder: h264_cuvid,
  name                                         h264_v4l2m2m, h264_nvmpi (Jetson) ...
NVDEC zero-copy    CUDA memory (torch, CHW)    NV12->RGB on GPU; never touches host
=================  ==========================  =====================================
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Protocol

import av
import numpy as np

from ..config import DecoderOptions, HWAccel
from ..errors import DecoderError

log = logging.getLogger(__name__)


@dataclass(slots=True)
class DecodedPicture:
    pts: int | None
    time_base: Fraction | None
    keyframe: bool
    width: int
    height: int
    device: str
    materialize: Callable[[], Any] | None = None
    image: Any = None
    corrupt: bool = False

    @property
    def pts_time(self) -> float | None:
        if self.pts is None or self.time_base is None:
            return None
        return float(self.pts * self.time_base)


class VideoDecoder(Protocol):
    name: str
    device: str

    def decode(self, packet: av.Packet) -> list[DecodedPicture]: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------- helpers

def available_hw_devices() -> list[str]:
    try:
        from av.codec.hwaccel import hwdevices_available  # PyAV >= 14
    except ImportError:
        return []
    try:
        return [str(d) for d in hwdevices_available()]
    except Exception:  # noqa: BLE001
        return []


def _auto_order() -> tuple[str, ...]:
    if sys.platform == "darwin":
        return ("videotoolbox",)
    if sys.platform.startswith("win"):
        return ("cuda", "d3d11va", "qsv", "dxva2")
    return ("cuda", "vaapi", "qsv")


def _make_hwaccel(device_type: str, opts: DecoderOptions) -> Any | None:
    try:
        from av.codec.hwaccel import HWAccel as AVHWAccel
    except ImportError:
        log.warning("PyAV build lacks hwaccel support (needs PyAV >= 14); using software decode")
        return None
    device = str(opts.device_index) if device_type == "cuda" and opts.device_index else None
    return AVHWAccel(device_type=device_type, device=device,
                     allow_software_fallback=opts.allow_software_fallback)


def _set_low_delay(ctx: Any) -> None:
    """Enable AV_CODEC_FLAG_LOW_DELAY across PyAV API generations."""
    try:
        from av.codec.context import Flags  # PyAV >= 13
        ctx.flags |= Flags.low_delay
        return
    except (ImportError, AttributeError, TypeError):
        pass
    for attr in ("low_delay",):
        try:
            setattr(ctx, attr, True)
            return
        except (AttributeError, TypeError):
            continue
    try:
        ctx.options = {**dict(getattr(ctx, "options", {}) or {}), "flags": "low_delay"}
    except (AttributeError, TypeError):
        log.debug("could not set low_delay on codec context")


def _frame_keyframe(frame: Any) -> bool:
    kf = getattr(frame, "key_frame", None)
    if kf is not None:
        return bool(kf)
    pict = getattr(frame, "pict_type", None)
    return str(pict).endswith("I") if pict is not None else False


# --------------------------------------------------------------------------- PyAV

class PyAVDecoder:
    """FFmpeg decoder in its own CodecContext (rebuildable after corruption)."""

    def __init__(self, stream: Any, opts: DecoderOptions, hw_device: str | None) -> None:
        src = stream.codec_context
        codec_name = opts.decoder_name or src.name
        hwaccel = _make_hwaccel(hw_device, opts) if hw_device and not opts.decoder_name else None
        try:
            if hwaccel is not None:
                ctx = av.CodecContext.create(codec_name, "r", hwaccel=hwaccel)
            else:
                ctx = av.CodecContext.create(codec_name, "r")
        except Exception as exc:
            raise DecoderError(f"cannot create decoder {codec_name!r} (hw={hw_device}): {exc}") from exc
        if src.extradata:
            ctx.extradata = src.extradata
        try:
            ctx.thread_type = opts.thread_type
            ctx.thread_count = opts.thread_count
        except (AttributeError, ValueError, TypeError):
            pass
        _set_low_delay(ctx)
        self._ctx = ctx
        self._opts = opts
        self._fmt = opts.output_format
        self._size = opts.output_size
        self.name = f"pyav:{codec_name}" + (f"+{hw_device}" if hwaccel is not None else "")
        self.device = "cpu"

    def decode(self, packet: av.Packet) -> list[DecodedPicture]:
        out: list[DecodedPicture] = []
        for frame in self._ctx.decode(packet):
            out.append(
                DecodedPicture(
                    pts=frame.pts if frame.pts is not None else packet.pts,
                    time_base=frame.time_base or packet.time_base,
                    keyframe=_frame_keyframe(frame),
                    width=self._size[0] if self._size else frame.width,
                    height=self._size[1] if self._size else frame.height,
                    device="cpu",
                    materialize=self._materializer(frame),
                    corrupt=bool(getattr(frame, "is_corrupt", False)),
                )
            )
        return out

    def _materializer(self, frame: Any) -> Callable[[], np.ndarray]:
        fmt, size = self._fmt, self._size

        def materialize() -> np.ndarray:
            if size is not None:
                # One swscale pass does scaling and colour conversion together.
                return frame.reformat(width=size[0], height=size[1], format=fmt,
                                      interpolation="BILINEAR").to_ndarray()
            return frame.to_ndarray(format=fmt)

        return materialize

    def close(self) -> None:
        ctx, self._ctx = self._ctx, None
        if ctx is not None:
            try:
                ctx.close()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- NVDEC zero-copy

def nv12_to_rgb_chw(nv12: Any, height: int, width: int, *, bt709: bool, full_range: bool = False,
                    out_size: tuple[int, int] | None = None) -> Any:
    """NV12 (H*3/2, W) uint8 CUDA tensor -> RGB (3, H', W') uint8 CUDA tensor, all on GPU."""
    import torch
    import torch.nn.functional as F

    y = nv12[:height, :width].float()
    uv = nv12[height: height + height // 2, : (width // 2) * 2].reshape(height // 2, width // 2, 2).float()
    uv = uv.permute(2, 0, 1).unsqueeze(0)
    uv = F.interpolate(uv, size=(height, width), mode="nearest")[0]
    u, v = uv[0] - 128.0, uv[1] - 128.0
    if not full_range:
        y = (y - 16.0) * (255.0 / 219.0)
        u = u * (255.0 / 224.0)
        v = v * (255.0 / 224.0)
    if bt709:
        r = y + 1.5748 * v
        g = y - 0.187324 * u - 0.468124 * v
        b = y + 1.8556 * u
    else:
        r = y + 1.402 * v
        g = y - 0.344136 * u - 0.714136 * v
        b = y + 1.772 * u
    rgb = torch.stack((r, g, b))
    if out_size is not None and (out_size[0] != width or out_size[1] != height):
        rgb = F.interpolate(rgb.unsqueeze(0), size=(out_size[1], out_size[0]), mode="bilinear",
                            align_corners=False)[0]
    return rgb.clamp_(0.0, 255.0).to(torch.uint8)


class NvdecDecoder:
    """PyNvVideoCodec decoder; output never leaves device memory.

    The decoder's output surface is recycled on the next ``Decode`` call, so every picture
    is converted into a fresh RGB tensor immediately (a device-to-device kernel, no PCIe
    traffic). Downstream code receives a ``torch.Tensor`` on ``cuda:<n>`` ready for
    inference -- the zero-copy path the inference engine expects.
    """

    _CODECS = {"h264": "H264", "hevc": "HEVC", "h265": "HEVC"}

    def __init__(self, stream: Any, opts: DecoderOptions) -> None:
        try:
            import PyNvVideoCodec as nvc
            import torch
        except ImportError as exc:
            raise DecoderError("NVDEC zero-copy requires PyNvVideoCodec and torch") from exc
        if not torch.cuda.is_available():
            raise DecoderError("CUDA not available for NVDEC")
        src = stream.codec_context
        codec_key = self._CODECS.get(src.name)
        if codec_key is None:
            raise DecoderError(f"codec {src.name!r} not supported by NVDEC backend")
        self._nvc = nvc
        self._torch = torch
        self._dev_index = opts.device_index
        self.device = f"cuda:{opts.device_index}"
        self.name = f"nvdec:{src.name}"
        self._size = opts.output_size
        self._width = src.width
        self._height = src.height
        self._bt709 = (src.height or 0) >= 720
        try:
            self._dec = nvc.CreateDecoder(gpuid=opts.device_index, codec=getattr(nvc.cudaVideoCodec, codec_key),
                                          cudacontext=0, cudastream=0, usedevicememory=True)
        except Exception as exc:
            raise DecoderError(f"PyNvVideoCodec.CreateDecoder failed: {exc}") from exc

        # NVDEC needs Annex-B with in-band parameter sets. RTSP extradata built from the SDP
        # sprop-parameter-sets is already Annex-B; avcC/hvcC goes through the BSF.
        extradata = bytes(src.extradata or b"")
        self._prefix = extradata if extradata[:4] == b"\x00\x00\x00\x01" or extradata[:3] == b"\x00\x00\x01" \
            else b""
        self._bsf: Any = None
        if extradata and not self._prefix:
            try:
                from av.bitstream import BitStreamFilterContext
                self._bsf = BitStreamFilterContext(f"{src.name}_mp4toannexb", stream)
            except Exception as exc:
                raise DecoderError(f"cannot convert {src.name} to Annex-B: {exc}") from exc
        self._sent_prefix = False

    def _packet_data(self, payload: bytes, pts: int) -> tuple[Any, np.ndarray]:
        buf = np.frombuffer(payload, dtype=np.uint8)
        pd = self._nvc.PacketData()
        # Attribute names differ between PyNvVideoCodec releases.
        for ptr_attr, len_attr in (("bsl_data", "bsl"), ("data", "size")):
            if hasattr(pd, ptr_attr):
                setattr(pd, ptr_attr, int(buf.ctypes.data))
                setattr(pd, len_attr, int(buf.size))
                break
        else:
            raise DecoderError("unsupported PyNvVideoCodec PacketData layout")
        if hasattr(pd, "pts"):
            pd.pts = pts
        return pd, buf  # keep ``buf`` alive until Decode returns

    def decode(self, packet: av.Packet) -> list[DecodedPicture]:
        packets = self._bsf.filter(packet) if self._bsf is not None else [packet]
        out: list[DecodedPicture] = []
        torch = self._torch
        for pkt in packets:
            payload = bytes(pkt)
            if self._prefix and (pkt.is_keyframe or not self._sent_prefix):
                payload = self._prefix + payload
                self._sent_prefix = True
            pd, keep = self._packet_data(payload, pkt.pts or 0)
            with torch.cuda.device(self._dev_index):
                frames = self._dec.Decode(pd)
                for f in frames:
                    nv12 = torch.from_dlpack(f)
                    if nv12.dim() == 3:
                        nv12 = nv12.squeeze(-1)
                    h = self._height or (nv12.shape[0] * 2 // 3)
                    w = self._width or nv12.shape[1]
                    rgb = nv12_to_rgb_chw(nv12, h, w, bt709=self._bt709, out_size=self._size)
                    get_pts = getattr(f, "getPTS", None)
                    pts = int(get_pts()) if callable(get_pts) else pkt.pts
                    out.append(
                        DecodedPicture(
                            pts=pts, time_base=pkt.time_base, keyframe=bool(pkt.is_keyframe),
                            width=int(rgb.shape[2]), height=int(rgb.shape[1]), device=self.device,
                            image=rgb,
                        )
                    )
            del keep
        return out

    def close(self) -> None:
        self._dec = None
        self._bsf = None


# --------------------------------------------------------------------------- factory

def create_decoder(stream: Any, opts: DecoderOptions) -> VideoDecoder:
    """Instantiate the configured backend, degrading gracefully under AUTO."""
    if opts.hwaccel is HWAccel.NVDEC:
        try:
            return NvdecDecoder(stream, opts)
        except DecoderError as exc:
            if not opts.allow_software_fallback:
                raise
            log.warning("NVDEC zero-copy unavailable (%s); falling back to PyAV hwaccel", exc)
            return create_decoder(stream, _replace_hw(opts, HWAccel.AUTO))

    if opts.decoder_name or opts.hwaccel is HWAccel.NONE or stream.codec_context.name == "mjpeg":
        return PyAVDecoder(stream, opts, None)

    if opts.hwaccel is HWAccel.AUTO:
        available = set(available_hw_devices())
        for dev in _auto_order():
            if available and dev not in available:
                continue
            try:
                return PyAVDecoder(stream, opts, dev)
            except DecoderError as exc:
                log.debug("hwaccel %s rejected: %s", dev, exc)
        return PyAVDecoder(stream, opts, None)

    try:
        return PyAVDecoder(stream, opts, opts.hwaccel.value)
    except DecoderError:
        if not opts.allow_software_fallback:
            raise
        log.warning("hwaccel %s unavailable; using software decode", opts.hwaccel.value)
        return PyAVDecoder(stream, opts, None)


def _replace_hw(opts: DecoderOptions, hw: HWAccel) -> DecoderOptions:
    from dataclasses import replace

    return replace(opts, hwaccel=hw)
