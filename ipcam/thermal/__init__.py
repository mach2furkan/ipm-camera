"""Thermal imaging: radiometric stream, display pipeline, analytics and bi-spectral fusion."""

from .bispectral import ChannelRegistration, ThermalConfirmation, ThermalEvidence
from .radiometry import (
    Blob,
    FireCandidate,
    FireDetector,
    HotspotDetector,
    PersonCandidateConfig,
    PersonCandidates,
    RoiAlarm,
    RoiMonitor,
    robust_background,
)
from .render import PALETTES, PlateauAGC, colorize, downsample_grid, encode_png, isotherm, palette_lut
from .stream import ThermalFrame, ThermalLayout, ThermalPayloadDecoder, ThermalStreamReader, decode_metadata

__all__ = [
    "PALETTES",
    "Blob",
    "ChannelRegistration",
    "FireCandidate",
    "FireDetector",
    "HotspotDetector",
    "PersonCandidateConfig",
    "PersonCandidates",
    "PlateauAGC",
    "RoiAlarm",
    "RoiMonitor",
    "ThermalConfirmation",
    "ThermalEvidence",
    "ThermalFrame",
    "ThermalLayout",
    "ThermalPayloadDecoder",
    "ThermalStreamReader",
    "colorize",
    "decode_metadata",
    "downsample_grid",
    "encode_png",
    "isotherm",
    "palette_lut",
    "robust_background",
]
