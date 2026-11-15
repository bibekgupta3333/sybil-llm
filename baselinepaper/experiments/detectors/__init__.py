"""Detectors of the protocol. Every module in this package is imported on first use of the registry, so a new
detector only needs its own file with a `@register` class (no edit here)."""

from baselinepaper.experiments.detectors.base import Detector, available, get_detector, register

__all__ = ["Detector", "available", "get_detector", "register"]
