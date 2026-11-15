"""The interface every detector implements, and the registry the runner looks detectors up in.

A detector is fitted on a `WindowSet` (x, mask) - a `LabelledWindowSet` (x, mask, labels) only in the `supervised`
regime - and scores windows from x and mask alone. It never sees a split name, a vehicle key or a test label.

Minimal detector:

    @register
    class MyDetector(Detector):
        name = "my_detector"
        regimes = (cfg.BENIGN_ONLY, cfg.BENIGN_GRIDFOLDERS, cfg.MIX_ALL, cfg.MIX_GRIDFOLDERS)
        is_stochastic = True
        score_names = ("reconstruction",)

        def fit(self, train, val, seed):
            ...

        def score(self, x, mask):
            return {"reconstruction": errors}
"""

from __future__ import annotations

import abc
import importlib
import pkgutil
from pathlib import Path
from typing import ClassVar

import numpy as np

from baselinepaper.experiments import config as cfg
from baselinepaper.experiments.data import FeatureStatistics, LabelledWindowSet, WindowSet

_REGISTRY: dict[str, type["Detector"]] = {}
_DISCOVERED = False


class Detector(abc.ABC):
    """One anomaly detector under the protocol.

    Class attributes (set by every subclass):
        name: registry key and run-folder name (lower case, digits, underscores).
        regimes: training regimes the detector supports (`config.REGIMES`), or `(config.CONTROL_REGIME,)` for a
            control that is not trained.
        is_stochastic: True if results depend on the seed (run for every `config.seeds`); False = run once with
            `config.seeds[0]`.
        score_names: names of the scores `score` returns; each gets its own score files and table row. Letters,
            digits and hyphens only (no underscore: file names are `scores_<score>_<split>.npy`).
        reused_modules: dotted names of packages outside `baselinepaper.experiments` whose code the detector
            imports (e.g. `"task.autoencoder_benign_t32"`); the runner records the sha256 of their .py files.

    Instance attributes (set by the runner before `fit`):
        regime: the regime of this run.
        device: "cpu", "mps" or "cuda" (already resolved; ignore it if the detector does not use torch).
        statistics: `WindowCache.regime_statistics(regime)` for the regimes in `config.RESTANDARDISED_REGIMES`,
            else None. A detector of such a regime must apply `self.standardise(x, mask)` to every input, in
            `fit` and in `score`; for the other regimes `standardise` returns its input unchanged.
    """

    name: ClassVar[str]
    regimes: ClassVar[tuple[str, ...]]
    is_stochastic: ClassVar[bool]
    score_names: ClassVar[tuple[str, ...]]
    reused_modules: ClassVar[tuple[str, ...]] = ()

    def __init__(self, regime: str, device: str = "cpu", statistics: FeatureStatistics | None = None) -> None:
        if regime not in self.regimes:
            raise ValueError(f"detector {self.name!r} does not support regime {regime!r} (supports {self.regimes})")
        self.regime = regime
        self.device = device
        self.statistics = statistics

    def standardise(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """x re-standardised with the regime's training-pool statistics (unchanged where none apply)."""
        return x if self.statistics is None else self.statistics.apply(x, mask)

    def hyperparameters(self) -> dict:
        """Every fixed setting of the detector (JSON-ready); written to config.json before fitting."""
        return {}

    @abc.abstractmethod
    def fit(
        self, train: WindowSet | LabelledWindowSet | None, val: WindowSet | LabelledWindowSet | None, seed: int
    ) -> None:
        """Fits the detector.

        Args:
            train: the regime's training windows (`x` (N, 32, 13) float32 in the cached normalisation, `mask`
                (N, 32) bool; plus `labels` (N,) int8, 0 benign / 1 GridSybil, only in the `supervised` regime).
                None for a control (`config.CONTROL_REGIME`).
            val: the regime's validation windows, same composition and type as `train`; the only data model
                selection, early stopping or calibration may use.
            seed: seeds every random choice (initialisation, batch order, subsamples).
        """

    @abc.abstractmethod
    def score(self, x: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
        """Scores windows; higher = more anomalous.

        Args:
            x: (N, 32, 13) float32 in the cached normalisation (may be a read-only memory map; batch internally).
            mask: (N, 32) bool prefix mask.

        Returns:
            {score name: (N,) finite floats} with exactly the keys of `score_names`, rows in the input order.
        """

    def history(self) -> list[dict]:
        """Training history, one dict per epoch (or iteration); empty if there is none. Written to history.json."""
        return []

    def fit_info(self) -> dict:
        """Facts about the fitted model for summary.json: parameter count, best epoch, subsample size, ..."""
        return {}

    def save(self, run_dir: Path) -> None:
        """Optional: writes model artefacts (checkpoint, fitted statistics) into the run folder."""


def register(cls: type[Detector]) -> type[Detector]:
    """Class decorator: adds a detector to the registry under `cls.name`."""
    for attr in ("name", "regimes", "is_stochastic", "score_names"):
        if not hasattr(cls, attr):
            raise TypeError(f"{cls.__name__} does not define `{attr}`")
    if not cls.name.replace("_", "").isalnum() or cls.name != cls.name.lower():
        raise ValueError(f"detector name {cls.name!r}: lower-case letters, digits and underscores only")
    for score in cls.score_names:
        if not score.replace("-", "").isalnum():
            raise ValueError(f"score name {score!r} of {cls.name!r}: letters, digits and hyphens only")
    unknown = set(cls.regimes) - set(cfg.REGIMES) - {cfg.CONTROL_REGIME}
    if unknown or not cls.regimes:
        raise ValueError(f"detector {cls.name!r}: unknown or empty regimes {sorted(unknown)}")
    if cls.name in _REGISTRY and _REGISTRY[cls.name].__qualname__ != cls.__qualname__:
        raise ValueError(f"detector name {cls.name!r} is already registered by {_REGISTRY[cls.name].__name__}")
    _REGISTRY[cls.name] = cls
    return cls


def _discover() -> None:
    """Imports every module of this package once, so their `@register` classes are known."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    package = importlib.import_module("baselinepaper.experiments.detectors")
    for module in pkgutil.iter_modules(package.__path__):
        importlib.import_module(f"{package.__name__}.{module.name}")
    _DISCOVERED = True


def get_detector(name: str) -> type[Detector]:
    """The detector class registered under `name`."""
    _discover()
    if name not in _REGISTRY:
        raise KeyError(f"unknown detector {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> dict[str, type[Detector]]:
    """{name: class} of every registered detector."""
    _discover()
    return dict(sorted(_REGISTRY.items()))
