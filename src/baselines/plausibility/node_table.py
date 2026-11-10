"""Per-receiver state of the F2MD detector: the node table and the Kalman filters kept per pseudonym.

Implements tracker 4.5. Ports of `veins-f2md` commit 3b306aa `mdBase/NodeTable.cc` (at most MAX_NODES_LENGTH
pseudonyms; a new one evicts the pseudonym with the oldest latest BSM), `mdBase/NodeMDMHistory.cc` (`initKalman`)
and `mdSupport/kalmanLib/` (`Kalman_SVI`, `Kalman_SC`, `Kalman_SI`; spec §5.1). All F2MD filters have H = I and
diagonal Q, R, P0, so `Kalman_SVI` splits into two independent 2-state filters (x and y axes) and `Kalman_SI` into
two scalar filters. Arithmetic is float64 (F2MD: float32). Eviction ignores F2MD's "already reported" exemption.
"""

from __future__ import annotations

import math

import numpy as np

from src.baselines.plausibility.config import PlausibilityConfig
from src.baselines.plausibility.parser import ReceiverStream


class Kalman2:
    """F2MD 2-state Kalman filter (position, velocity) per axis: A = [[1, T], [0, 1]], B = (T^2/2, T), H = I.

    Used for one axis of `Kalman_SVI` and for `Kalman_SC`. B multiplies U element-wise (F2MD `multiply1D`).
    """

    def __init__(self, q: tuple[float, float]) -> None:
        self._q = q
        self.x = [0.0, 0.0]
        self.p = [0.0, 0.0, 0.0, 0.0]  # row-major 2 x 2

    def reset(self, x0: float, x1: float, p0: tuple[float, float]) -> None:
        """`setInitial`: state (x0, x1), diagonal covariance p0."""
        self.x = [x0, x1]
        self.p = [p0[0], 0.0, 0.0, p0[1]]

    def step(
        self, t: float, u: tuple[float, float], z: tuple[float, float], r: tuple[float, float]
    ) -> tuple[float, float]:
        """`predict(U)` then `correct(Z)` with R = diag(r); returns |X_corrected - Z| per state."""
        p00, p01, p10, p11 = self.p
        x0 = self.x[0] + t * self.x[1] + 0.5 * t * t * u[0]
        x1 = self.x[1] + t * u[1]
        a00 = p00 + t * p10 + t * (p01 + t * p11) + self._q[0]
        a01 = p01 + t * p11
        a10 = p10 + t * p11
        a11 = p11 + self._q[1]
        s00, s01, s10, s11 = a00 + r[0], a01, a10, a11 + r[1]
        det = s00 * s11 - s01 * s10
        i00, i01, i10, i11 = s11 / det, -s01 / det, -s10 / det, s00 / det
        k00, k01 = a00 * i00 + a01 * i10, a00 * i01 + a01 * i11
        k10, k11 = a10 * i00 + a11 * i10, a10 * i01 + a11 * i11
        e0, e1 = z[0] - x0, z[1] - x1
        x0, x1 = x0 + k00 * e0 + k01 * e1, x1 + k10 * e0 + k11 * e1
        self.p = [
            (1 - k00) * a00 - k01 * a10,
            (1 - k00) * a01 - k01 * a11,
            -k10 * a00 + (1 - k11) * a10,
            -k10 * a01 + (1 - k11) * a11,
        ]
        self.x = [x0, x1]
        return abs(x0 - z[0]), abs(x1 - z[1])


class Kalman1:
    """F2MD `Kalman_SI` on one axis: A = 1, B = T, diagonal Q, R and P (both axes are independent)."""

    def __init__(self, q: float) -> None:
        self._q = q
        self.x = 0.0
        self.p = 0.0

    def reset(self, x0: float, p0: float) -> None:
        """`setInitial`."""
        self.x, self.p = x0, p0

    def step(self, t: float, u: float, z: float, r: float) -> float:
        """`predict(U)` (u = 0 for the no-input predict) then `correct(Z)`; returns |X_corrected - Z|."""
        x = self.x + t * u
        p = self.p + self._q
        k = p / (p + r)
        self.x = x + k * (z - x)
        self.p = (1 - k) * p
        return abs(self.x - z)


class PseudonymFilters:
    """The Kalman filters F2MD keeps per pseudonym (`MDMHistory`): SVI (x and y axes), SC, SI (position), VI."""

    def __init__(self, config: PlausibilityConfig) -> None:
        self._config = config
        self.svi_x, self.svi_y = Kalman2(config.svi_q), Kalman2(config.svi_q)
        self.sc = Kalman2(config.sc_q)
        self.sc_pos = 0.0
        self.si_x, self.si_y = Kalman1(config.si_q), Kalman1(config.si_q)
        self.vi_x, self.vi_y = Kalman1(config.si_q), Kalman1(config.si_q)

    def init_all(self, pos: np.ndarray, spd: np.ndarray) -> None:
        """`initKalman` on a pseudonym's first BSM."""
        self.reset_svi(pos, spd)
        self.reset_sc(float(math.hypot(spd[0], spd[1])))
        self.reset_si(pos)
        self.reset_vi(spd)

    def reset_svi(self, pos: np.ndarray, spd: np.ndarray) -> None:
        """`Kalman_SVI::setInitial(x, y, vx, vy)`."""
        self.svi_x.reset(float(pos[0]), float(spd[0]), self._config.svi_p0)
        self.svi_y.reset(float(pos[1]), float(spd[1]), self._config.svi_p0)

    def reset_sc(self, speed: float) -> None:
        """`Kalman_SC::setInitial(0, speed)` (also resets the accumulated distance)."""
        self.sc.reset(0.0, speed, self._config.sc_p0)
        self.sc_pos = 0.0

    def reset_si(self, pos: np.ndarray) -> None:
        """`Kalman_SI::setInitial(x, y)` of the position filter."""
        self.si_x.reset(float(pos[0]), self._config.si_p0)
        self.si_y.reset(float(pos[1]), self._config.si_p0)

    def reset_vi(self, spd: np.ndarray) -> None:
        """`Kalman_SI::setInitial(vx, vy)` of the speed filter."""
        self.vi_x.reset(float(spd[0]), self._config.si_p0)
        self.vi_y.reset(float(spd[1]), self._config.si_p0)


class NodeEntry:
    """One pseudonym in the receiver's node table: its latest BSM (index into the stream) and its filters."""

    __slots__ = ("pseudo", "last", "slot", "filters")

    def __init__(self, pseudo: int, last: int, slot: int, filters: PseudonymFilters) -> None:
        self.pseudo, self.last, self.slot, self.filters = pseudo, last, slot, filters


class NodeTable:
    """F2MD `NodeTable` of one receiver: at most `max_nodes` pseudonyms; a new one evicts the oldest latest BSM.

    Parallel arrays (time, position, heading of each slot's latest BSM) let the overlap check filter candidates
    without a Python loop.
    """

    def __init__(self, config: PlausibilityConfig) -> None:
        self._config = config
        self.entries: dict[int, NodeEntry] = {}
        self.slots: list[NodeEntry] = []
        size = config.max_nodes
        self.time = np.full(size, np.nan)
        self.send_time = np.full(size, np.inf)
        self.pos = np.zeros((size, 2))
        self.hed = np.zeros((size, 2))
        self.conf = np.zeros(size)  # latest BSM's position confidence (x), for CaTCh

    def get(self, pseudo: int) -> NodeEntry | None:
        """The pseudonym's entry, or None."""
        return self.entries.get(pseudo)

    def add(self, stream: ReceiverStream, i: int) -> None:
        """Adds BSM i after its checks: updates the pseudonym's entry or inserts it (initialising its filters)."""
        pseudo = int(stream.pseudo[i])
        entry = self.entries.get(pseudo)
        if entry is None:
            filters = PseudonymFilters(self._config)
            filters.init_all(stream.pos[i], stream.spd[i])
            if len(self.slots) < self._config.max_nodes:
                slot = len(self.slots)
                entry = NodeEntry(pseudo, i, slot, filters)
                self.slots.append(entry)
            else:
                slot = int(np.argmin(self.send_time))  # first slot with the oldest latest sending time
                del self.entries[self.slots[slot].pseudo]
                entry = NodeEntry(pseudo, i, slot, filters)
                self.slots[slot] = entry
            self.entries[pseudo] = entry
        entry.last = i
        slot = entry.slot
        self.time[slot] = stream.rcv_time[i]
        self.send_time[slot] = stream.send_time[i]
        self.pos[slot] = stream.pos[i]
        self.hed[slot] = stream.hed[i]
        self.conf[slot] = stream.pos_conf[i, 0]
