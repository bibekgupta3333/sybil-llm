"""The F2MD LegacyChecks plausibility and consistency checks, one class per check, plus the receiver's node table.

Implements tracker 4.5. A port of `veins-f2md` commit 3b306aa `mdChecks/LegacyChecks.cc` (formulas and parameter
values: `docs/research-notes/veremi-plausibility-baseline-spec.md` §4-§5). Each receiver processes its BSMs in
reception order; every check sees the BSM and the receiver's node table *before* the BSM is added (F2MD `onBSM`).
A check returns its outputs (1 plausible, 0 implausible) or None when it does not apply (F2MD keeps the default 1).
Range and speed plausibility are vectorised over the stream; everything that reads the node table runs per BSM.

The node table and Kalman filters live in `node_table.py`. Known differences from the C++: float64 arithmetic
(F2MD's Kalman filters use float32); node-table eviction ignores F2MD's "already reported" exemption.
"""

from __future__ import annotations

import abc
import math

import numpy as np

from src.baselines.plausibility.config import PlausibilityConfig
from src.baselines.plausibility.node_table import NodeEntry, NodeTable
from src.baselines.plausibility.parser import ReceiverStream


def ieee_div(num: float, den: float) -> float:
    """num / den with IEEE-754 semantics (C++ double): x/0 = +-inf, 0/0 = NaN."""
    if den != 0.0:
        return num / den
    if num == 0.0 or math.isnan(num):
        return math.nan
    return math.copysign(math.inf, num) * math.copysign(1.0, den)


def heading_angle(hx: float, hy: float) -> float:
    """F2MD `calculateHeadingAngle`: clockwise angle of (hx, hy) from +x in degrees, (MDMLib.cc 25-39)."""
    angle = math.degrees(math.atan2(-hy, hx))
    if (hx >= 0 and hy > 0) or (hx < 0 and hy >= 0):
        angle += 360.0
    return angle


def rect_vertices(cx: float, cy: float, w: float, h: float, angle_deg: float) -> list[tuple[float, float]]:
    """F2MD `RectIntLib::RectangleVertices`: corners of a w x h rectangle rotated by angle_deg (counter-clockwise)."""
    a = math.radians(angle_deg)
    dx, dy = w / 2.0, h / 2.0
    c, s = math.cos(a), math.sin(a)
    corners = ((-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy))
    return [(cx + px * c - py * s, cy + px * s + py * c) for px, py in corners]


def convex_intersection_area(p: list[tuple[float, float]], q: list[tuple[float, float]]) -> float:
    """Area of the intersection of two counter-clockwise convex polygons (Sutherland-Hodgman clipping)."""
    poly = list(p)
    for k in range(len(q)):
        if len(poly) <= 2:
            return 0.0
        ax, ay = q[k]
        bx, by = q[(k + 1) % len(q)]
        side = [(bx - ax) * (y - ay) - (by - ay) * (x - ax) for x, y in poly]  # >= 0 inside (left of edge)
        clipped: list[tuple[float, float]] = []
        for j, (sx, sy) in enumerate(poly):
            tx, ty = poly[(j + 1) % len(poly)]
            s_val, t_val = side[j], side[(j + 1) % len(poly)]
            if s_val >= 0:
                clipped.append((sx, sy))
            if s_val * t_val < 0:
                ratio = s_val / (s_val - t_val)
                clipped.append((sx + ratio * (tx - sx), sy + ratio * (ty - sy)))
        poly = clipped
    if len(poly) <= 2:
        return 0.0
    area = 0.0
    for j, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(j + 1) % len(poly)]
        area += x1 * y2 - y1 * x2
    return abs(area) / 2.0


def rect_rect_factor(
    c1: tuple[float, float], c2: tuple[float, float], heading1: float, heading2: float, size: tuple[float, float]
) -> float:
    """F2MD `MDMLib::RectRectFactor`: intersection over union of two (width, length) rectangles (both of `size`)."""
    w, h = size
    area = convex_intersection_area(rect_vertices(*c1, w, h, heading1), rect_vertices(*c2, w, h, heading2))
    return area / (2.0 * w * h - area)


def max_min_dist(cur: float, old: float, time: float, accel: float, decel: float, vmax: float) -> tuple[float, float]:
    """F2MD `MDMLib::calculateMaxMinDist` (MDMLib.cc 594-644): (min, max) distance reachable between two speeds."""
    cur, old = max(cur, 0.0), max(old, 0.0)
    t1 = (cur - old + time * decel) / (accel + decel)
    t2 = time - t1
    max_speed = accel * t1 + old
    if max_speed > vmax:
        n1 = (vmax - old) / accel
        n2 = (vmax - cur) / decel
        max_d = old * n1 + 0.5 * accel * n1 * n1 + max_speed * n2 - 0.5 * decel * n2 * n2 + vmax * (time - n1 - n2)
    else:
        max_d = old * t1 + 0.5 * accel * t1 * t1 + max_speed * t2 - 0.5 * decel * t2 * t2
    min_speed = -decel * t2 + old
    if min_speed < 0:
        n1 = cur / accel
        n2 = old / decel
        min_d = old * n2 - 0.5 * decel * n2 * n2 + 0.5 * accel * n1 * n1
    else:
        min_d = old * t2 - 0.5 * decel * t2 * t2 + min_speed * t1 + 0.5 * accel * t1 * t1
    return min_d, max_d


class BsmContext:
    """What a check sees for BSM i: the stream, the sender's previous entry (or None), Δt and the node table."""

    __slots__ = ("stream", "i", "entry", "prev", "dt", "table")

    def __init__(self, stream: ReceiverStream, i: int, entry: NodeEntry | None, table: NodeTable) -> None:
        self.stream, self.i, self.entry, self.table = stream, i, entry, table
        self.prev = entry.last if entry is not None else -1
        self.dt = abs(float(stream.rcv_time[i] - stream.rcv_time[self.prev])) if entry is not None else math.nan


class Check(abc.ABC):
    """One F2MD check. `names` are its outputs (a Kalman check can produce two)."""

    names: tuple[str, ...] = ()
    needs_history: bool = True

    def __init__(self, config: PlausibilityConfig) -> None:
        self._config = config

    @abc.abstractmethod
    def evaluate(self, ctx: BsmContext) -> tuple[float, ...] | None:
        """Outputs for the BSM (one per name), or None when the check does not apply."""


class RangePlausibility(Check):
    """C1: the claimed position lies within MAX_PLAUSIBLE_RANGE of the receiver (vectorised in the suite)."""

    names = ("range_plausibility",)
    needs_history = False

    def evaluate(self, ctx: BsmContext) -> tuple[float, ...] | None:
        s, i = ctx.stream, ctx.i
        d = math.hypot(s.pos[i, 0] - s.rx_pos[i, 0], s.pos[i, 1] - s.rx_pos[i, 1])
        return (1.0 if d < self._config.max_plausible_range else 0.0,)

    def evaluate_stream(self, stream: ReceiverStream) -> np.ndarray:
        """(n,) outputs for every BSM at once."""
        d = np.hypot(*(stream.pos - stream.rx_pos).T)
        return np.where(d < self._config.max_plausible_range, 1.0, 0.0)


class SpeedPlausibility(Check):
    """C2: the claimed speed is below MAX_PLAUSIBLE_SPEED (vectorised in the suite)."""

    names = ("speed_plausibility",)
    needs_history = False

    def evaluate(self, ctx: BsmContext) -> tuple[float, ...] | None:
        speed = math.hypot(*ctx.stream.spd[ctx.i])
        return (1.0 if speed < self._config.max_plausible_speed else 0.0,)

    def evaluate_stream(self, stream: ReceiverStream) -> np.ndarray:
        """(n,) outputs for every BSM at once."""
        return np.where(np.hypot(*stream.spd.T) < self._config.max_plausible_speed, 1.0, 0.0)


class Intersection(Check):
    """C6: the sender's rectangle does not overlap the receiver's or another pseudonym's recent one (IoU test).

    Candidates: the receiver itself (Δt fixed at `receiver_inter_dt`) and every other pseudonym in the node table
    whose latest BSM is less than MAX_DELTA_INTER old. The output is the smallest pair value (F2MD keeps every value
    below 1; ThresholdApp fires when any is <= θ). Pairs whose bounding circles do not touch have value 1.
    """

    names = ("intersection",)
    needs_history = False

    def evaluate(self, ctx: BsmContext) -> tuple[float, ...] | None:
        cfg, s, i = self._config, ctx.stream, ctx.i
        mine = (float(s.pos[i, 0]), float(s.pos[i, 1]))
        my_conf = float(s.pos_conf[i, 0])
        my_angle = heading_angle(float(s.hed[i, 0]), float(s.hed[i, 1]))
        my_radius = self._radius(my_conf)
        value = 1.0
        if np.isfinite(s.rx_pos[i, 0]):
            rx = (float(s.rx_pos[i, 0]), float(s.rx_pos[i, 1]))
            rx_conf = float(s.rx_pos_conf[i, 0])
            if math.hypot(rx[0] - mine[0], rx[1] - mine[1]) < self._radius(rx_conf) + my_radius:
                rx_angle = heading_angle(float(s.rx_hed[i, 0]), float(s.rx_hed[i, 1]))
                pair = self._value(rx, rx_conf, rx_angle, mine, my_conf, my_angle, cfg.receiver_inter_dt)
                value = min(value, pair)
        table = ctx.table
        used = len(table.slots)
        if used:
            dt = np.abs(table.time[:used] - s.rcv_time[i])
            reach = self._radius_array(table.conf[:used]) + my_radius
            near = (dt < cfg.max_delta_inter) & (np.hypot(*(table.pos[:used] - s.pos[i]).T) < reach)
            sender = ctx.entry.slot if ctx.entry is not None else -1
            for slot in np.flatnonzero(near):
                if slot == sender:
                    continue
                other = (float(table.pos[slot, 0]), float(table.pos[slot, 1]))
                angle = heading_angle(float(table.hed[slot, 0]), float(table.hed[slot, 1]))
                pair = self._value(other, float(table.conf[slot]), angle, mine, my_conf, my_angle, float(dt[slot]))
                value = min(value, pair)
        return (value,)

    def _radius(self, conf: float) -> float:
        """Radius of a circle around the vehicle shape (rectangle: half its diagonal)."""
        return math.hypot(self._config.vehicle_width, self._config.vehicle_length) / 2

    def _radius_array(self, conf: np.ndarray) -> np.ndarray:
        """`_radius` for an array of confidences."""
        return np.full(conf.shape, self._radius(0.0))

    def _value(self, c1, conf1: float, a1: float, c2, conf2: float, a2: float, dt: float) -> float:
        """F2MD `IntersectionCheck`: 0 when RectRectFactor x (MAX_DELTA_INTER - dt) / MAX_DELTA_INTER > 0.5."""
        cfg = self._config
        size = (cfg.vehicle_width, cfg.vehicle_length)
        inter = rect_rect_factor(c1, c2, a1, a2, size) * ((cfg.max_delta_inter - dt) / cfg.max_delta_inter)
        return 0.0 if inter > cfg.intersection_limit else 1.0


class PositionConsistency(Check):
    """C3: the displacement since the previous BSM is below MAX_PLAUSIBLE_SPEED * Δt."""

    names = ("position_consistency",)

    def evaluate(self, ctx: BsmContext) -> tuple[float, ...] | None:
        s = ctx.stream
        d = math.hypot(*(s.pos[ctx.i] - s.pos[ctx.prev]))
        return (1.0 if d < self._config.max_plausible_speed * ctx.dt else 0.0,)


class SpeedConsistency(Check):
    """C4: the speed change since the previous BSM respects MAX_PLAUSIBLE_ACCEL / DECEL * Δt."""

    names = ("speed_consistency",)

    def evaluate(self, ctx: BsmContext) -> tuple[float, ...] | None:
        s, cfg = ctx.stream, self._config
        delta = math.hypot(*s.spd[ctx.i]) - math.hypot(*s.spd[ctx.prev])
        limit = (cfg.max_plausible_accel if delta > 0 else cfg.max_plausible_decel) * ctx.dt
        return (1.0 if abs(delta) < limit else 0.0,)


class BeaconFrequency(Check):
    """C8: two BSMs of one pseudonym are at least MAX_BEACON_FREQUENCY apart (arrival times)."""

    names = ("beacon_frequency",)

    def evaluate(self, ctx: BsmContext) -> tuple[float, ...] | None:
        s = ctx.stream
        gap = float(s.rcv_time[ctx.i] - s.rcv_time[ctx.prev])
        return (0.0 if gap < self._config.max_beacon_frequency else 1.0,)


class PositionHeadingConsistency(Check):
    """C7: the claimed heading agrees with the displacement direction (Δt < POS_HEADING_TIME)."""

    names = ("position_heading_consistency",)

    def evaluate(self, ctx: BsmContext) -> tuple[float, ...] | None:
        cfg, s, i = self._config, ctx.stream, ctx.i
        if not ctx.dt < cfg.pos_heading_time:
            return None
        move = s.pos[i] - s.pos[ctx.prev]
        if math.hypot(*move) < cfg.pos_heading_min_distance or math.hypot(*s.spd[i]) < cfg.pos_heading_min_speed:
            return (1.0,)
        delta = abs(
            heading_angle(float(s.hed[i, 0]), float(s.hed[i, 1])) - heading_angle(float(move[0]), float(move[1]))
        )
        if delta > 180:
            delta = 360 - delta
        return (0.0 if cfg.max_heading_change < delta else 1.0,)


class PositionSpeedConsistency(Check):
    """C5: the displacement lies within the min / max distance reachable between the two claimed speeds."""

    names = ("position_speed_consistency",)

    def evaluate(self, ctx: BsmContext) -> tuple[float, ...] | None:
        cfg, s, i = self._config, ctx.stream, ctx.i
        if not ctx.dt < cfg.max_time_delta:
            return None
        d = math.hypot(*(s.pos[i] - s.pos[ctx.prev]))
        cur, old = math.hypot(*s.spd[i]), math.hypot(*s.spd[ctx.prev])
        v_min = min(cur, old)
        min_d, max_d = max_min_dist(
            cur, old, ctx.dt, cfg.max_plausible_accel, cfg.max_plausible_decel, cfg.max_plausible_speed
        )
        addon = max(0.0, cfg.max_mgt_rng_down + 0.3571 * v_min - 0.01694 * v_min * v_min)
        bad = (d - min_d + addon < 0) or (max_d - d + cfg.max_mgt_rng_up < 0)
        return (0.0 if bad else 1.0,)


class PositionSpeedMaxConsistency(Check):
    """C5b: the speed implied by the displacement is within the claimed speeds +- (accel + MAX_MGT_RNG) * Δt."""

    names = ("position_speed_max_consistency",)

    def evaluate(self, ctx: BsmContext) -> tuple[float, ...] | None:
        cfg, s, i = self._config, ctx.stream, ctx.i
        if not ctx.dt < cfg.max_time_delta:
            return None
        implied = ieee_div(math.hypot(*(s.pos[i] - s.pos[ctx.prev])), ctx.dt)
        cur, old = math.hypot(*s.spd[i]), math.hypot(*s.spd[ctx.prev])
        if max(cur, old) - implied > (cfg.max_plausible_decel + cfg.max_mgt_rng) * ctx.dt:
            return (0.0,)
        if implied - min(cur, old) > (cfg.max_plausible_accel + cfg.max_mgt_rng) * ctx.dt:
            return (0.0,)
        return (1.0,)


class SuddenAppearance(Check):
    """C9 (diagnostic, not in the decision): a new or long-silent pseudonym appears within MAX_SA_RANGE."""

    names = ("sudden_appearance",)
    needs_history = False

    def evaluate(self, ctx: BsmContext) -> tuple[float, ...] | None:
        cfg, s, i = self._config, ctx.stream, ctx.i
        if ctx.entry is not None and not ctx.dt > cfg.max_sa_time:
            return None
        d = math.hypot(*(s.pos[i] - s.rx_pos[i]))
        return (0.0 if d < cfg.max_sa_range else 1.0,)


class _KalmanCheck(Check):
    """Shared Kalman helpers: confidence clamps and the 0.5 binarisation."""

    def _binary(self, value: float) -> float:
        """LegacyChecks: NaN -> 0, then 0 / 1 at `kalman_binarise`; CaTCh (kalman_binarise None): clip below 0."""
        if math.isnan(value):
            value = 0.0
        if self._config.kalman_binarise is None:
            return max(value, 0.0)
        return 0.0 if value < self._config.kalman_binarise else 1.0

    def _pos_conf(self, ctx: BsmContext) -> tuple[float, float]:
        conf = ctx.stream.pos_conf[ctx.i]
        floor = self._config.kalman_min_pos_range
        return max(float(conf[0]), floor), max(float(conf[1]), floor)

    def _spd_conf(self, ctx: BsmContext) -> tuple[float, float]:
        conf = ctx.stream.spd_conf[ctx.i]
        floor = self._config.kalman_min_speed_range
        return max(float(conf[0]), floor), max(float(conf[1]), floor)


class KalmanPositionSpeed(_KalmanCheck):
    """C11 (KalmanPSCP, KalmanPSCS): Kalman_SVI on (x, vx, y, vy) with U = (ax, ax, ay, ay)."""

    names = ("kalman_pscp", "kalman_pscs")

    def evaluate(self, ctx: BsmContext) -> tuple[float, ...] | None:
        cfg, s, i, f = self._config, ctx.stream, ctx.i, ctx.entry.filters
        if not ctx.dt < cfg.max_kalman_time:
            f.reset_svi(s.pos[i], s.spd[i])
            return None
        cx, cy = self._pos_conf(ctx)
        cvx, cvy = self._spd_conf(ctx)
        ax, ay = float(s.acl[i, 0]), float(s.acl[i, 1])
        dx, dvx = f.svi_x.step(ctx.dt, (ax, ax), (float(s.pos[i, 0]), float(s.spd[i, 0])), (cx, cvx))
        dy, dvy = f.svi_y.step(ctx.dt, (ay, ay), (float(s.pos[i, 1]), float(s.spd[i, 1])), (cy, cvy))
        pos = 1 - ieee_div(math.hypot(dx, dy), cfg.kalman_pos_range * cx * ctx.dt)
        spd = 1 - ieee_div(math.hypot(dvx, dvy), cfg.kalman_speed_range * cvx * ctx.dt)
        return self._binary(pos), self._binary(spd)


class KalmanPositionSpeedScalar(_KalmanCheck):
    """C12 (KalmanPSCSP, KalmanPSCSS): Kalman_SC on (accumulated distance, speed) with U = (|a|, |a|)."""

    names = ("kalman_pscsp", "kalman_pscss")

    def evaluate(self, ctx: BsmContext) -> tuple[float, ...] | None:
        cfg, s, i, f = self._config, ctx.stream, ctx.i, ctx.entry.filters
        speed = math.hypot(*s.spd[i])
        if not ctx.dt < cfg.max_kalman_time:
            f.reset_sc(speed)
            return None
        cx, _ = self._pos_conf(ctx)
        cvx, _ = self._spd_conf(ctx)
        acl = math.hypot(*s.acl[i])
        f.sc_pos += math.hypot(*(s.pos[i] - s.pos[ctx.prev]))
        d_pos, d_spd = f.sc.step(ctx.dt, (acl, acl), (f.sc_pos, speed), (cx, cvx))
        pos = 1 - ieee_div(d_pos, cfg.kalman_pos_range * cx * ctx.dt)
        spd = 1 - ieee_div(d_spd, cfg.kalman_speed_range * cvx * ctx.dt)
        return self._binary(pos), self._binary(spd)


class KalmanPosition(_KalmanCheck):
    """C13 (KalmanPCC): Kalman_SI on position, predict without input."""

    names = ("kalman_pcc",)

    def evaluate(self, ctx: BsmContext) -> tuple[float, ...] | None:
        return _position_si(self, ctx, (0.0, 0.0))


class KalmanPositionAcc(_KalmanCheck):
    """C14 (KalmanPACS): the same Kalman_SI as C13, updated again with U = the previous BSM's velocity."""

    names = ("kalman_pacs",)

    def evaluate(self, ctx: BsmContext) -> tuple[float, ...] | None:
        prev_spd = ctx.stream.spd[ctx.prev]
        return _position_si(self, ctx, (float(prev_spd[0]), float(prev_spd[1])))


def _position_si(check: _KalmanCheck, ctx: BsmContext, u: tuple[float, float]) -> tuple[float, ...] | None:
    """C13 / C14 body: step the position Kalman_SI and score 1 - |Δ| / (4 * KALMAN_POS_RANGE * cx * Δt)."""
    cfg, s, i, f = check._config, ctx.stream, ctx.i, ctx.entry.filters
    if not ctx.dt < cfg.max_kalman_time:
        f.reset_si(s.pos[i])
        return None
    cx, cy = check._pos_conf(ctx)
    dx = f.si_x.step(ctx.dt, u[0], float(s.pos[i, 0]), cx)
    dy = f.si_y.step(ctx.dt, u[1], float(s.pos[i, 1]), cy)
    return (check._binary(1 - ieee_div(math.hypot(dx, dy), 4 * cfg.kalman_pos_range * cx * ctx.dt)),)


class KalmanSpeed(_KalmanCheck):
    """C15 (KalmanSCC): Kalman_SI on velocity with U = acceleration."""

    names = ("kalman_scc",)

    def evaluate(self, ctx: BsmContext) -> tuple[float, ...] | None:
        cfg, s, i, f = self._config, ctx.stream, ctx.i, ctx.entry.filters
        if not ctx.dt < cfg.max_kalman_time:
            f.reset_vi(s.spd[i])
            return None
        cvx, cvy = self._spd_conf(ctx)
        dvx = f.vi_x.step(ctx.dt, float(s.acl[i, 0]), float(s.spd[i, 0]), cvx)
        dvy = f.vi_y.step(ctx.dt, float(s.acl[i, 1]), float(s.spd[i, 1]), cvy)
        return (self._binary(1 - ieee_div(math.hypot(dvx, dvy), cfg.kalman_speed_range * cvx * ctx.dt)),)


# LegacyChecks::CheckBSM order (spec §5): C1, C2, C6 always; with history C3, C4, C8, C7, C5, C5b, C9, C11-C15.
CHECK_CLASSES: tuple[type[Check], ...] = (
    RangePlausibility,
    SpeedPlausibility,
    Intersection,
    PositionConsistency,
    SpeedConsistency,
    BeaconFrequency,
    PositionHeadingConsistency,
    PositionSpeedConsistency,
    PositionSpeedMaxConsistency,
    SuddenAppearance,
    KalmanPositionSpeed,
    KalmanPositionSpeedScalar,
    KalmanPosition,
    KalmanPositionAcc,
    KalmanSpeed,
)


class CheckResults:
    """Outputs of every check on one stream: name -> (n,) value (1 when not evaluated) and (n,) evaluated."""

    def __init__(self, names: list[str], n: int) -> None:
        self.value = {name: np.ones(n) for name in names}
        self.evaluated = {name: np.zeros(n, dtype=bool) for name in names}


class CheckSuite:
    """Runs a set of checks over a receiver stream in reception order, maintaining the node table."""

    def __init__(self, config: PlausibilityConfig, check_classes: tuple[type[Check], ...] = CHECK_CLASSES) -> None:
        self._config = config
        self._checks = [cls(config) for cls in check_classes]
        self.names = [name for check in self._checks for name in check.names]

    def run(self, stream: ReceiverStream) -> CheckResults:
        """Scores every BSM of the stream."""
        n = len(stream)
        results = CheckResults(self.names, n)
        vectorised = [c for c in self._checks if hasattr(c, "evaluate_stream")]
        for check in vectorised:
            results.value[check.names[0]] = check.evaluate_stream(stream)
            results.evaluated[check.names[0]][:] = True
        sequential = [c for c in self._checks if c not in vectorised]
        table = NodeTable(self._config)
        for i in range(n):
            entry = table.get(int(stream.pseudo[i]))
            ctx = BsmContext(stream, i, entry, table)
            for check in sequential:
                if check.needs_history and entry is None:
                    continue
                out = check.evaluate(ctx)
                if out is None:
                    continue
                for name, value in zip(check.names, out):
                    results.value[name][i] = value
                    results.evaluated[name][i] = True
            table.add(stream, i)
        return results
