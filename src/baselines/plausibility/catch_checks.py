"""The F2MD CaTChChecks: the checks of `checks.py` with continuous outputs that use the BSM confidence fields.

Implements tracker 4.5 (spec `docs/research-notes/veremi-plausibility-baseline-spec.md` §6). A port of `veins-f2md`
commit 3b306aa `mdChecks/CaTChChecks.cc` and the `MDMLib.cc` factor functions it calls. Confidences: a BSM's
position confidence is `pos_noise.x`, its speed confidence `|spd_noise|`, its heading confidence `hed_noise.x`; the
receiver's position confidence is the `pos_noise.x` of its own latest fix. Beacon frequency and the Kalman checks
are shared with `checks.py` (the Kalman outputs stay continuous when `kalman_binarise` is None).

Known approximation: F2MD computes the ellipse-ellipse overlap area exactly (`ellipseIntLib`, a quartic root
solver); here it is estimated on `ellipse_vertices` fixed points spread over the first ellipse (`EllipseOverlap`).
"""

from __future__ import annotations

import math

import numpy as np

from src.baselines.plausibility import checks as legacy
from src.baselines.plausibility.checks import BsmContext, Check, heading_angle, ieee_div, max_min_dist
from src.baselines.plausibility.config import PlausibilityConfig


def safe_acos(x: float) -> float:
    """`MDMLib::SafeAcos`: acos with the argument clipped to [-1, 1]."""
    return math.acos(min(max(x, -1.0), 1.0))


def circle_segment(radius: float, int_distance: float) -> float:
    """`MDMLib::calculateCircleSegment` (MDMLib.cc 81-113): area of a circle cut at depth `int_distance`."""
    if radius <= 0 or int_distance <= 0:
        return 0.0
    if int_distance > 2 * radius:
        return math.pi * radius * radius
    if radius > int_distance:
        return radius * radius * safe_acos((radius - int_distance) / radius) - (radius - int_distance) * math.sqrt(
            2 * radius * int_distance - int_distance * int_distance
        )
    other = 2 * radius - int_distance
    area = radius * radius * safe_acos((radius - other) / radius) - (radius - other) * math.sqrt(
        2 * radius * other - other * other
    )
    return math.pi * radius * radius - area


def circle_circle_intersection(r0: float, r1: float, d: float) -> float:
    """`MDMLib::calculateCircleCircleIntersection` (MDMLib.cc 115-150): overlap area of two circles."""
    if r0 <= 0 or r1 <= 0:
        return 0.0
    rr0, rr1 = r0 * r0, r1 * r1
    if d > r1 + r0:
        return 0.0
    if d <= abs(r0 - r1):
        return math.pi * (rr1 if r0 >= r1 else rr0)
    phi = safe_acos(ieee_div(rr0 + d * d - rr1, 2 * r0 * d)) * 2
    theta = safe_acos(ieee_div(rr1 + d * d - rr0, 2 * r1 * d)) * 2
    return 0.5 * theta * rr1 - 0.5 * rr1 * math.sin(theta) + 0.5 * phi * rr0 - 0.5 * rr0 * math.sin(phi)


def _split_distances(d: float, r1: float, r2: float, rng: float) -> tuple[float, float]:
    """The (d1, d2) split of the centre distance shared by the MDMLib factor functions."""
    d1 = d2 = 0.0
    if d > 0:
        d1 = (r1 * r1 + d * d - r2 * r2) / (2 * d)
        d2 = (r2 * r2 + d * d - r1 * r1) / (2 * d)
        if (d1 + r1) < rng / 2 and (d2 + r2) > rng / 2:
            d2 = d2 - (rng / 2 - (d1 + r1))
            d1 = d1 + (rng / 2 - (d1 + r1))
        if (d2 + r2) < rng / 2 and (d1 + r1) > rng / 2:
            d1 = d1 - (rng / 2 - (d2 + r2))
            d2 = d2 + (rng / 2 - (d2 + r2))
    return d1, d2


def circle_circle_factor(d: float, r1: float, r2: float, rng: float) -> float:
    """`MDMLib::CircleCircleFactor` (MDMLib.cc 160-208)."""
    d1, d2 = _split_distances(d, r1, r2, rng)
    if r1 <= 0 and r2 <= 0:
        return 1.0 if rng >= d else 0.0
    if r1 <= 0:
        return circle_circle_intersection(r2, rng / 2, d2) / (math.pi * r2 * r2) if rng / 2 >= d1 else 0.0
    if r2 <= 0:
        return circle_circle_intersection(r1, rng / 2, d1) / (math.pi * r1 * r1) if rng / 2 >= d2 else 0.0
    area1 = circle_circle_intersection(r1, rng / 2, d1)
    area2 = circle_circle_intersection(r2, rng / 2, d2)
    return (area1 + area2) / (math.pi * r1 * r1 + math.pi * r2 * r2)


def segment_segment_factor(d: float, r1: float, r2: float, rng: float) -> float:
    """`MDMLib::SegmentSegmentFactor` (MDMLib.cc 210-274), including its `r1 == 0 && r1 == 0` test."""
    d1, d2 = _split_distances(d, r1, r2, rng)
    overlap1 = overlap2 = 0.0
    if (d1 - rng / 2) < r1:
        overlap1 = rng / 2 - (d1 - r1) if (d1 - rng / 2) > -r1 else 2 * r1
    if (d2 - rng / 2) < r2:
        overlap2 = rng / 2 - (d2 - r2) if (d2 - rng / 2) > -r2 else 2 * r2
    if r1 == 0:
        return 0.0 if d > rng else 1.0
    return ieee_div(overlap1 + overlap2, 2 * r1 + 2 * r2)


def one_sided_circle_segment_factor(d: float, r1: float, r2: float, rng: float) -> float:
    """`MDMLib::OneSidedCircleSegmentFactor` (MDMLib.cc 371-443)."""
    if d < 0:
        return 1.0
    if rng > d + r1 + r2:
        return 1.0
    if rng < d - r1 - r2:
        return 0.0
    d1, d2 = _split_distances(d, r1, r2, rng)
    if r1 <= 0 and r2 <= 0:
        return 1.0 if rng >= d else 0.0
    if r1 <= 0:
        return circle_segment(r2, rng / 2 - (d2 - r2)) / (math.pi * r2 * r2) if rng / 2 >= d1 else 0.0
    if r2 <= 0:
        return circle_segment(r1, rng / 2 - (d1 - r1)) / (math.pi * r1 * r1) if rng / 2 >= d2 else 0.0
    area1 = circle_segment(r1, rng / 2 - (d1 - r1))
    area2 = circle_segment(r2, rng / 2 - (d2 - r2))
    return (area1 + area2) / (math.pi * r1 * r1 + math.pi * r2 * r2)


def gaussian_sum(x: float, sig: float) -> float:
    """`MDMLib::gaussianSum`: normal CDF of x / sig (Abramowitz-Stegun 7.1.26)."""
    x = ieee_div(x, sig)
    sign = -1 if x < 0 else 1
    x = abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592
    if math.isnan(x):
        return math.nan
    y = 1.0 - poly * t * math.exp(-x * x) if math.isfinite(x) else 1.0
    return 0.5 * (1.0 + sign * y)


def importance_factor(r1: float, r2: float, d: float) -> float:
    """`MDMLib::importanceFactor` (MDMLib.cc 521-592)."""
    overlap = r2 + r1 - d
    if overlap < 0:
        return 0.0
    if r2 > r1:
        if overlap < 2 * r1:
            s1, e1, s2, e2 = -r1, -r1 + overlap, -r2, -r2 + overlap
        else:
            s1, e1, s2, e2 = -r1, r1, -(d + r1), -(d - r1)
    elif overlap < 2 * r2:
        s2, e2, s1, e1 = -r2, -r2 + overlap, -r1, -r1 + overlap
    else:
        s2, e2, s1, e1 = -r2, r2, -(d + r2), -(d - r2)
    sig1, sig2 = r1 / 3, r2 / 3
    if s1 == -r1:
        f1 = 1.0 if e1 == r1 else gaussian_sum(e1, sig1)
    else:
        f1 = gaussian_sum(e1, sig1) - gaussian_sum(s1, sig1)
    if s2 == -r2:
        f2 = 1.0 if e2 == r2 else gaussian_sum(e2, sig2)
    else:
        f2 = gaussian_sum(e2, sig2) - gaussian_sum(s2, sig2)
    f1 = ieee_div(f1, ieee_div(e1 - s1, 2 * r1))
    f2 = ieee_div(f2, ieee_div(e2 - s2, 2 * r2))
    return ieee_div(ieee_div(2 * (f1 * f2), f1 + f2), d)


class EllipseOverlap:
    """Overlap area of two ellipses, estimated on a fixed, evenly spread point set inside the first ellipse.

    F2MD (`ellipseIntLib`) solves the intersection exactly; this estimate uses `n_points` points of a Fermat spiral
    in the unit disk mapped onto ellipse 1 (relative error of order 1 / n_points^(3/4) for smooth boundaries).
    """

    def __init__(self, n_points: int) -> None:
        k = np.arange(n_points) + 0.5
        radius = np.sqrt(k / n_points)
        angle = k * math.pi * (3.0 - math.sqrt(5.0))
        self._unit = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)

    def area(self, c1, a1: float, b1: float, angle1: float, c2, a2: float, b2: float, angle2: float) -> float:
        """Overlap of ellipse 1 (centre c1, semi-axes a1 along x / b1 along y, rotated by angle1 rad) and ellipse 2."""
        if a1 <= 0 or b1 <= 0 or a2 <= 0 or b2 <= 0:
            return 0.0
        cos1, sin1 = math.cos(angle1), math.sin(angle1)
        local = self._unit * np.array([a1, b1])
        px = c1[0] + local[:, 0] * cos1 - local[:, 1] * sin1 - c2[0]
        py = c1[1] + local[:, 0] * sin1 + local[:, 1] * cos1 - c2[1]
        cos2, sin2 = math.cos(angle2), math.sin(angle2)
        u = (px * cos2 + py * sin2) / a2
        v = (-px * sin2 + py * cos2) / b2
        inside = np.count_nonzero(u * u + v * v <= 1.0)
        return math.pi * a1 * b1 * inside / len(self._unit)


class CatchRangePlausibility(Check):
    """C1: CircleCircleFactor(d, conf_sender, conf_receiver, MAX_PLAUSIBLE_RANGE)."""

    names = ("range_plausibility",)
    needs_history = False

    def evaluate(self, ctx: BsmContext) -> tuple[float, ...] | None:
        s, i = ctx.stream, ctx.i
        if not np.isfinite(s.rx_pos[i, 0]):
            return None
        d = math.hypot(*(s.pos[i] - s.rx_pos[i]))
        conf_s, conf_r = float(s.pos_conf[i, 0]), float(s.rx_pos_conf[i, 0])
        return (circle_circle_factor(d, conf_s, conf_r, self._config.max_plausible_range),)


class CatchSpeedPlausibility(Check):
    """C2: the claimed speed +- half its confidence against MAX_PLAUSIBLE_SPEED."""

    names = ("speed_plausibility",)
    needs_history = False

    def evaluate(self, ctx: BsmContext) -> tuple[float, ...] | None:
        speed = abs(math.hypot(*ctx.stream.spd[ctx.i]))
        conf = abs(math.hypot(*ctx.stream.spd_conf[ctx.i]))
        vmax = self._config.max_plausible_speed
        if speed + conf / 2 < vmax:
            return (1.0,)
        if speed - conf / 2 > vmax:
            return (0.0,)
        return (ieee_div(conf / 2 + (vmax - speed), conf),)


class CatchIntersection(legacy.Intersection):
    """C6: ellipses (vehicle size + 2 x position confidence) overlapping; 1.01 - factor x time weight, clipped."""

    def __init__(self, config: PlausibilityConfig) -> None:
        super().__init__(config)
        self._overlap = EllipseOverlap(config.ellipse_vertices)

    def _radius(self, conf: float) -> float:
        cfg = self._config
        return max(cfg.vehicle_width + 2 * conf, cfg.vehicle_length + 2 * conf) / 2

    def _radius_array(self, conf: np.ndarray) -> np.ndarray:
        cfg = self._config
        return np.maximum(cfg.vehicle_width + 2 * conf, cfg.vehicle_length + 2 * conf) / 2

    def _value(self, c1, conf1: float, a1: float, c2, conf2: float, a2: float, dt: float) -> float:
        cfg = self._config
        factor = self._factor(c1, conf1, a1, c2, conf2, a2)
        value = 1.01 - factor * ((cfg.max_delta_inter - dt) / cfg.max_delta_inter)
        return min(max(value, 0.0), 1.0)

    def _factor(self, c1, conf1: float, h1: float, c2, conf2: float, h2: float) -> float:
        """`MDMLib::EllipseEllipseIntersectionFactor` (MDMLib.cc 701-774); both ellipses use heading2 (as F2MD)."""
        cfg = self._config
        dx1, dy1 = cfg.vehicle_width + conf1 * 2, cfg.vehicle_length + conf1 * 2
        dx2, dy2 = cfg.vehicle_width + conf2 * 2, cfg.vehicle_length + conf2 * 2
        h1 = int(h1 / 0.01) * 0.01
        h2 = int(h2 / 0.01) * 0.01
        rot = math.radians(h2)
        area = self._overlap.area(c1, dx1 / 2, dy1 / 2, rot, c2, dx2 / 2, dy2 / 2, rot)
        area_factor = area / (math.pi * (dx1 / 2) * (dy1 / 2) + math.pi * (dx2 / 2) * (dy2 / 2) - area)
        distance = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
        centres = math.radians(heading_angle(c2[0] - c1[0], c2[1] - c1[1]))
        diffs = []
        for heading in (math.radians(h1), math.radians(h2)):
            diff = math.atan2(math.sin(heading - centres), math.cos(heading - centres))
            if diff > math.pi / 2:
                diff -= math.pi
            if diff < -math.pi / 2:
                diff += math.pi
            diffs.append(diff)
        imp1 = (dx1 / 2) * math.sin(diffs[0]) + (dy1 / 2) * math.cos(diffs[0])
        imp2 = (dx2 / 2) * math.sin(diffs[1]) + (dy2 / 2) * math.cos(diffs[1])
        factor1 = importance_factor(imp1, imp2, distance) * area_factor
        return factor1 if factor1 < area_factor else area_factor


class CatchPositionConsistency(Check):
    """C3: CircleCircleFactor(distance, conf_cur, conf_old, MAX_PLAUSIBLE_SPEED * Δt)."""

    names = ("position_consistency",)

    def evaluate(self, ctx: BsmContext) -> tuple[float, ...] | None:
        s, i, j = ctx.stream, ctx.i, ctx.prev
        d = math.hypot(*(s.pos[i] - s.pos[j]))
        rng = self._config.max_plausible_speed * ctx.dt
        return (circle_circle_factor(d, float(s.pos_conf[i, 0]), float(s.pos_conf[j, 0]), rng),)


class CatchSpeedConsistency(Check):
    """C4: SegmentSegmentFactor(|Δspeed|, conf_cur, conf_old, ACCEL or DECEL * Δt)."""

    names = ("speed_consistency",)

    def evaluate(self, ctx: BsmContext) -> tuple[float, ...] | None:
        cfg, s, i, j = self._config, ctx.stream, ctx.i, ctx.prev
        delta = math.hypot(*s.spd[i]) - math.hypot(*s.spd[j])
        c_cur, c_old = math.hypot(*s.spd_conf[i]), math.hypot(*s.spd_conf[j])
        limit = (cfg.max_plausible_accel if delta > 0 else cfg.max_plausible_decel) * ctx.dt
        return (segment_segment_factor(abs(delta), c_cur, c_old, limit),)


class CatchPositionHeadingConsistency(Check):
    """C7: the heading interval (angle +- heading confidence) through the two position-confidence circles."""

    names = ("position_heading_consistency",)

    def evaluate(self, ctx: BsmContext) -> tuple[float, ...] | None:
        cfg, s, i, j = self._config, ctx.stream, ctx.i, ctx.prev
        if not ctx.dt < cfg.pos_heading_time:
            return None
        move = s.pos[i] - s.pos[j]
        distance = math.hypot(*move)
        if distance < 1:
            return (1.0,)
        if math.hypot(*s.spd[i]) - math.hypot(*s.spd_conf[i]) < 1:
            return (1.0,)
        delta = abs(
            heading_angle(float(s.hed[i, 0]), float(s.hed[i, 1])) - heading_angle(float(move[0]), float(move[1]))
        )
        if delta > 180:
            delta = 360 - delta
        conf_h = float(s.hed_conf[i, 0])
        low, high = max(delta - conf_h, 0.0), min(delta + conf_h, 180.0)
        cur, old = float(s.pos_conf[i, 0]), float(s.pos_conf[j, 0])
        parts = []
        for angle in (low, high):
            x = distance * math.cos(math.radians(angle))
            ok = 1.0 if angle <= cfg.max_heading_change else 0.0
            parts.append(ok if cur == 0 else circle_segment(cur, cur + x) / (math.pi * cur * cur))
            parts.append(ok if old == 0 else 1 - circle_segment(old, old - x) / (math.pi * old * old))
        return (sum(parts) / 4,)


class CatchPositionSpeedConsistency(Check):
    """C5: mean over three speed hypotheses of the min / max reachable-distance factors; min of the two means."""

    names = ("position_speed_consistency",)

    def evaluate(self, ctx: BsmContext) -> tuple[float, ...] | None:
        cfg, s, i, j = self._config, ctx.stream, ctx.i, ctx.prev
        if not ctx.dt < cfg.max_time_delta:
            return None
        d = math.hypot(*(s.pos[i] - s.pos[j]))
        cur, old = math.hypot(*s.spd[i]), math.hypot(*s.spd[j])
        c_cur, c_old = math.hypot(*s.spd_conf[i]), math.hypot(*s.spd_conf[j])
        p_cur, p_old = float(s.pos_conf[i, 0]), float(s.pos_conf[j, 0])
        cur1, old1 = cur + c_cur, old - c_old
        cur2, old2 = cur - c_cur, old + c_old
        if cur2 < old2:
            cur2 = old2 = (cur + old) / 2
        v_min = min(cur, old)
        addon = max(0.0, cfg.max_mgt_rng_down + 0.3571 * v_min - 0.01694 * v_min * v_min)
        limits = (cfg.max_plausible_accel, cfg.max_plausible_decel, cfg.max_plausible_speed)
        min1, max1 = max_min_dist(cur1, old1, ctx.dt, *limits)
        min2, max2 = max_min_dist(cur2, old2, ctx.dt, *limits)
        min0, max0 = max_min_dist(cur, old, ctx.dt, *limits)
        up = cfg.max_mgt_rng_up
        f_min = (
            (1 - circle_circle_factor(d, p_cur, p_old, min1))
            + (1 - one_sided_circle_segment_factor(d, p_cur, p_old, min0 - addon))
            + (1 - one_sided_circle_segment_factor(d, p_cur, p_old, min2 - addon))
        ) / 3.0
        f_max = (
            one_sided_circle_segment_factor(d, p_cur, p_old, max1 + up)
            + one_sided_circle_segment_factor(d, p_cur, p_old, max0 + up)
            + one_sided_circle_segment_factor(d, p_cur, p_old, max2 + up)
        ) / 3.0
        return (min(f_min, f_max),)


class CatchPositionSpeedMaxConsistency(Check):
    """C5b: OneSidedCircleSegmentFactor of the implied-speed excess with radii conf_pos / Δt + conf_speed."""

    names = ("position_speed_max_consistency",)

    def evaluate(self, ctx: BsmContext) -> tuple[float, ...] | None:
        cfg, s, i, j = self._config, ctx.stream, ctx.i, ctx.prev
        if not ctx.dt < cfg.max_time_delta:
            return None
        implied = ieee_div(math.hypot(*(s.pos[i] - s.pos[j])), ctx.dt)
        cur, old = math.hypot(*s.spd[i]), math.hypot(*s.spd[j])
        cur_r = ieee_div(float(s.pos_conf[i, 0]), ctx.dt) + math.hypot(*s.spd_conf[i])
        old_r = ieee_div(float(s.pos_conf[j, 0]), ctx.dt) + math.hypot(*s.spd_conf[j])
        max_f = one_sided_circle_segment_factor(
            max(cur, old) - implied, cur_r, old_r, (cfg.max_plausible_decel + cfg.max_mgt_rng) * ctx.dt
        )
        min_f = one_sided_circle_segment_factor(
            implied - min(cur, old), cur_r, old_r, (cfg.max_plausible_accel + cfg.max_mgt_rng) * ctx.dt
        )
        return (min_f if min_f < max_f else max_f,)


class CatchSuddenAppearance(Check):
    """C9 (diagnostic): 1 - share of the sender's confidence circle inside MAX_SA_RANGE + receiver confidence."""

    names = ("sudden_appearance",)
    needs_history = False

    def evaluate(self, ctx: BsmContext) -> tuple[float, ...] | None:
        cfg, s, i = self._config, ctx.stream, ctx.i
        if ctx.entry is not None and not ctx.dt > cfg.max_sa_time:
            return None
        if not np.isfinite(s.rx_pos[i, 0]):
            return None
        d = math.hypot(*(s.pos[i] - s.rx_pos[i]))
        r2 = cfg.max_sa_range + float(s.rx_pos_conf[i, 0])
        conf = float(s.pos_conf[i, 0])
        if conf <= 0:
            return (0.0 if d < r2 else 1.0,)
        return (1 - circle_circle_intersection(conf, r2, d) / (math.pi * conf * conf),)


# CaTChChecks::CheckBSM order: C1, C2, C6; with history C3, C4, C8, C7, C5, C5b, C9, C11-C15.
CATCH_CHECK_CLASSES: tuple[type[Check], ...] = (
    CatchRangePlausibility,
    CatchSpeedPlausibility,
    CatchIntersection,
    CatchPositionConsistency,
    CatchSpeedConsistency,
    legacy.BeaconFrequency,
    CatchPositionHeadingConsistency,
    CatchPositionSpeedConsistency,
    CatchPositionSpeedMaxConsistency,
    CatchSuddenAppearance,
    legacy.KalmanPositionSpeed,
    legacy.KalmanPositionSpeedScalar,
    legacy.KalmanPosition,
    legacy.KalmanPositionAcc,
    legacy.KalmanSpeed,
)
