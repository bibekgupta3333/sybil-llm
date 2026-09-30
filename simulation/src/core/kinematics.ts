import type { Identity } from "./identity";

/** Thresholds for the plausibility flags, in SI units. */
export const TELEPORT_SPEED_MPS = 60; // no urban vehicle moves this fast; a forged jump does
export const INCONSISTENT_RESIDUAL_MPS = 5; // ‖Δpos/Δt − spd‖ beyond ordinary GPS jitter

/** Bit flags on `KinematicsSeries.flags`. */
export const enum StepFlag {
  None = 0,
  /** `dt <= 0`: no previous step inside this identity (or a same-timestamp collision). */
  UndefinedDt = 1 << 0,
  /** Implied speed from Δpos/Δt exceeds `TELEPORT_SPEED_MPS`. */
  Teleport = 1 << 1,
  /** Claimed velocity disagrees with observed displacement beyond `INCONSISTENT_RESIDUAL_MPS`. */
  Inconsistent = 1 << 2,
}

/**
 * Derived per-step signals, computed once per identity so renderers never do math per frame.
 *
 * Everything here is what a label-free physics-consistency detector could compute from the 13
 * broadcast features alone — no class information is used.
 */
export class KinematicsSeries {
  private constructor(
    /** ‖spd‖ (m/s). */
    readonly speed: Float32Array,
    /** ‖acl‖ (m/s²). */
    readonly accel: Float32Array,
    /** ‖Δpos‖ / Δt (m/s); NaN where dt <= 0. */
    readonly impliedSpeed: Float32Array,
    /** ‖Δpos/Δt − spd‖ (m/s); NaN where dt <= 0. */
    readonly residual: Float32Array,
    /** Signed angle from heading vector to velocity vector, radians in (−π, π]; NaN if speed ≈ 0. */
    readonly headingAngle: Float32Array,
    readonly flags: Uint8Array,
  ) {}

  static compute(identity: Identity): KinematicsSeries {
    const n = identity.stepCount;
    const speed = new Float32Array(n);
    const accel = new Float32Array(n);
    const implied = new Float32Array(n);
    const residual = new Float32Array(n);
    const angle = new Float32Array(n);
    const flags = new Uint8Array(n);

    for (let i = 0; i < n; i++) {
      const sx = identity.spdX(i);
      const sy = identity.spdY(i);
      speed[i] = Math.hypot(sx, sy);
      accel[i] = Math.hypot(identity.aclX(i), identity.aclY(i));
      angle[i] = KinematicsSeries.signedAngle(identity.hedX(i), identity.hedY(i), sx, sy);

      const dt = identity.dt(i);
      let flag = StepFlag.None;
      if (dt <= 0) {
        implied[i] = NaN;
        residual[i] = NaN;
        flag |= StepFlag.UndefinedDt;
      } else {
        const vx = identity.dposX(i) / dt;
        const vy = identity.dposY(i) / dt;
        implied[i] = Math.hypot(vx, vy);
        residual[i] = Math.hypot(vx - sx, vy - sy);
        if (implied[i] > TELEPORT_SPEED_MPS) flag |= StepFlag.Teleport;
        if (residual[i] > INCONSISTENT_RESIDUAL_MPS) flag |= StepFlag.Inconsistent;
      }
      flags[i] = flag;
    }
    return new KinematicsSeries(speed, accel, implied, residual, angle, flags);
  }

  has(step: number, flag: StepFlag): boolean {
    return (this.flags[step] & flag) !== 0;
  }

  /** Fraction of steps (with defined dt) carrying a given flag. */
  fraction(flag: StepFlag): number {
    let defined = 0;
    let hit = 0;
    for (let i = 0; i < this.flags.length; i++) {
      if (this.flags[i] & StepFlag.UndefinedDt) continue;
      defined++;
      if (this.flags[i] & flag) hit++;
    }
    return defined === 0 ? 0 : hit / defined;
  }

  max(series: Float32Array): number {
    let m = -Infinity;
    for (let i = 0; i < series.length; i++) if (series[i] > m) m = series[i];
    return m === -Infinity ? 0 : m;
  }

  /** Wrap-safe signed angle between two vectors via atan2(cross, dot). */
  static signedAngle(ax: number, ay: number, bx: number, by: number): number {
    if (Math.hypot(ax, ay) < 1e-9 || Math.hypot(bx, by) < 1e-3) return NaN;
    return Math.atan2(ax * by - ay * bx, ax * bx + ay * by);
  }
}
