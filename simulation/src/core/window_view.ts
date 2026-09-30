import type { Identity } from "./identity";

/** Per-feature z-score parameters from `config.json` (train split only). */
export interface NormStats {
  readonly mean: readonly number[];
  readonly std: readonly number[];
}

/**
 * The model's-eye view of an identity: its stride-10 windows of `windowSize` steps.
 *
 * `raw(k)` is a zero-copy view; `normalized(k)` applies the train-split z-score exactly as
 * the training pipeline does, treating a zero std as "leave the value at 0".
 */
export class WindowView {
  constructor(
    private readonly identity: Identity,
    private readonly norm: NormStats,
  ) {
    if (norm.mean.length !== identity.features.count || norm.std.length !== identity.features.count) {
      throw new Error("normalization stats do not match the feature count");
    }
  }

  get count(): number {
    return this.identity.windowCount;
  }

  get windowSize(): number {
    return this.identity.windowSize;
  }

  get featureCount(): number {
    return this.identity.features.count;
  }

  /** Step index of the window's first row. */
  startStep(k: number): number {
    this.check(k);
    return k * this.identity.stride;
  }

  /** Raw `windowSize × featureCount` values, row-major, as a view into the identity's steps. */
  raw(k: number): Float32Array {
    const f = this.featureCount;
    const start = this.startStep(k) * f;
    return this.identity.steps.subarray(start, start + this.windowSize * f);
  }

  /** Z-scored copy of window `k`: `(x - mean) / std`, with `std == 0` mapped to 0. */
  normalized(k: number): Float32Array {
    const raw = this.raw(k);
    const out = new Float32Array(raw.length);
    const f = this.featureCount;
    for (let i = 0; i < raw.length; i++) {
      const j = i % f;
      const std = this.norm.std[j];
      out[i] = std === 0 ? 0 : (raw[i] - this.norm.mean[j]) / std;
    }
    return out;
  }

  /** Inverse of `normalized`, for round-trip checks and readouts. */
  denormalize(values: Float32Array): Float32Array {
    const out = new Float32Array(values.length);
    const f = this.featureCount;
    for (let i = 0; i < values.length; i++) {
      const j = i % f;
      out[i] = values[i] * this.norm.std[j] + this.norm.mean[j];
    }
    return out;
  }

  private check(k: number): void {
    if (!Number.isInteger(k) || k < 0 || k >= this.count) {
      throw new RangeError(`window ${k} out of range [0, ${this.count})`);
    }
  }
}
