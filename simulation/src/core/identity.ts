import type { BBox, FeatureIndex, IdentityRecord } from "./manifest";

/**
 * One broadcast identity (a `senderPseudo` within one run) and its stitched step sequence.
 *
 * Steps are a zero-copy view into the dataset's Float32Array (row-major, stride = feature count);
 * no per-step objects are ever allocated.
 */
export class Identity {
  constructor(
    readonly record: IdentityRecord,
    /** `n_steps * featureCount` values, row-major. */
    readonly steps: Float32Array,
    /** Absolute simulation seconds, one per step. */
    readonly times: Float64Array,
    readonly features: FeatureIndex,
    readonly stride: number,
    readonly windowSize: number,
  ) {
    if (steps.length !== record.n_steps * features.count) {
      throw new Error(`identity ${record.id}: steps view has ${steps.length} values, expected ${record.n_steps * features.count}`);
    }
    if (times.length !== record.n_steps) {
      throw new Error(`identity ${record.id}: times view has ${times.length} values, expected ${record.n_steps}`);
    }
  }

  get id(): number {
    return this.record.id;
  }

  get label(): string {
    return this.record.label;
  }

  get stepCount(): number {
    return this.record.n_steps;
  }

  /** Number of stride-10 windows the pipeline cut from this identity. */
  get windowCount(): number {
    return Math.floor((this.stepCount - this.windowSize) / this.stride) + 1;
  }

  /** Key shared by all pseudonyms of one physical vehicle within one run. */
  get senderKey(): string {
    return `${this.record.subfolder}/${this.record.sender ?? "?"}`;
  }

  get isAttacker(): boolean {
    return this.record.label !== "Benign";
  }

  /** "prepared" (from X_windows.npy) or "raw_veremi" (a fabricated broadcast, never in training data). */
  get dataSource(): "prepared" | "raw_veremi" {
    return this.record.data_source;
  }

  /** The id of this identity's real/fabricated counterpart for the same physical vehicle, if any. */
  get pairedId(): number | null {
    return this.record.paired_identity_id;
  }

  hasTag(tag: string): boolean {
    return this.record.tags.includes(tag);
  }

  value(step: number, feature: number): number {
    return this.steps[step * this.features.count + feature];
  }

  posX(step: number): number {
    return this.value(step, this.features.posX);
  }
  posY(step: number): number {
    return this.value(step, this.features.posY);
  }
  spdX(step: number): number {
    return this.value(step, this.features.spdX);
  }
  spdY(step: number): number {
    return this.value(step, this.features.spdY);
  }
  aclX(step: number): number {
    return this.value(step, this.features.aclX);
  }
  aclY(step: number): number {
    return this.value(step, this.features.aclY);
  }
  hedX(step: number): number {
    return this.value(step, this.features.hedX);
  }
  hedY(step: number): number {
    return this.value(step, this.features.hedY);
  }
  dt(step: number): number {
    return this.value(step, this.features.dt);
  }
  dposX(step: number): number {
    return this.value(step, this.features.dposX);
  }
  dposY(step: number): number {
    return this.value(step, this.features.dposY);
  }

  /** First window index whose 20-step span contains `step` (the window the model would see it in). */
  windowIndexOf(step: number): number {
    const k = Math.floor(step / this.stride);
    return Math.max(0, Math.min(k, this.windowCount - 1));
  }

  /**
   * A frozen, single-window view of this identity: just the 20 steps of window `k`, as an
   * `Identity` of its own (windowCount 1). Lets any renderer built for a whole identity (heatmap,
   * strips, consistency panel) draw one specific window with no other code path.
   */
  windowSlice(k: number): Identity {
    if (!Number.isInteger(k) || k < 0 || k >= this.windowCount) {
      throw new RangeError(`window ${k} out of range [0, ${this.windowCount})`);
    }
    const start = k * this.stride;
    const f = this.features.count;
    const steps = this.steps.subarray(start * f, (start + this.windowSize) * f);
    const times = this.times.subarray(start, start + this.windowSize);
    const record: IdentityRecord = {
      ...this.record,
      row_offset: this.record.row_offset + start,
      n_steps: this.windowSize,
      n_windows: 1,
      x_windows_rows: [this.record.x_windows_rows[k]],
      t_start: times[0],
      t_end: times[times.length - 1],
    };
    return new Identity(record, steps, times, this.features, this.stride, this.windowSize);
  }

  /**
   * The steps with `t0 <= time <= t1`, as an `Identity` of its own. The start is snapped forward to
   * a window boundary, so the slice's windows are exactly this identity's windows (and its
   * `x_windows_rows` stay truthful). Returns `this` unchanged if fewer than one window would remain.
   */
  timeSlice(t0: number, t1: number): Identity {
    let first = 0;
    while (first < this.stepCount && this.times[first] < t0) first++;
    first = Math.ceil(first / this.stride) * this.stride;
    let last = this.stepCount - 1;
    while (last >= 0 && this.times[last] > t1) last--;
    const n = last - first + 1;
    if (n < this.windowSize) return this;
    const f = this.features.count;
    const times = this.times.subarray(first, first + n);
    const nWindows = Math.floor((n - this.windowSize) / this.stride) + 1;
    const k0 = first / this.stride;
    const rows = this.record.x_windows_rows;
    const record: IdentityRecord = {
      ...this.record,
      row_offset: this.record.row_offset + first,
      n_steps: n,
      n_windows: nWindows,
      x_windows_rows: rows.length > 1 ? rows.slice(k0, k0 + nWindows) : rows,
      t_start: times[0],
      t_end: times[n - 1],
    };
    return new Identity(record, this.steps.subarray(first * f, (first + n) * f), times, this.features, this.stride, this.windowSize);
  }

  /** Position bounding box over all steps. */
  bbox(): BBox {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (let i = 0; i < this.stepCount; i++) {
      const x = this.posX(i);
      const y = this.posY(i);
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    return { minX, minY, maxX, maxY };
  }
}
