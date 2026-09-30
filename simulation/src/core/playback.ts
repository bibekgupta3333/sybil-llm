/** Injectable clock/frame hooks so the loop is testable without a browser. */
export interface PlaybackClock {
  readonly now: () => number; // milliseconds
  readonly requestFrame: (cb: (nowMs: number) => void) => number;
  readonly cancelFrame: (handle: number) => void;
}

export type FrameListener = (timeSec: number, stepIndex: number) => void;

const DEFAULT_MAX_GAP_SEC = 2;

/**
 * The single animation clock. Runs in simulation seconds, owns the one `requestAnimationFrame`
 * loop, and exposes `stepIndex` via binary search so renderers never scan.
 *
 * "Compress gaps" replaces any inter-step gap longer than `maxGapSec` with `maxGapSec` on a
 * virtual timeline, so a 101 s beacon dropout does not freeze the replay.
 */
export class Playback {
  private times: Float64Array = new Float64Array(0);
  private virtual: Float64Array = new Float64Array(0);
  private cursor = 0; // position on the active timeline, seconds
  /** Exact step chosen by a discrete seek; disambiguates duplicate timestamps until the clock moves. */
  private pinnedStep: number | null = null;
  private speedMultiplier = 1;
  private compress = false;
  private running = false;
  private frameHandle: number | null = null;
  private lastNowMs = 0;
  private readonly listeners = new Set<FrameListener>();

  constructor(private readonly clock: PlaybackClock = Playback.browserClock()) {}

  static browserClock(): PlaybackClock {
    return {
      now: () => performance.now(),
      requestFrame: (cb) => requestAnimationFrame(cb),
      cancelFrame: (h) => cancelAnimationFrame(h),
    };
  }

  /** Load a new identity's timestamps; resets the cursor to the start. */
  setTimes(times: Float64Array, maxGapSec = DEFAULT_MAX_GAP_SEC): void {
    this.times = times;
    this.virtual = Playback.compressGaps(times, maxGapSec);
    this.cursor = this.timeline[0] ?? 0;
    this.pinnedStep = times.length ? 0 : null;
    this.emit();
  }

  get playing(): boolean {
    return this.running;
  }

  get speed(): number {
    return this.speedMultiplier;
  }

  get compressGapsEnabled(): boolean {
    return this.compress;
  }

  /** Current time on the active timeline (simulation seconds when gaps are not compressed). */
  get time(): number {
    return this.cursor;
  }

  /** Absolute simulation time of the current step. */
  get simulationTime(): number {
    return this.times[this.stepIndex] ?? 0;
  }

  get duration(): number {
    const t = this.timeline;
    return t.length === 0 ? 0 : t[t.length - 1] - t[0];
  }

  /** 0..1 along the active timeline. */
  get progress(): number {
    const d = this.duration;
    return d <= 0 ? 0 : (this.cursor - this.timeline[0]) / d;
  }

  get stepCount(): number {
    return this.times.length;
  }

  /** Index of the last step whose time is <= the cursor (binary search; ties resolve to the last duplicate). */
  get stepIndex(): number {
    if (this.pinnedStep !== null) return this.pinnedStep;
    return Playback.stepIndexAt(this.timeline, this.cursor);
  }

  onFrame(listener: FrameListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  play(): void {
    if (this.running || this.times.length === 0) return;
    if (this.progress >= 1) this.seekToStep(0);
    this.running = true;
    this.lastNowMs = this.clock.now();
    this.frameHandle = this.clock.requestFrame(this.tick);
  }

  pause(): void {
    this.running = false;
    if (this.frameHandle !== null) this.clock.cancelFrame(this.frameHandle);
    this.frameHandle = null;
  }

  toggle(): void {
    if (this.running) this.pause();
    else this.play();
  }

  setSpeed(multiplier: number): void {
    this.speedMultiplier = Math.min(8, Math.max(0.25, multiplier));
  }

  setCompressGaps(enabled: boolean): void {
    if (enabled === this.compress) return;
    const step = this.stepIndex;
    this.compress = enabled;
    this.seekToStep(step);
  }

  seekToStep(step: number): void {
    if (this.times.length === 0) return;
    const i = Math.max(0, Math.min(this.times.length - 1, Math.floor(step)));
    this.cursor = this.timeline[i];
    this.pinnedStep = i;
    this.emit();
  }

  seekToProgress(p: number): void {
    const t = this.timeline;
    if (t.length === 0) return;
    this.cursor = t[0] + Math.min(1, Math.max(0, p)) * this.duration;
    this.pinnedStep = null;
    this.emit();
  }

  stepBy(delta: number): void {
    this.pause();
    this.seekToStep(this.stepIndex + delta);
  }

  /** Advance the clock by real milliseconds; exposed for tests, used by the frame loop. */
  advance(elapsedMs: number): void {
    const t = this.timeline;
    if (t.length === 0) return;
    const end = t[t.length - 1];
    this.cursor = Math.min(end, this.cursor + (elapsedMs / 1000) * this.speedMultiplier);
    this.pinnedStep = null;
    if (this.cursor >= end) this.pause();
    this.emit();
  }

  static stepIndexAt(times: Float64Array, t: number): number {
    if (times.length === 0) return 0;
    let lo = 0;
    let hi = times.length - 1;
    if (t >= times[hi]) return hi;
    if (t < times[0]) return 0;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (times[mid] <= t) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }

  static compressGaps(times: Float64Array, maxGapSec: number): Float64Array {
    const out = new Float64Array(times.length);
    if (times.length === 0) return out;
    out[0] = times[0];
    for (let i = 1; i < times.length; i++) {
      out[i] = out[i - 1] + Math.min(maxGapSec, Math.max(0, times[i] - times[i - 1]));
    }
    return out;
  }

  private get timeline(): Float64Array {
    return this.compress ? this.virtual : this.times;
  }

  private readonly tick = (nowMs: number): void => {
    if (!this.running) return;
    this.advance(nowMs - this.lastNowMs);
    this.lastNowMs = nowMs;
    if (this.running) this.frameHandle = this.clock.requestFrame(this.tick);
  };

  private emit(): void {
    const step = this.stepIndex;
    for (const l of this.listeners) l(this.cursor, step);
  }
}
