import type { Identity } from "../core/identity";
import { WindowView, type NormStats } from "../core/window_view";
import { THEME } from "../ui/theme";
import type { Layer } from "./layer";

/**
 * The 20×13 window tensor the model consumes, drawn as features (rows) × timesteps (columns).
 * Raw mode scales each feature row to its own min/max; normalized mode uses a diverging scale
 * around 0 in units of train-split standard deviations. Re-rendered only when the window or mode changes.
 */
export class HeatmapRenderer implements Layer {
  private view: WindowView | null = null;
  private identity: Identity | null = null;
  private normalized = true;
  private windowIndex = -1;
  private stepInWindow = 0;
  private cache: { key: string; values: Float32Array; lo: Float32Array; hi: Float32Array } | null = null;
  private readonly labelGutter = 58;
  private readonly topGutter = 16;

  constructor(private readonly norm: NormStats) {}

  setIdentity(identity: Identity): void {
    this.identity = identity;
    this.view = new WindowView(identity, this.norm);
    this.cache = null;
    this.windowIndex = -1;
  }

  setNormalized(on: boolean): void {
    this.normalized = on;
    this.cache = null;
  }

  get isNormalized(): boolean {
    return this.normalized;
  }

  /** Point the heatmap at the window containing `step`; returns true if the window changed. */
  setStep(step: number): boolean {
    if (!this.identity) return false;
    const k = this.identity.windowIndexOf(step);
    this.stepInWindow = step - k * this.identity.stride;
    if (k === this.windowIndex) return false;
    this.windowIndex = k;
    return true;
  }

  get currentWindow(): number {
    return this.windowIndex;
  }

  drawStatic(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    ctx.fillStyle = THEME.panel;
    ctx.fillRect(0, 0, width, height);
    if (!this.view || !this.identity || this.windowIndex < 0) return;
    const f = this.view.featureCount;
    const T = this.view.windowSize;
    const data = this.values();
    const cellW = (width - this.labelGutter - 8) / T;
    const cellH = (height - this.topGutter - 4) / f;
    ctx.font = "10px Inter, system-ui, sans-serif";
    for (let j = 0; j < f; j++) {
      ctx.fillStyle = THEME.muted;
      ctx.fillText(this.identity.features.names[j], 4, this.topGutter + j * cellH + cellH / 2 + 4);
      for (let t = 0; t < T; t++) {
        const v = data.values[t * f + j];
        ctx.fillStyle = this.normalized ? diverging(v / 3) : sequential((v - data.lo[j]) / Math.max(1e-9, data.hi[j] - data.lo[j]));
        ctx.fillRect(this.labelGutter + t * cellW, this.topGutter + j * cellH, Math.ceil(cellW), Math.ceil(cellH));
      }
    }
    ctx.fillStyle = THEME.dim;
    for (let t = 0; t < T; t += 5) ctx.fillText(`t${t}`, this.labelGutter + t * cellW + 2, 11);
    ctx.fillText(this.normalized ? "z-score · blue −3σ · white 0 · red +3σ" : "raw · per-feature min→max", width - 190, 11);
  }

  drawDynamic(ctx: CanvasRenderingContext2D, width: number, height: number, _step: number): void {
    if (!this.view || this.windowIndex < 0) return;
    const T = this.view.windowSize;
    const cellW = (width - this.labelGutter - 8) / T;
    ctx.strokeStyle = THEME.textStrong;
    ctx.lineWidth = 2;
    ctx.strokeRect(this.labelGutter + this.stepInWindow * cellW, this.topGutter, cellW, height - this.topGutter - 4);
  }

  private values(): { values: Float32Array; lo: Float32Array; hi: Float32Array } {
    const key = `${this.windowIndex}:${this.normalized}`;
    if (this.cache && this.cache.key === key) return this.cache;
    const view = this.view!;
    const values = this.normalized ? view.normalized(this.windowIndex) : new Float32Array(view.raw(this.windowIndex));
    const f = view.featureCount;
    const lo = new Float32Array(f).fill(Infinity);
    const hi = new Float32Array(f).fill(-Infinity);
    for (let i = 0; i < values.length; i++) {
      const j = i % f;
      if (values[i] < lo[j]) lo[j] = values[i];
      if (values[i] > hi[j]) hi[j] = values[i];
    }
    this.cache = { key, values, lo, hi };
    return this.cache;
  }
}

/** −3σ → primary blue, 0 → light neutral, +3σ → negative red. */
function diverging(t: number): string {
  const x = Math.max(-1, Math.min(1, t));
  const center: readonly [number, number, number] = [240, 240, 240];
  const negative: readonly [number, number, number] = [37, 99, 235]; // primary blue
  const positive: readonly [number, number, number] = [220, 38, 38]; // negative red
  const [r0, g0, b0] = center;
  const [r1, g1, b1] = x < 0 ? negative : positive;
  const a = Math.abs(x);
  return `rgb(${Math.round(r0 + (r1 - r0) * a)},${Math.round(g0 + (g1 - g0) * a)},${Math.round(b0 + (b1 - b0) * a)})`;
}

/** 0 → primary-light, 1 → primary blue: a light single-hue sequential scale for raw values. */
function sequential(t: number): string {
  const x = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0));
  return `rgb(${Math.round(219 + (37 - 219) * x)},${Math.round(234 + (99 - 234) * x)},${Math.round(254 + (235 - 254) * x)})`;
}
