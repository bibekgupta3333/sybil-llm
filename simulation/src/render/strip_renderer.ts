import type { Identity } from "../core/identity";
import { KinematicsSeries, StepFlag } from "../core/kinematics";
import { THEME, withAlpha } from "../ui/theme";
import type { Layer } from "./layer";

interface Strip {
  readonly name: string;
  readonly unit: string;
  readonly color: string;
  readonly values: Float32Array;
  readonly max: number;
}

/**
 * Stacked time-series strips on a shared step axis, with stride-10 window bands and a playhead.
 * The x-axis is the step index (not wall time) so bands line up with the model's windows.
 */
export class StripRenderer implements Layer {
  private identity: Identity | null = null;
  private strips: Strip[] = [];
  private kinematics: KinematicsSeries | null = null;
  private readonly leftGutter = 64;
  private readonly rightGutter = 12;
  private readonly topPad = 6;

  setIdentity(identity: Identity): void {
    this.identity = identity;
    const k = KinematicsSeries.compute(identity);
    this.kinematics = k;
    const dt = new Float32Array(identity.stepCount);
    const dpos = new Float32Array(identity.stepCount);
    for (let i = 0; i < identity.stepCount; i++) {
      dt[i] = identity.dt(i);
      dpos[i] = Math.hypot(identity.dposX(i), identity.dposY(i));
    }
    const cap = (arr: Float32Array, floor: number): number => Math.max(floor, robustMax(arr));
    this.strips = [
      { name: "speed ‖spd‖", unit: "m/s", color: THEME.blue, values: k.speed, max: cap(k.speed, 5) },
      { name: "accel ‖acl‖", unit: "m/s²", color: THEME.orange, values: k.accel, max: cap(k.accel, 1) },
      { name: "dt", unit: "s", color: THEME.amber, values: dt, max: cap(dt, 1) },
      { name: "‖Δpos‖", unit: "m", color: THEME.purple, values: dpos, max: cap(dpos, 5) },
      { name: "residual ‖Δpos/dt − spd‖", unit: "m/s", color: THEME.red, values: k.residual, max: cap(k.residual, 2) },
    ];
  }

  drawStatic(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    ctx.fillStyle = THEME.panel;
    ctx.fillRect(0, 0, width, height);
    if (!this.identity) return;
    const n = this.identity.stepCount;
    const plotW = width - this.leftGutter - this.rightGutter;
    const stripH = (height - this.topPad) / this.strips.length;
    // window bands: alternate shading per stride block so the stride-10 tiling reads at a glance
    for (let w = 0; w < this.identity.windowCount; w++) {
      const s0 = w * this.identity.stride;
      const x0 = this.leftGutter + (s0 / Math.max(1, n - 1)) * plotW;
      const x1 = this.leftGutter + (Math.min(n - 1, s0 + this.identity.stride) / Math.max(1, n - 1)) * plotW;
      ctx.fillStyle = w % 2 === 0 ? THEME.windowBand : THEME.windowBandAlt;
      ctx.fillRect(x0, this.topPad, x1 - x0, height - this.topPad);
      ctx.fillStyle = THEME.dim;
      ctx.font = "9px Inter, system-ui, sans-serif";
      if (plotW / this.identity.windowCount > 22) ctx.fillText(`w${w}`, x0 + 2, this.topPad + 9);
    }
    this.strips.forEach((strip, si) => {
      const top = this.topPad + si * stripH;
      const base = top + stripH - 4;
      ctx.strokeStyle = THEME.borderSoft;
      ctx.beginPath(); ctx.moveTo(this.leftGutter, base); ctx.lineTo(width - this.rightGutter, base); ctx.stroke();
      ctx.fillStyle = THEME.muted;
      ctx.font = "10px Inter, system-ui, sans-serif";
      ctx.fillText(strip.name, 4, top + 11);
      ctx.fillStyle = THEME.dim;
      ctx.fillText(`${fmt(strip.max)} ${strip.unit}`, 4, top + 22);
      ctx.strokeStyle = strip.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < n; i++) {
        const v = strip.values[i];
        if (!Number.isFinite(v)) { started = false; continue; }
        const x = this.leftGutter + (i / Math.max(1, n - 1)) * plotW;
        const y = base - Math.min(1, v / strip.max) * (stripH - 18);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      }
      ctx.stroke();
      // clipped-to-max markers (teleports blow past the robust max)
      ctx.fillStyle = THEME.red;
      for (let i = 0; i < n; i++) {
        if (strip.values[i] > strip.max) {
          const x = this.leftGutter + (i / Math.max(1, n - 1)) * plotW;
          ctx.fillRect(x - 1, top + 2, 2, 6);
        }
      }
    });
    // flagged steps as thin red ticks along the bottom
    if (this.kinematics) {
      ctx.fillStyle = withAlpha(THEME.red, 0.6);
      for (let i = 0; i < n; i++) {
        if (this.kinematics.has(i, StepFlag.Teleport) || this.kinematics.has(i, StepFlag.Inconsistent)) {
          const x = this.leftGutter + (i / Math.max(1, n - 1)) * plotW;
          ctx.fillRect(x - 0.5, height - 4, 1, 4);
        }
      }
    }
  }

  drawDynamic(ctx: CanvasRenderingContext2D, width: number, height: number, step: number): void {
    if (!this.identity) return;
    const n = this.identity.stepCount;
    const plotW = width - this.leftGutter - this.rightGutter;
    const x = this.leftGutter + (step / Math.max(1, n - 1)) * plotW;
    ctx.strokeStyle = THEME.textStrong;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, this.topPad); ctx.lineTo(x, height); ctx.stroke();
    const stripH = (height - this.topPad) / this.strips.length;
    ctx.font = "10px Inter, system-ui, sans-serif";
    this.strips.forEach((strip, si) => {
      const v = strip.values[step];
      const label = Number.isFinite(v) ? `${fmt(v)} ${strip.unit}` : "—";
      const tx = x + 6 + 70 > width ? x - 76 : x + 6;
      ctx.fillStyle = withAlpha(THEME.card, 0.92);
      ctx.fillRect(tx - 2, this.topPad + si * stripH + 2, 72, 13);
      ctx.fillStyle = strip.color;
      ctx.fillText(label, tx, this.topPad + si * stripH + 12);
    });
  }
}

function robustMax(values: Float32Array): number {
  const finite = Array.from(values).filter(Number.isFinite).sort((a, b) => a - b);
  if (finite.length === 0) return 1;
  return finite[Math.min(finite.length - 1, Math.floor(0.98 * finite.length))] || 1;
}

function fmt(v: number): string {
  const a = Math.abs(v);
  return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toFixed(2);
}
