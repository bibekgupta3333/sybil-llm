import { THEME } from "../../ui/theme";
import { shortFeature } from "./colors";

export interface StripsInput {
  /** n × features, real rows only (normalised or raw). */
  rows: Float32Array;
  n: number;
  featureCount: number;
  selected: readonly number[];
  names: readonly string[];
  units: readonly string[];
  raw: boolean;
}

const STRIP_COLORS = [THEME.blue, THEME.purple, THEME.orange, THEME.green, THEME.red];

/** One small line chart per selected feature over the real messages of the window (x = message index). */
export class FeatureStrips {
  static draw(ctx: CanvasRenderingContext2D, width: number, height: number, input: StripsInput): void {
    ctx.fillStyle = THEME.panel;
    ctx.fillRect(0, 0, width, height);
    const { rows, n, featureCount: f, selected } = input;
    if (selected.length === 0 || n === 0) {
      ctx.fillStyle = THEME.dim;
      ctx.font = "12px Inter, system-ui, sans-serif";
      ctx.fillText(n === 0 ? "no real rows" : "select features above", 12, 20);
      return;
    }
    const left = 118;
    const right = 10;
    const top = 6;
    const bottom = 16;
    const rowH = (height - top - bottom) / selected.length;
    const plotW = width - left - right;
    const xOf = (i: number): number => left + (n === 1 ? plotW / 2 : (i / (n - 1)) * plotW);
    ctx.font = "10px Inter, system-ui, sans-serif";
    selected.forEach((j, s) => {
      const y0 = top + s * rowH;
      const color = STRIP_COLORS[s % STRIP_COLORS.length];
      let lo = Infinity;
      let hi = -Infinity;
      for (let i = 0; i < n; i++) {
        const v = rows[i * f + j];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      if (hi - lo < 1e-9) {
        lo -= 1;
        hi += 1;
      }
      const yOf = (v: number): number => y0 + 6 + (1 - (v - lo) / (hi - lo)) * (rowH - 14);
      ctx.strokeStyle = THEME.gridLine;
      ctx.beginPath();
      ctx.moveTo(left, y0 + rowH - 2 + 0.5);
      ctx.lineTo(width - right, y0 + rowH - 2 + 0.5);
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.font = "600 11px Inter, system-ui, sans-serif";
      ctx.fillText(shortFeature(input.names[j]), 6, y0 + 16);
      ctx.font = "10px Inter, system-ui, sans-serif";
      ctx.fillStyle = THEME.dim;
      ctx.fillText(input.raw ? input.units[j] : "z (train std)", 6, y0 + 29);
      ctx.textAlign = "right";
      ctx.fillText(fmt(hi), left - 4, y0 + 12);
      ctx.fillText(fmt(lo), left - 4, y0 + rowH - 6);
      ctx.textAlign = "left";
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = xOf(i);
        const y = yOf(rows[i * f + j]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.fillStyle = color;
      if (n <= 64) {
        for (let i = 0; i < n; i++) {
          ctx.beginPath();
          ctx.arc(xOf(i), yOf(rows[i * f + j]), 1.8, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    });
    ctx.fillStyle = THEME.dim;
    ctx.fillText("message 0", left, height - 3);
    ctx.textAlign = "right";
    ctx.fillText(`message ${n - 1} (real rows only)`, width - right, height - 3);
    ctx.textAlign = "left";
  }
}

function fmt(v: number): string {
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}
