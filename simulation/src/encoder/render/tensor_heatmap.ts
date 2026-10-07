import { THEME } from "../../ui/theme";
import { diverging, hatchPattern, sequential, shortFeature } from "./colors";

export interface TensorInput {
  /** seq_len × features, row-major (time rows). Normalised: padding = 0. Raw: padding = NaN. */
  matrix: Float32Array;
  mask: Uint8Array;
  features: readonly string[];
  units: readonly string[];
  raw: boolean;
}

const MASK_COL_W = 16;

/**
 * The encoder's input tensor drawn as time (rows, t = 0 at the top) × features (columns), with padding rows
 * hatched and the mask as a separate column on the right. Normalised mode: diverging scale ±3 train std.
 * Raw mode: each feature column scaled to its own min/max over the real rows.
 */
export class TensorHeatmap {
  static draw(ctx: CanvasRenderingContext2D, width: number, height: number, input: TensorInput): void {
    const { matrix, mask, features, raw } = input;
    const f = features.length;
    const T = mask.length;
    const left = 30;
    const top = 74;
    const bottom = 22;
    const gap = 8;
    const gridW = Math.max(10, width - left - gap - MASK_COL_W - 8);
    const cellW = gridW / f;
    const cellH = Math.max(1, (height - top - bottom) / T);
    const n = countReal(mask);
    const { lo, hi } = columnRange(matrix, mask, f);

    ctx.fillStyle = THEME.panel;
    ctx.fillRect(0, 0, width, height);
    const hatch = hatchPattern(ctx);
    for (let t = 0; t < T; t++) {
      const y = top + t * cellH;
      if (!mask[t]) {
        ctx.fillStyle = hatch;
        ctx.fillRect(left, y, gridW, Math.ceil(cellH));
        continue;
      }
      for (let j = 0; j < f; j++) {
        const v = matrix[t * f + j];
        ctx.fillStyle = raw ? sequential((v - lo[j]) / Math.max(1e-9, hi[j] - lo[j])) : diverging(v / 3);
        ctx.fillRect(left + j * cellW, y, Math.ceil(cellW), Math.ceil(cellH));
      }
    }
    // Mask column.
    const mx = left + gridW + gap;
    for (let t = 0; t < T; t++) {
      ctx.fillStyle = mask[t] ? THEME.green : "#E2E8F0";
      ctx.fillRect(mx, top + t * cellH, MASK_COL_W, Math.max(1, cellH - (cellH > 4 ? 1 : 0)));
    }
    // Column separators.
    ctx.strokeStyle = "rgba(255,255,255,0.7)";
    ctx.lineWidth = 1;
    for (let j = 1; j < f; j++) {
      ctx.beginPath();
      ctx.moveTo(left + j * cellW + 0.5, top);
      ctx.lineTo(left + j * cellW + 0.5, top + n * cellH);
      ctx.stroke();
    }
    // Real / padding boundary.
    if (n > 0 && n < T) {
      const y = top + n * cellH;
      ctx.strokeStyle = THEME.textStrong;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(left - 4, y);
      ctx.lineTo(mx + MASK_COL_W + 4, y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    // Feature headers (rotated).
    ctx.font = "10px Inter, system-ui, sans-serif";
    ctx.fillStyle = THEME.muted;
    for (let j = 0; j < f; j++) {
      ctx.save();
      ctx.translate(left + j * cellW + cellW / 2 + 3, top - 4);
      ctx.rotate(-Math.PI / 3);
      ctx.fillText(shortFeature(features[j]), 0, 0);
      ctx.restore();
    }
    ctx.save();
    ctx.translate(mx + MASK_COL_W / 2 + 3, top - 4);
    ctx.rotate(-Math.PI / 3);
    ctx.fillStyle = THEME.green;
    ctx.font = "600 10px Inter, system-ui, sans-serif";
    ctx.fillText("mask", 0, 0);
    ctx.restore();
    // Time labels.
    ctx.font = "10px Inter, system-ui, sans-serif";
    ctx.fillStyle = THEME.dim;
    ctx.textAlign = "right";
    for (let t = 0; t < T; t += 8) ctx.fillText(`t${t}`, left - 4, top + t * cellH + Math.min(cellH, 10));
    ctx.fillText(`t${T - 1}`, left - 4, top + T * cellH - 1);
    ctx.textAlign = "left";
    // Legend line.
    ctx.fillStyle = THEME.muted;
    const legend = raw ? "raw units · each column min→max over real rows" : "normalised (what the encoder sees) · blue −3σ · 0 · red +3σ";
    ctx.fillText(legend, left, height - 7);
    if (n < T) {
      const y = top + n * cellH + 12;
      if (y < top + T * cellH - 4) {
        ctx.fillStyle = THEME.muted;
        ctx.font = "600 11px Inter, system-ui, sans-serif";
        const label = `padding · ${T - n} rows · mask = 0`;
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = "rgba(255,255,255,0.85)";
        ctx.fillRect(left + gridW / 2 - tw / 2 - 5, y - 10, tw + 10, 15);
        ctx.fillStyle = THEME.text;
        ctx.fillText(label, left + gridW / 2 - tw / 2, y + 1);
      }
    }
  }

  /** Thumbnail: no labels, just cells + hatch + a mask strip on the right. */
  static drawMini(ctx: CanvasRenderingContext2D, width: number, height: number, input: TensorInput): void {
    const { matrix, mask, features, raw } = input;
    const f = features.length;
    const T = mask.length;
    const maskW = 5;
    const gridW = width - maskW - 2;
    const cellW = gridW / f;
    const cellH = height / T;
    const { lo, hi } = columnRange(matrix, mask, f);
    ctx.fillStyle = hatchPattern(ctx);
    ctx.fillRect(0, 0, gridW, height);
    for (let t = 0; t < T; t++) {
      if (!mask[t]) continue;
      for (let j = 0; j < f; j++) {
        const v = matrix[t * f + j];
        ctx.fillStyle = raw ? sequential((v - lo[j]) / Math.max(1e-9, hi[j] - lo[j])) : diverging(v / 3);
        ctx.fillRect(j * cellW, t * cellH, Math.ceil(cellW), Math.ceil(cellH));
      }
    }
    for (let t = 0; t < T; t++) {
      ctx.fillStyle = mask[t] ? THEME.green : "#E2E8F0";
      ctx.fillRect(gridW + 2, t * cellH, maskW, Math.ceil(cellH));
    }
  }
}

function countReal(mask: Uint8Array): number {
  let n = 0;
  for (const m of mask) n += m ? 1 : 0;
  return n;
}

function columnRange(matrix: Float32Array, mask: Uint8Array, f: number): { lo: Float32Array; hi: Float32Array } {
  const lo = new Float32Array(f).fill(Infinity);
  const hi = new Float32Array(f).fill(-Infinity);
  for (let t = 0; t < mask.length; t++) {
    if (!mask[t]) continue;
    for (let j = 0; j < f; j++) {
      const v = matrix[t * f + j];
      if (!Number.isFinite(v)) continue;
      if (v < lo[j]) lo[j] = v;
      if (v > hi[j]) hi[j] = v;
    }
  }
  return { lo, hi };
}
