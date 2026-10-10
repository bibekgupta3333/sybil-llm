import { THEME, withAlpha } from "../../ui/theme";
import { binScores, rangeFraction, type Range } from "../core/histogram";
import { classColor } from "../core/types";

export interface HistogramRow {
  readonly cls: string;
  readonly scores: readonly number[];
}

export interface HistogramInput {
  readonly rows: readonly HistogramRow[];
  readonly range: Range;
  readonly theta: number;
  readonly nBins: number;
  /** Score of the selected window (marker), if any. */
  readonly marker?: { readonly cls: string; readonly score: number } | null;
}

/**
 * One histogram row per class on a shared score axis, each row scaled to its own max (counts differ per
 * class), with the label-free threshold θ as a dashed vertical line through every row.
 */
export class ScoreHistogram {
  static draw(ctx: CanvasRenderingContext2D, width: number, height: number, g: HistogramInput): void {
    ctx.fillStyle = THEME.panel;
    ctx.fillRect(0, 0, width, height);
    const left = 132;
    const right = 14;
    const top = 8;
    const bottom = 26;
    const plotW = Math.max(10, width - left - right);
    const nRows = Math.max(1, g.rows.length);
    const rowH = (height - top - bottom) / nRows;
    const bw = plotW / g.nBins;
    const xOf = (v: number): number => left + rangeFraction(v, g.range) * plotW;
    ctx.font = "11px Inter, system-ui, sans-serif";

    g.rows.forEach((row, r) => {
      const y0 = top + r * rowH;
      const base = y0 + rowH - 3;
      const counts = binScores(row.scores, g.range, g.nBins);
      const max = Math.max(1, ...counts);
      const color = classColor(row.cls);
      ctx.fillStyle = r % 2 ? THEME.bg : THEME.panel;
      ctx.fillRect(left, y0, plotW, rowH);
      counts.forEach((c, b) => {
        if (c <= 0) return;
        const h = ((rowH - 8) * c) / max;
        ctx.fillStyle = withAlpha(color, 0.75);
        ctx.fillRect(left + b * bw + 0.5, base - h, Math.max(1, bw - 1), h);
      });
      ctx.strokeStyle = THEME.border;
      ctx.beginPath();
      ctx.moveTo(left, base + 0.5);
      ctx.lineTo(left + plotW, base + 0.5);
      ctx.stroke();
      ctx.fillStyle = THEME.textStrong;
      ctx.textAlign = "right";
      ctx.fillText(row.cls, left - 8, y0 + rowH / 2);
      ctx.fillStyle = THEME.dim;
      ctx.fillText(`n=${row.scores.length} · max ${max}`, left - 8, y0 + rowH / 2 + 13);
      ctx.textAlign = "left";
      if (g.marker && g.marker.cls === row.cls) {
        const mx = xOf(g.marker.score);
        ctx.fillStyle = THEME.textStrong;
        ctx.beginPath();
        ctx.moveTo(mx, base - 1);
        ctx.lineTo(mx - 5, base + 7);
        ctx.lineTo(mx + 5, base + 7);
        ctx.closePath();
        ctx.fill();
      }
    });

    // θ line through all rows.
    const tx = xOf(g.theta);
    ctx.strokeStyle = THEME.red;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(tx, top);
    ctx.lineTo(tx, height - bottom + 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.lineWidth = 1;

    // Axis labels.
    ctx.fillStyle = THEME.dim;
    const yAxis = height - 8;
    ctx.fillText(fmtScore(g.range.lo), left, yAxis);
    ctx.textAlign = "right";
    ctx.fillText(fmtScore(g.range.hi), left + plotW, yAxis);
    ctx.textAlign = "center";
    ctx.fillStyle = THEME.red;
    ctx.font = "600 11px Inter, system-ui, sans-serif";
    const label = `θ = ${g.theta}`;
    const lx = Math.min(Math.max(tx, left + 60), left + plotW - 60);
    ctx.fillText(label, lx, yAxis);
    ctx.textAlign = "left";
    ctx.font = "11px Inter, system-ui, sans-serif";
    ctx.fillStyle = THEME.muted;
    ctx.fillText("anomaly score →", 8, yAxis);
  }
}

function fmtScore(v: number): string {
  return Math.abs(v) >= 100 ? v.toFixed(0) : v.toPrecision(3);
}
