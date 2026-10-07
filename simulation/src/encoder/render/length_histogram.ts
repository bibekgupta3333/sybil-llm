import { THEME } from "../../ui/theme";

/** Bar chart of window lengths n = 1..64 (64 bins); the n = 64 bin is the "full window" bar. */
export class LengthHistogram {
  static draw(ctx: CanvasRenderingContext2D, width: number, height: number, bins: readonly number[]): void {
    ctx.fillStyle = THEME.panel;
    ctx.fillRect(0, 0, width, height);
    const left = 34;
    const bottom = 18;
    const top = 8;
    const max = Math.max(1, ...bins);
    const bw = (width - left - 8) / Math.max(1, bins.length);
    ctx.font = "10px Inter, system-ui, sans-serif";
    ctx.fillStyle = THEME.dim;
    ctx.textAlign = "right";
    ctx.fillText(String(max), left - 4, top + 8);
    ctx.fillText("0", left - 4, height - bottom);
    ctx.textAlign = "left";
    bins.forEach((c, i) => {
      if (c <= 0) return;
      const h = ((height - top - bottom) * c) / max;
      ctx.fillStyle = i === bins.length - 1 ? THEME.purple : THEME.blue;
      ctx.fillRect(left + i * bw + 0.5, height - bottom - h, Math.max(1, bw - 1), h);
    });
    ctx.strokeStyle = THEME.border;
    ctx.beginPath();
    ctx.moveTo(left, height - bottom + 0.5);
    ctx.lineTo(width - 8, height - bottom + 0.5);
    ctx.stroke();
    ctx.fillStyle = THEME.dim;
    for (const n of [1, 16, 32, 48, 64]) {
      if (n > bins.length) continue;
      ctx.fillText(`n=${n}`, left + (n - 1) * bw, height - 4);
    }
  }
}
