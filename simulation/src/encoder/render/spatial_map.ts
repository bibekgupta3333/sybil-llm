import { THEME, withAlpha } from "../../ui/theme";

export interface SpatialInput {
  claimed: readonly (readonly [number, number])[];
  receiver: readonly (readonly [number, number])[];
  range: readonly number[];
  bbox: readonly [number, number, number, number];
}

export const CLAIMED_COLOR = THEME.orange;
export const RECEIVER_COLOR = THEME.blue;

/**
 * Claimed sender positions (what the message says) against the receiver's own GPS positions, in SUMO metres,
 * with a thin line per message joining the two (its length is the `range` feature). Auto-fits the window's
 * bounding box at equal aspect; the scale bar switches to km for large extents.
 */
export class SpatialMap {
  static draw(ctx: CanvasRenderingContext2D, width: number, height: number, g: SpatialInput | null): void {
    ctx.fillStyle = THEME.panel;
    ctx.fillRect(0, 0, width, height);
    if (!g || g.claimed.length === 0) {
      ctx.fillStyle = THEME.dim;
      ctx.font = "12px Inter, system-ui, sans-serif";
      ctx.fillText("no window selected", 12, 20);
      return;
    }
    const pad = 28;
    const [x0, y0, x1, y1] = g.bbox;
    const spanX = Math.max(x1 - x0, 20);
    const spanY = Math.max(y1 - y0, 20);
    const scale = Math.min((width - 2 * pad) / spanX, (height - 2 * pad - 14) / spanY);
    const cx = (x0 + x1) / 2;
    const cy = (y0 + y1) / 2;
    const px = (x: number): number => width / 2 + (x - cx) * scale;
    const py = (y: number): number => (height - 14) / 2 - (y - cy) * scale; // SUMO y points up

    drawGrid(ctx, width, height, scale, px, py, x0, y0, x1, y1);

    // Range lines.
    ctx.strokeStyle = withAlpha(THEME.muted, 0.35);
    ctx.lineWidth = 1;
    const m = Math.min(g.claimed.length, g.receiver.length);
    for (let i = 0; i < m; i++) {
      ctx.beginPath();
      ctx.moveTo(px(g.claimed[i][0]), py(g.claimed[i][1]));
      ctx.lineTo(px(g.receiver[i][0]), py(g.receiver[i][1]));
      ctx.stroke();
    }
    polyline(ctx, g.receiver, px, py, RECEIVER_COLOR);
    polyline(ctx, g.claimed, px, py, CLAIMED_COLOR);

    // Farthest message: label its range.
    let far = 0;
    for (let i = 1; i < g.range.length; i++) if (g.range[i] > g.range[far]) far = i;
    if (g.range.length > 0 && far < m) {
      const mxp = (px(g.claimed[far][0]) + px(g.receiver[far][0])) / 2;
      const myp = (py(g.claimed[far][1]) + py(g.receiver[far][1])) / 2;
      const label = `max range ${fmtDistance(g.range[far])}`;
      ctx.font = "600 11px Inter, system-ui, sans-serif";
      const tw = ctx.measureText(label).width;
      const lx = Math.min(Math.max(4, mxp - tw / 2), width - tw - 8);
      const ly = Math.min(Math.max(16, myp), height - 30);
      ctx.fillStyle = "rgba(255,255,255,0.88)";
      ctx.fillRect(lx - 4, ly - 12, tw + 8, 16);
      ctx.fillStyle = THEME.textStrong;
      ctx.fillText(label, lx, ly);
    }
    drawScaleBar(ctx, width, height, scale);
  }
}

function polyline(
  ctx: CanvasRenderingContext2D,
  pts: readonly (readonly [number, number])[],
  px: (x: number) => number,
  py: (y: number) => number,
  color: string,
): void {
  if (pts.length === 0) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(px(x), py(y)) : ctx.lineTo(px(x), py(y))));
  ctx.stroke();
  ctx.fillStyle = color;
  for (const [x, y] of pts) {
    ctx.beginPath();
    ctx.arc(px(x), py(y), 2.4, 0, Math.PI * 2);
    ctx.fill();
  }
  // First message: ring marker.
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.fillStyle = THEME.panel;
  ctx.beginPath();
  ctx.arc(px(pts[0][0]), py(pts[0][1]), 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
}

function drawGrid(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  scale: number,
  px: (x: number) => number,
  py: (y: number) => number,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
): void {
  const step = niceStep(80 / scale);
  ctx.strokeStyle = THEME.gridLine;
  ctx.lineWidth = 1;
  const span = Math.max(x1 - x0, y1 - y0) + 4 * step + width / scale + height / scale;
  const gx0 = Math.floor((x0 - span) / step) * step;
  const gy0 = Math.floor((y0 - span) / step) * step;
  for (let x = gx0; x <= x1 + span; x += step) {
    const sx = Math.round(px(x)) + 0.5;
    if (sx < 0 || sx > width) continue;
    ctx.beginPath();
    ctx.moveTo(sx, 0);
    ctx.lineTo(sx, height);
    ctx.stroke();
  }
  for (let y = gy0; y <= y1 + span; y += step) {
    const sy = Math.round(py(y)) + 0.5;
    if (sy < 0 || sy > height) continue;
    ctx.beginPath();
    ctx.moveTo(0, sy);
    ctx.lineTo(width, sy);
    ctx.stroke();
  }
}

function drawScaleBar(ctx: CanvasRenderingContext2D, width: number, height: number, scale: number): void {
  const metres = niceStep(120 / scale);
  const len = metres * scale;
  const x = width - len - 16;
  const y = height - 12;
  ctx.strokeStyle = THEME.textStrong;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, y - 5);
  ctx.lineTo(x, y);
  ctx.lineTo(x + len, y);
  ctx.lineTo(x + len, y - 5);
  ctx.stroke();
  ctx.fillStyle = THEME.textStrong;
  ctx.font = "11px Inter, system-ui, sans-serif";
  const label = fmtDistance(metres);
  ctx.fillText(label, x + len / 2 - ctx.measureText(label).width / 2, y - 7);
}

/** 1-2-5 rounding. */
export function niceStep(raw: number): number {
  if (!Number.isFinite(raw) || raw <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

export function fmtDistance(m: number): string {
  if (Math.abs(m) >= 1000) return `${(m / 1000).toFixed(m >= 10000 ? 1 : 2)} km`;
  return `${Math.round(m)} m`;
}
