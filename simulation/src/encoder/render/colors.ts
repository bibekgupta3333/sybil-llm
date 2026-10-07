/** Colour scales shared by the encoder renderers (same ramps as the legacy heatmap). */

/** −1 → primary blue, 0 → light neutral, +1 → red. Input is clamped. */
export function diverging(t: number): string {
  const x = Math.max(-1, Math.min(1, t));
  const c = [240, 240, 240];
  const end = x < 0 ? [37, 99, 235] : [220, 38, 38];
  const a = Math.abs(x);
  return `rgb(${Math.round(c[0] + (end[0] - c[0]) * a)},${Math.round(c[1] + (end[1] - c[1]) * a)},${Math.round(c[2] + (end[2] - c[2]) * a)})`;
}

/** 0 → primary-light, 1 → primary blue. */
export function sequential(t: number): string {
  const x = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0));
  return `rgb(${Math.round(219 + (37 - 219) * x)},${Math.round(234 + (99 - 234) * x)},${Math.round(254 + (235 - 254) * x)})`;
}

/** Diagonal hatch pattern used for padding rows (what the mask hides from the encoder). */
export function hatchPattern(ctx: CanvasRenderingContext2D): CanvasPattern | string {
  const tile = document.createElement("canvas");
  tile.width = 6;
  tile.height = 6;
  const t = tile.getContext("2d");
  if (!t) return "#E5E7EB";
  t.fillStyle = "#EEF2F6";
  t.fillRect(0, 0, 6, 6);
  t.strokeStyle = "#CBD5E1";
  t.lineWidth = 1;
  t.beginPath();
  t.moveTo(0, 6);
  t.lineTo(6, 0);
  t.moveTo(-1, 1);
  t.lineTo(1, -1);
  t.moveTo(5, 7);
  t.lineTo(7, 5);
  t.stroke();
  return ctx.createPattern(tile, "repeat") ?? "#E5E7EB";
}

/** Compact feature label for column headers: `claimed_pos_x` → `c.pos x`, `rx_vel_y` → `r.vel y`. */
export function shortFeature(name: string): string {
  return name
    .replace(/^claimed_/, "c.")
    .replace(/^rx_/, "r.")
    .replace(/_([xy])$/, " $1")
    .replace(/_/g, " ");
}
