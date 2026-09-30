import type { BBox } from "./manifest";

export interface FitOptions {
  /** Fraction of the larger extent added on every side. */
  readonly padding?: number;
  /** Smallest world extent (metres) to show, so a parked vehicle is not zoomed to a pixel. */
  readonly minExtent?: number;
}

/**
 * World (SUMO planar metres, y up) ↔ canvas (CSS pixels, y down) transform with locked aspect.
 */
export class Viewport {
  private scale = 1; // px per metre
  private originX = 0; // world x at canvas x = 0
  private originY = 0; // world y at canvas y = height
  private width = 1;
  private height = 1;
  private fitted: BBox | null = null;
  private lastBox: BBox | null = null;
  private lastOptions: FitOptions = {};
  private manuallyAdjusted = false;

  resize(width: number, height: number): void {
    this.width = Math.max(1, width);
    this.height = Math.max(1, height);
    if (this.lastBox) this.fit(this.lastBox, this.lastOptions);
  }

  get pixelsPerMetre(): number {
    return this.scale;
  }

  get bounds(): BBox | null {
    return this.fitted;
  }

  /** Fit `bbox` inside the canvas, aspect-locked, centred. Clears any manual zoom/pan. */
  fit(bbox: BBox, options: FitOptions = {}): void {
    this.lastBox = bbox;
    this.lastOptions = options;
    this.manuallyAdjusted = false;
    const padding = options.padding ?? 0.08;
    const minExtent = options.minExtent ?? 300;
    const cx = (bbox.minX + bbox.maxX) / 2;
    const cy = (bbox.minY + bbox.maxY) / 2;
    const extentX = Math.max(bbox.maxX - bbox.minX, minExtent);
    const extentY = Math.max(bbox.maxY - bbox.minY, minExtent);
    const padded = Math.max(extentX, extentY) * (1 + 2 * padding);
    const sx = this.width / Math.max(extentX * (1 + 2 * padding), 1e-9);
    const sy = this.height / Math.max(extentY * (1 + 2 * padding), 1e-9);
    this.scale = Math.min(sx, sy);
    this.originX = cx - this.width / (2 * this.scale);
    this.originY = cy - this.height / (2 * this.scale);
    this.fitted = { minX: cx - padded / 2, minY: cy - padded / 2, maxX: cx + padded / 2, maxY: cy + padded / 2 };
  }

  toCanvasX(worldX: number): number {
    return (worldX - this.originX) * this.scale;
  }

  toCanvasY(worldY: number): number {
    return this.height - (worldY - this.originY) * this.scale;
  }

  toWorldX(px: number): number {
    return px / this.scale + this.originX;
  }

  toWorldY(py: number): number {
    return (this.height - py) / this.scale + this.originY;
  }

  get isManuallyAdjusted(): boolean {
    return this.manuallyAdjusted;
  }

  /** Zoom by `factor` (>1 in, <1 out) around a canvas point (default: centre). Clamped to sane bounds. */
  zoomBy(factor: number, aroundCanvasPt?: { x: number; y: number }): void {
    const pt = aroundCanvasPt ?? { x: this.width / 2, y: this.height / 2 };
    const worldX = this.toWorldX(pt.x);
    const worldY = this.toWorldY(pt.y);
    const newScale = Math.min(500, Math.max(0.001, this.scale * factor));
    this.originX = worldX - pt.x / newScale;
    this.originY = worldY - (this.height - pt.y) / newScale;
    this.scale = newScale;
    this.manuallyAdjusted = true;
  }

  /** Pan by a canvas-pixel delta (e.g. from a mouse drag). */
  panBy(dxPx: number, dyPx: number): void {
    this.originX -= dxPx / this.scale;
    this.originY += dyPx / this.scale; // canvas y grows downward; world y grows upward
    this.manuallyAdjusted = true;
  }

  /** Discards any manual zoom/pan and restores the last `fit()`. */
  reset(): void {
    this.manuallyAdjusted = false;
    if (this.lastBox) this.fit(this.lastBox, this.lastOptions);
  }

  /** True when the world point falls inside the canvas. */
  contains(worldX: number, worldY: number): boolean {
    const x = this.toCanvasX(worldX);
    const y = this.toCanvasY(worldY);
    return x >= 0 && x <= this.width && y >= 0 && y <= this.height;
  }

  static union(a: BBox, b: BBox): BBox {
    return {
      minX: Math.min(a.minX, b.minX),
      minY: Math.min(a.minY, b.minY),
      maxX: Math.max(a.maxX, b.maxX),
      maxY: Math.max(a.maxY, b.maxY),
    };
  }

  static fromArray(box: readonly [number, number, number, number]): BBox {
    return { minX: box[0], minY: box[1], maxX: box[2], maxY: box[3] };
  }
}
