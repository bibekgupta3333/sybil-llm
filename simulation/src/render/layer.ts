/**
 * Static/dynamic split for 60 fps canvas rendering.
 *
 * `drawStatic` is rasterised once per (identity, viewport) into an offscreen canvas; each frame
 * only blits it and calls `drawDynamic` for the moving parts (markers, playhead, readouts).
 */
export interface Layer {
  drawStatic(ctx: CanvasRenderingContext2D, width: number, height: number): void;
  drawDynamic(ctx: CanvasRenderingContext2D, width: number, height: number, step: number): void;
}

/** Owns a `<canvas>`, its DPR-aware sizing, and the offscreen static cache for a set of layers. */
export class CanvasSurface {
  readonly ctx: CanvasRenderingContext2D;
  private staticCache: HTMLCanvasElement | null = null;
  private dirty = true;
  private cssWidth = 1;
  private cssHeight = 1;
  private layers: Layer[] = [];
  private readonly observer: ResizeObserver | null;

  constructor(readonly canvas: HTMLCanvasElement, private readonly onResize?: (w: number, h: number) => void) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2D canvas context unavailable");
    this.ctx = ctx;
    this.observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(() => this.fitToElement());
    this.observer?.observe(canvas);
    this.fitToElement();
  }

  get width(): number {
    return this.cssWidth;
  }

  get height(): number {
    return this.cssHeight;
  }

  setLayers(layers: Layer[]): void {
    this.layers = layers;
    this.invalidate();
  }

  /** Mark the static cache stale (identity changed, viewport moved, toggle flipped). */
  invalidate(): void {
    this.dirty = true;
  }

  render(step: number): void {
    const { ctx } = this;
    const dpr = window.devicePixelRatio || 1;
    if (this.dirty) this.rebuildStatic(dpr);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (this.staticCache) ctx.drawImage(this.staticCache, 0, 0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    for (const layer of this.layers) layer.drawDynamic(ctx, this.cssWidth, this.cssHeight, step);
  }

  toBlob(): Promise<Blob | null> {
    return new Promise((resolve) => this.canvas.toBlob(resolve, "image/png"));
  }

  dispose(): void {
    this.observer?.disconnect();
  }

  /** Force a re-measure of the backing element — needed after it goes from `hidden` to visible. */
  remeasure(): void {
    this.fitToElement();
  }

  private fitToElement(): void {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    if (w === this.cssWidth && h === this.cssHeight && this.canvas.width === Math.round(w * dpr)) return;
    this.cssWidth = w;
    this.cssHeight = h;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.invalidate();
    this.onResize?.(w, h);
  }

  private rebuildStatic(dpr: number): void {
    const off = document.createElement("canvas");
    off.width = this.canvas.width;
    off.height = this.canvas.height;
    const octx = off.getContext("2d");
    if (!octx) return;
    octx.setTransform(dpr, 0, 0, dpr, 0, 0);
    for (const layer of this.layers) layer.drawStatic(octx, this.cssWidth, this.cssHeight);
    this.staticCache = off;
    this.dirty = false;
  }
}
