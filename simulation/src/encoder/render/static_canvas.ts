/** Draw callback for a static (non-animated) canvas, in CSS pixels. */
export type DrawFn = (ctx: CanvasRenderingContext2D, width: number, height: number) => void;

/**
 * A DPR-aware canvas that redraws when its element is resized. The encoder view has no playback clock,
 * so every panel is a single static draw.
 */
export class StaticCanvas {
  readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly observer: ResizeObserver | null;
  private width = 0;
  private height = 0;

  constructor(private draw: DrawFn, className = "enc-canvas") {
    this.canvas = document.createElement("canvas");
    this.canvas.className = className;
    const ctx = this.canvas.getContext("2d");
    if (!ctx) throw new Error("2D canvas context unavailable");
    this.ctx = ctx;
    this.observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(() => this.redraw());
    this.observer?.observe(this.canvas);
  }

  setDraw(draw: DrawFn): void {
    this.draw = draw;
    this.redraw();
  }

  /** Re-measure and redraw; cheap enough to call on every state change. */
  redraw(): void {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.width = Math.max(1, Math.round(rect.width));
    this.height = Math.max(1, Math.round(rect.height));
    if (rect.width === 0 || rect.height === 0) return;
    const bw = Math.round(this.width * dpr);
    const bh = Math.round(this.height * dpr);
    if (this.canvas.width !== bw) this.canvas.width = bw;
    if (this.canvas.height !== bh) this.canvas.height = bh;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.ctx.clearRect(0, 0, this.width, this.height);
    this.draw(this.ctx, this.width, this.height);
  }

  dispose(): void {
    this.observer?.disconnect();
  }
}
