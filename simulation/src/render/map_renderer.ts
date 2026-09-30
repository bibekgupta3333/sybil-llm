import type { Identity } from "../core/identity";
import { KinematicsSeries, StepFlag } from "../core/kinematics";
import type { BBox } from "../core/manifest";
import { Playback } from "../core/playback";
import { Viewport } from "../core/viewport";
import { THEME, accelColor, withAlpha } from "../ui/theme";
import type { Layer } from "./layer";

/** Distinct hues for pseudonyms of one physical sender in the multi-identity view. */
const PSEUDONYM_HUES = ["#2563EB", "#06B6D4", "#8B5CF6", "#14B8A6", "#F59E0B", "#F97316", "#EC4899", "#64748B"];

/** One identity drawn on the map: static trail + dynamic vehicle marker. */
export class TrajectoryLayer {
  readonly kinematics: KinematicsSeries;

  constructor(
    readonly identity: Identity,
    readonly color: string,
    readonly label: string,
    /** Position within the scene; used to stagger overlapping labels. */
    readonly index: number,
  ) {
    this.kinematics = KinematicsSeries.compute(identity);
  }

  /** Step of this identity at an absolute simulation time (pseudonyms are time-aligned this way). */
  stepAt(simTime: number): number {
    return Playback.stepIndexAt(this.identity.times, simTime);
  }
}

/** Background grid, honest-road context box, trails, markers, vectors. */
export class MapRenderer implements Layer {
  readonly viewport = new Viewport();
  private layers: TrajectoryLayer[] = [];
  private primary: TrajectoryLayer | null = null;
  private lockToRoad = false;
  private accelScale = 3; // m/s² that maps to the hottest trail colour
  private contextBox: BBox;
  private contextLabel = "benign road extent";
  /** The identity whose step index the playback clock reports; defaults to the first layer. */
  private clock: Identity | null = null;
  /** When set, the auto-fit uses this box instead of the scene's own bbox (for shared extents). */
  private fitBox: BBox | null = null;
  /** Colour trails by layer colour even for a single-identity scene (instead of by acceleration). */
  private useLayerColor = false;

  constructor(private readonly roadBox: BBox) {
    this.contextBox = roadBox;
  }

  get scene(): readonly TrajectoryLayer[] {
    return this.layers;
  }

  /** Show a set of identities; the first is the one the playback clock follows. */
  setScene(
    identities: Identity[],
    colorFor: (identity: Identity, index: number) => string,
    labelFor: (identity: Identity, index: number) => string = (ident) => `#${ident.record.sender_pseudo}`,
  ): void {
    this.layers = identities.map((ident, i) => new TrajectoryLayer(ident, colorFor(ident, i), labelFor(ident, i), i));
    this.primary = this.layers[0] ?? null;
    this.clock = this.primary?.identity ?? null;
    this.accelScale = Math.max(1, ...this.layers.map((l) => percentile(l.kinematics.accel, 0.95)));
    this.refit();
  }

  /** Follow `identity`'s clock even when it isn't drawn on this map. Call after `setScene`. */
  setClock(identity: Identity): void {
    this.clock = identity;
  }

  /** Fit to `box` instead of the scene's own extent, so several maps can share one frame. */
  setFitBox(box: BBox | null): void {
    this.fitBox = box;
    this.refit();
  }

  setUseLayerColor(on: boolean): void {
    this.useLayerColor = on;
  }

  setLockToRoad(lock: boolean): void {
    this.lockToRoad = lock;
    this.refit();
  }

  /** Swap the dashed context box + `lockToRoad` fit target (default: the manifest-wide road bbox). */
  setContextBox(box: BBox | null, label: string): void {
    this.contextBox = box ?? this.roadBox;
    this.contextLabel = box ? label : "benign road extent";
    this.refit();
  }

  resize(width: number, height: number): void {
    this.viewport.resize(width, height);
  }

  /** Discards manual zoom/pan and restores the last auto-fit. */
  resetView(): void {
    this.viewport.reset();
  }

  /** Zoom around a canvas point (mouse wheel); clears the auto-fit until the scene changes again. */
  zoomBy(factor: number, aroundCanvasPt?: { x: number; y: number }): void {
    this.viewport.zoomBy(factor, aroundCanvasPt);
  }

  /** Pan by a canvas-pixel delta (mouse drag). */
  panBy(dxPx: number, dyPx: number): void {
    this.viewport.panBy(dxPx, dyPx);
  }

  static pseudonymColor(index: number): string {
    return PSEUDONYM_HUES[index % PSEUDONYM_HUES.length];
  }

  drawStatic(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    ctx.fillStyle = THEME.panel;
    ctx.fillRect(0, 0, width, height);
    this.drawGrid(ctx, width, height);
    this.drawRoadBox(ctx);
    for (const layer of this.layers) this.drawTrail(ctx, layer);
    this.drawScaleBar(ctx, width, height);
  }

  drawDynamic(ctx: CanvasRenderingContext2D, width: number, height: number, step: number): void {
    const simTime = this.clock?.times[step] ?? 0;
    for (const layer of this.layers) {
      const s = layer.identity === this.clock ? step : layer.stepAt(simTime);
      this.drawVehicle(ctx, layer, s, width, height);
    }
  }

  private refit(): void {
    if (this.lockToRoad || this.layers.length === 0) {
      this.viewport.fit(this.contextBox, { padding: 0.05 });
      return;
    }
    if (this.fitBox) {
      this.viewport.fit(this.fitBox, { padding: 0.08, minExtent: 300 });
      return;
    }
    let box = this.layers[0].identity.bbox();
    for (const l of this.layers.slice(1)) box = Viewport.union(box, l.identity.bbox());
    this.viewport.fit(box, { padding: 0.08, minExtent: 300 });
  }

  private drawGrid(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    const step = niceStep(width / this.viewport.pixelsPerMetre / 8);
    ctx.strokeStyle = THEME.gridLine;
    ctx.lineWidth = 1;
    ctx.fillStyle = THEME.dim;
    ctx.font = "10px Inter, system-ui, sans-serif";
    const x0 = Math.floor(this.viewport.toWorldX(0) / step) * step;
    const x1 = this.viewport.toWorldX(width);
    for (let x = x0; x <= x1; x += step) {
      const px = this.viewport.toCanvasX(x);
      ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, height); ctx.stroke();
      ctx.fillText(`${Math.round(x)} m`, px + 3, height - 4);
    }
    const y0 = Math.floor(this.viewport.toWorldY(height) / step) * step;
    const y1 = this.viewport.toWorldY(0);
    for (let y = y0; y <= y1; y += step) {
      const py = this.viewport.toCanvasY(y);
      ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(width, py); ctx.stroke();
      ctx.fillText(`${Math.round(y)}`, 3, py - 3);
    }
  }

  private drawRoadBox(ctx: CanvasRenderingContext2D): void {
    const b = this.contextBox;
    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = THEME.roadBox;
    ctx.lineWidth = 1.5;
    const x = this.viewport.toCanvasX(b.minX);
    const y = this.viewport.toCanvasY(b.maxY);
    ctx.strokeRect(x, y, (b.maxX - b.minX) * this.viewport.pixelsPerMetre, (b.maxY - b.minY) * this.viewport.pixelsPerMetre);
    ctx.fillStyle = THEME.roadBox;
    ctx.font = "10px Inter, system-ui, sans-serif";
    ctx.fillText(this.contextLabel, x + 6, y + 12);
    ctx.restore();
  }

  private drawTrail(ctx: CanvasRenderingContext2D, layer: TrajectoryLayer): void {
    const { identity: id, kinematics: k } = layer;
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    for (let i = 1; i < id.stepCount; i++) {
      const x0 = this.viewport.toCanvasX(id.posX(i - 1));
      const y0 = this.viewport.toCanvasY(id.posY(i - 1));
      const x1 = this.viewport.toCanvasX(id.posX(i));
      const y1 = this.viewport.toCanvasY(id.posY(i));
      ctx.beginPath();
      if (k.has(i, StepFlag.Teleport)) {
        ctx.setLineDash([4, 6]);
        ctx.strokeStyle = withAlpha(THEME.red, 0.7);
        ctx.lineWidth = 1;
      } else {
        ctx.setLineDash([]);
        ctx.strokeStyle = this.layers.length > 1 || this.useLayerColor ? withAlpha(layer.color, 0.75) : accelColor(k.accel[i] / this.accelScale);
        ctx.lineWidth = 2;
      }
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    // window starts as small ticks so the stride-10 tiling is visible on the map
    ctx.fillStyle = withAlpha(THEME.textStrong, 0.55);
    for (let w = 0; w < id.windowCount; w++) {
      const s = w * id.stride;
      ctx.beginPath();
      ctx.arc(this.viewport.toCanvasX(id.posX(s)), this.viewport.toCanvasY(id.posY(s)), 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  private drawVehicle(ctx: CanvasRenderingContext2D, layer: TrajectoryLayer, step: number, width: number, height: number): void {
    const id = layer.identity;
    const wx = id.posX(step);
    const wy = id.posY(step);
    const x = this.viewport.toCanvasX(wx);
    const y = this.viewport.toCanvasY(wy);
    if (!this.viewport.contains(wx, wy)) {
      this.drawEdgeTick(ctx, x, y, width, height, layer);
      return;
    }
    const vecScale = 2.5; // px per m/s
    // velocity vector (claimed)
    ctx.strokeStyle = withAlpha(THEME.blue, 0.9);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + id.spdX(step) * vecScale, y - id.spdY(step) * vecScale);
    ctx.stroke();
    // heading arrow (unit vector, fixed 22 px)
    const hx = id.hedX(step);
    const hy = id.hedY(step);
    ctx.strokeStyle = THEME.amber;
    ctx.fillStyle = THEME.amber;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + hx * 22, y - hy * 22);
    ctx.stroke();
    ctx.beginPath();
    const ax = x + hx * 22;
    const ay = y - hy * 22;
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax - hx * 6 + hy * 4, ay + hy * 6 + hx * 4);
    ctx.lineTo(ax - hx * 6 - hy * 4, ay + hy * 6 - hx * 4);
    ctx.closePath();
    ctx.fill();
    // marker
    const flagged = layer.kinematics.has(step, StepFlag.Teleport) || layer.kinematics.has(step, StepFlag.Inconsistent);
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fillStyle = layer.color;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = flagged ? THEME.red : THEME.textStrong;
    ctx.stroke();
    if (this.layers.length > 1) {
      // Real Sybil pseudonyms often cluster within a few metres of each other, so labels are
      // fanned out radially by index (not just stacked vertically) and given a background box —
      // otherwise a 6-identity cluster renders as unreadable overlapping text.
      const angle = (layer.index / Math.max(1, this.layers.length)) * Math.PI * 2 - Math.PI / 2;
      const radius = 16 + (layer.index % 3) * 12;
      const lx = x + Math.cos(angle) * radius;
      const ly = y + Math.sin(angle) * radius;
      ctx.font = "11px Inter, system-ui, sans-serif";
      const w = ctx.measureText(layer.label).width;
      ctx.fillStyle = withAlpha(THEME.card, 0.92);
      ctx.fillRect(lx - 2, ly - 10, w + 4, 13);
      ctx.fillStyle = layer.color;
      ctx.fillText(layer.label, lx, ly);
    }
  }

  private drawEdgeTick(ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number, layer: TrajectoryLayer): void {
    const cx = Math.min(width - 6, Math.max(6, x));
    const cy = Math.min(height - 6, Math.max(6, y));
    const dist = Math.hypot(x - cx, y - cy) / this.viewport.pixelsPerMetre;
    ctx.fillStyle = layer.color;
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = THEME.textStrong;
    ctx.font = "10px Inter, system-ui, sans-serif";
    ctx.fillText(`${layer.label} · ${Math.round(dist)} m off-screen`, Math.min(cx + 8, width - 140), Math.max(12, Math.min(cy, height - 4)));
  }

  private drawScaleBar(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    const metres = niceStep((width / this.viewport.pixelsPerMetre) / 5);
    const px = metres * this.viewport.pixelsPerMetre;
    const x = width - px - 16;
    const y = height - 18;
    ctx.strokeStyle = THEME.text;
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + px, y); ctx.stroke();
    ctx.fillStyle = THEME.text;
    ctx.font = "11px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(metres >= 1000 ? `${metres / 1000} km` : `${metres} m`, x + px / 2, y - 5);
    ctx.textAlign = "left";
  }
}

function niceStep(raw: number): number {
  const pow = 10 ** Math.floor(Math.log10(Math.max(raw, 1e-6)));
  const m = raw / pow;
  const nice = m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10;
  return nice * pow;
}

function percentile(values: Float32Array, p: number): number {
  const finite = Array.from(values).filter(Number.isFinite).sort((a, b) => a - b);
  if (finite.length === 0) return 0;
  return finite[Math.min(finite.length - 1, Math.floor(p * finite.length))];
}
