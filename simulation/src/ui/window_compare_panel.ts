import type { Dataset } from "../core/dataset";
import type { Identity } from "../core/identity";
import { HeatmapRenderer } from "../render/heatmap_renderer";
import { CanvasSurface } from "../render/layer";
import { StripRenderer } from "../render/strip_renderer";
import { ConsistencyPanel } from "./consistency_panel";
import { ProvenanceStrip } from "./provenance";

type Side = "truth" | "attack";

interface SideWidgets {
  readonly heatmap: HeatmapRenderer;
  readonly heatmapSurface: CanvasSurface;
  readonly strip: StripRenderer;
  readonly stripSurface: CanvasSurface;
  readonly consistency: ConsistencyPanel;
  readonly provenance: ProvenanceStrip;
  readonly caption: HTMLElement;
  readonly readout: HTMLElement;
}

/**
 * Two frozen `(identity, window)` snapshots side by side — no shared playback clock. Each side is
 * `Identity.windowSlice(k)` fed into the same heatmap/strip/consistency/provenance widgets the
 * live views use, so a ground-truth window and an attack window can be held next to each other and
 * stepped through independently.
 */
export class WindowComparePanel {
  private readonly truth: SideWidgets;
  private readonly attack: SideWidgets;
  private truthWindow: Identity | null = null;
  private truthStep = 0;
  private attackWindow: Identity | null = null;
  private attackStep = 0;

  constructor(
    root: HTMLElement,
    private readonly dataset: Dataset,
    titles: { readonly truth: string; readonly attack: string } = { truth: "Ground truth", attack: "Attack window" },
  ) {
    root.innerHTML = `
      <div class="wc-col" data-side="truth"><h3>${titles.truth}</h3>${WindowComparePanel.sideMarkup()}</div>
      <div class="wc-col" data-side="attack"><h3>${titles.attack}</h3>${WindowComparePanel.sideMarkup()}</div>`;
    this.truth = this.buildSide(root, "truth");
    this.attack = this.buildSide(root, "attack");
  }

  /** Point one side at a specific window of an identity; freezes it there until called again. */
  setSide(side: Side, identity: Identity, windowIndex: number): void {
    const sliced = identity.windowSlice(windowIndex);
    const w = side === "truth" ? this.truth : this.attack;
    if (side === "truth") { this.truthWindow = sliced; this.truthStep = 0; } else { this.attackWindow = sliced; this.attackStep = 0; }
    w.heatmap.setIdentity(sliced);
    w.heatmap.setStep(0);
    w.strip.setIdentity(sliced);
    w.consistency.setIdentity(sliced);
    w.provenance.setIdentity(sliced, 1);
    const row = identity.record.x_windows_rows[windowIndex];
    const rowText = row === undefined || row < 0 ? "raw broadcast · not in X_windows" : `X_windows row ${row}`;
    w.caption.textContent = `${identity.label} · #${identity.record.sender_pseudo} · window ${windowIndex} of ${identity.windowCount - 1} · ${rowText}`;
    w.heatmapSurface.invalidate();
    w.stripSurface.invalidate();
    this.renderSide(side);
  }

  /** Call after the container becomes visible (its canvases may have measured 0×0 while hidden). */
  remeasure(): void {
    this.truth.heatmapSurface.remeasure();
    this.truth.stripSurface.remeasure();
    this.attack.heatmapSurface.remeasure();
    this.attack.stripSurface.remeasure();
    this.renderSide("truth");
    this.renderSide("attack");
  }

  private static sideMarkup(): string {
    return `
      <div class="wc-caption"></div>
      <canvas class="wc-heatmap"></canvas>
      <canvas class="wc-strip"></canvas>
      <div class="wc-step-nav">
        <button class="btn" data-nav="-1">◂ step</button>
        <span class="wc-step-readout"></span>
        <button class="btn" data-nav="1">step ▸</button>
      </div>
      <div class="wc-consistency"></div>
      <div class="wc-provenance"></div>`;
  }

  private buildSide(root: HTMLElement, side: Side): SideWidgets {
    const col = root.querySelector<HTMLElement>(`[data-side="${side}"]`)!;
    const heatmapCanvas = col.querySelector<HTMLCanvasElement>(".wc-heatmap")!;
    const stripCanvas = col.querySelector<HTMLCanvasElement>(".wc-strip")!;
    const heatmap = new HeatmapRenderer({ mean: this.dataset.manifest.norm_mean, std: this.dataset.manifest.norm_std });
    const strip = new StripRenderer();
    const heatmapSurface = new CanvasSurface(heatmapCanvas, () => this.renderSide(side));
    const stripSurface = new CanvasSurface(stripCanvas, () => this.renderSide(side));
    heatmapSurface.setLayers([heatmap]);
    stripSurface.setLayers([strip]);
    const consistency = new ConsistencyPanel(col.querySelector<HTMLElement>(".wc-consistency")!);
    const provenance = new ProvenanceStrip(col.querySelector<HTMLElement>(".wc-provenance")!, (l) => this.dataset.colorFor(l));
    const caption = col.querySelector<HTMLElement>(".wc-caption")!;
    const readout = col.querySelector<HTMLElement>(".wc-step-readout")!;
    col.querySelector('[data-nav="-1"]')!.addEventListener("click", () => this.stepBy(side, -1));
    col.querySelector('[data-nav="1"]')!.addEventListener("click", () => this.stepBy(side, 1));
    return { heatmap, heatmapSurface, strip, stripSurface, consistency, provenance, caption, readout };
  }

  private stepBy(side: Side, delta: number): void {
    const win = side === "truth" ? this.truthWindow : this.attackWindow;
    if (!win) return;
    const clamped = Math.max(0, Math.min(win.stepCount - 1, (side === "truth" ? this.truthStep : this.attackStep) + delta));
    if (side === "truth") this.truthStep = clamped;
    else this.attackStep = clamped;
    this.renderSide(side);
  }

  private renderSide(side: Side): void {
    const win = side === "truth" ? this.truthWindow : this.attackWindow;
    const w = side === "truth" ? this.truth : this.attack;
    const step = side === "truth" ? this.truthStep : this.attackStep;
    if (!win) return;
    w.heatmap.setStep(step);
    w.heatmapSurface.render(step);
    w.stripSurface.render(step);
    w.consistency.update(step);
    w.provenance.update(step, 0);
    w.readout.textContent = `step ${step} / ${win.stepCount - 1} within this window`;
  }
}
