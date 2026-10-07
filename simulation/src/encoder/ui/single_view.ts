import type { EncoderDataset } from "../core/encoder_dataset";
import { WindowGeometry } from "../core/geometry";
import type { PredictionStore } from "../core/predictions";
import type { WindowMeta } from "../core/types";
import { FeatureStrips } from "../render/feature_strips";
import { shortFeature } from "../render/colors";
import { CLAIMED_COLOR, RECEIVER_COLOR, SpatialMap } from "../render/spatial_map";
import { StaticCanvas } from "../render/static_canvas";
import { TensorHeatmap } from "../render/tensor_heatmap";
import { clear, h } from "./dom";
import { classChip } from "./labels";
import { caveatsBanner, detectionPanel, fullDatasetPanel, maskPanel, panel, provenancePanel, rangeCaption } from "./panels";

const DEFAULT_STRIPS = ["range", "bearing", "claimed_vel_x", "log_dtau"];
const MAX_STRIPS = 4;

/** Single-window stage: tensor + mask, spatial map, feature strips, provenance, detection, full-data summary. */
export class SingleView {
  private readonly heat: StaticCanvas;
  private readonly map: StaticCanvas;
  private readonly strips: StaticCanvas;
  private selected: number[];
  private current: { w: WindowMeta; raw: boolean; labels: boolean } | null = null;

  constructor(
    private readonly root: HTMLElement,
    private readonly ds: EncoderDataset,
    private readonly preds: PredictionStore,
  ) {
    this.heat = new StaticCanvas(() => undefined, "enc-canvas enc-heat-canvas");
    this.map = new StaticCanvas(() => undefined, "enc-canvas enc-map-canvas");
    this.strips = new StaticCanvas(() => undefined, "enc-canvas enc-strips-canvas");
    this.selected = DEFAULT_STRIPS.map((n) => ds.manifest.features.indexOf(n)).filter((j) => j >= 0);
  }

  show(w: WindowMeta | null, raw: boolean, labels: boolean): void {
    this.current = w ? { w, raw, labels } : null;
    clear(this.root);
    if (!w) {
      this.root.append(caveatsBanner(this.ds), h("div", { class: "enc-empty" }, "No windows match this filter."));
      return;
    }
    const T = this.ds.seqLen;
    const title = h("div", { class: "enc-title" },
      h("span", { class: "enc-title-id" }, w.id),
      labels ? classChip(w.label_name) : null,
      h("span", { class: "enc-dim" }, `${w.split} · ${w.scenario} · link crop ${w.k + 1}/${w.K} · n = ${w.n}`),
    );
    const heatPanel = panel("Encoder input · 64 × 13 + mask", `${w.n} real · ${T - w.n} padding · ${raw ? "raw units" : "normalised"}`, this.heat.canvas);
    heatPanel.classList.add("enc-heat-panel");
    const g = WindowGeometry.of(this.ds, w);
    const legend = h("span", { class: "enc-legend" },
      swatch(CLAIMED_COLOR), "claimed sender position", swatch(RECEIVER_COLOR), "receiver position", h("span", { class: "enc-line-swatch" }), "range");
    const mapPanel = panel("Spatial view · SUMO metres", h("span", { class: "panel-caption" }, rangeCaption(g.range)), h("div", { class: "enc-map-legend" }, legend), this.map.canvas);
    mapPanel.classList.add("enc-map-panel");
    const stripsPanel = panel("Feature strips · real rows only", this.stripPicker(), this.strips.canvas);
    stripsPanel.classList.add("enc-strips-panel");

    this.root.append(
      caveatsBanner(this.ds),
      title,
      h("div", { class: "enc-main" }, heatPanel, h("div", { class: "enc-right" }, mapPanel, stripsPanel)),
      h("div", { class: "enc-cards" },
        maskPanel(this.ds, w),
        provenancePanel(this.ds, w, labels),
        detectionPanel(this.preds, w, labels)),
      fullDatasetPanel(this.ds, labels),
    );

    const matrix = this.ds.matrix(w, raw);
    const mask = this.ds.mask(w);
    const m = this.ds.manifest;
    this.heat.setDraw((ctx, cw, ch) => TensorHeatmap.draw(ctx, cw, ch, { matrix, mask, features: m.features, units: m.units, raw }));
    this.map.setDraw((ctx, cw, ch) => SpatialMap.draw(ctx, cw, ch, g));
    this.drawStrips();
  }

  private drawStrips(): void {
    if (!this.current) return;
    const { w, raw } = this.current;
    const rows = raw ? this.ds.rawRows(w) : this.ds.realRows(w);
    const m = this.ds.manifest;
    const selected = [...this.selected];
    this.strips.setDraw((ctx, cw, ch) => FeatureStrips.draw(ctx, cw, ch, {
      rows, n: w.n, featureCount: this.ds.nFeatures, selected, names: m.features, units: m.units, raw,
    }));
  }

  private stripPicker(): HTMLElement {
    const box = h("div", { class: "enc-picker" });
    this.ds.manifest.features.forEach((name, j) => {
      const on = this.selected.includes(j);
      const b = h("button", { type: "button", class: on ? "on" : "", title: name }, shortFeature(name));
      b.addEventListener("click", () => {
        if (this.selected.includes(j)) this.selected = this.selected.filter((x) => x !== j);
        else this.selected = [...this.selected, j].slice(-MAX_STRIPS);
        box.querySelectorAll("button").forEach((btn, k) => btn.classList.toggle("on", this.selected.includes(k)));
        this.drawStrips();
      });
      box.append(b);
    });
    return box;
  }
}

function swatch(color: string): HTMLElement {
  const s = h("span", { class: "enc-swatch" });
  s.style.background = color;
  return s;
}
