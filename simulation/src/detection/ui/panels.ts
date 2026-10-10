import { sequential, shortFeature } from "../../encoder/render/colors";
import { SpatialMap } from "../../encoder/render/spatial_map";
import { StaticCanvas } from "../../encoder/render/static_canvas";
import { fmtInt, h } from "../../encoder/ui/dom";
import { panel } from "../../encoder/ui/panels";
import { fmtMetric, type DetectionDataset } from "../core/detection_dataset";
import { sharedRange } from "../core/histogram";
import { beatsLengthRule, fmt, friendlyClass, outcomePhrase, pct } from "../core/plain";
import { modelIsRight, modelSaysAttack, outcome, type QuizStage } from "../core/quiz";
import { classColor, type DetectionWindow } from "../core/types";
import { ScoreHistogram } from "../render/score_histogram";

/** Class chip in the class colour, with a readable name. */
export function detClassChip(cls: string): HTMLElement {
  const chip = h("span", { class: "chip", title: cls }, friendlyClass(cls));
  chip.style.setProperty("--c", classColor(cls));
  return chip;
}

const kv = (k: string, v: Node | string): HTMLElement[] => [h("dt", {}, k), h("dd", {}, v)];

/** A collapsed section: the title is always visible, the body opens on click. */
function fold(title: string, hint: string, ...body: (Node | null)[]): HTMLElement {
  return h("details", { class: "panel enc-panel det-fold" },
    h("summary", {}, h("span", { class: "panel-title" }, title), h("span", { class: "panel-caption" }, hint)),
    ...body);
}

// ---------------------------------------------------------------------------------------------------------------------
// The selected window: one card, three steps (look → the model decides → the answer).

/** What the window card shows: the test number, how far the round has gone and the buttons' actions. */
export interface QuizView {
  /** Position in the shuffled list (the sample index would give the class away). */
  number: number;
  /** In Explore mode every step is shown at once ("revealed"). */
  stage: QuizStage;
  blind: boolean;
  onDetect(): void;
  onReveal(): void;
  onNext(): void;
}

export class DetailPanel {
  readonly el: HTMLElement;
  private readonly map: StaticCanvas;
  private readonly steps: HTMLElement;
  private readonly body: HTMLElement;
  private readonly raw: HTMLElement;
  private readonly title: HTMLElement;
  private current: DetectionWindow | null = null;

  constructor(private readonly ds: DetectionDataset) {
    this.map = new StaticCanvas((ctx, w, hgt) => {
      const p = this.current ? this.ds.positions(this.current) : null;
      SpatialMap.draw(ctx, w, hgt, p);
    }, "enc-canvas det-map-canvas");
    this.title = h("span", { class: "panel-title" });
    this.steps = h("ol", { class: "det-steps" });
    this.body = h("div", { class: "det-card-body" });
    this.raw = h("div", { class: "det-rows" });
    this.el = h("section", { class: "panel enc-panel det-detail" },
      h("div", { class: "panel-head" }, this.title, this.steps),
      h("div", { class: "det-detail-grid" },
        this.body,
        h("div", { class: "det-map-wrap" },
          h("div", { class: "enc-map-legend" }, h("span", { class: "enc-legend" },
            h("span", { class: "enc-swatch", style: "background: var(--orange)" }), "where the sender says it is",
            h("span", { class: "enc-swatch", style: "background: var(--blue)" }), "where the receiver is")),
          this.map.canvas)),
      this.raw,
    );
  }

  show(w: DetectionWindow | null, view: QuizView): void {
    this.current = w;
    this.body.replaceChildren();
    this.raw.replaceChildren();
    if (!w) {
      this.title.textContent = "No window";
      this.steps.replaceChildren();
      this.body.append(h("div", { class: "enc-empty" }, "No window matches the filter."));
      this.map.redraw();
      return;
    }
    const theta = this.ds.manifest.method.theta;
    const s = view.stage;
    this.title.textContent = view.blind ? `Window #${view.number}` : `Window #${view.number} · ${friendlyClass(w.class)}`;
    this.steps.replaceChildren(
      ...["Look", "Model decides", "Answer"].map((label, k) =>
        h("li", { class: k <= ["hidden", "detected", "revealed"].indexOf(s) ? "on" : "" }, label)),
    );

    // Step 1: the window itself.
    this.body.append(h("p", { class: "det-lead" },
      `${w.n} message${w.n === 1 ? "" : "s"} one receiver heard from one sender ID. `,
      view.blind && s === "hidden" ? "What do you think it is? Then let the model decide." : ""));
    if (s === "hidden") {
      this.body.append(this.button("Let the model decide", view.onDetect, true));
    } else {
      // Step 2: the model's call.
      const attack = modelSaysAttack(w);
      this.body.append(
        h("div", { class: `det-verdict ${attack ? "attack" : "normal"}` },
          attack ? "Model says: ATTACK" : "Model says: normal",
          h("span", { class: "det-verdict-sub" }, attack ? "it looks unusual" : "it looks like the windows it learned from")),
        scoreMeter(w.score, theta, this.ds.scoreMax()),
      );
      if (s === "detected") {
        this.body.append(this.button("Show the answer", view.onReveal, true));
      } else {
        // Step 3: the answer.
        const right = modelIsRight(w);
        this.body.append(
          h("div", { class: `det-verdict ${right ? "right" : "wrong"}` },
            right ? "✓ The model was right" : "✗ The model was wrong",
            h("span", { class: "det-verdict-sub" }, "It was ", detClassChip(w.class), ` — ${outcomePhrase(outcome(w))}`)),
          h("details", { class: "det-tech" }, h("summary", {}, "Technical details"),
            h("dl", { class: "enc-kv" },
              ...kv("score", String(w.score)),
              ...kv("threshold", String(theta)),
              ...kv("rank", `${w.rank_pct}% of normal-looking training windows score lower`),
              ...kv("scenario", w.scenario),
              ...kv("window id", h("code", { class: "det-id" }, w.id)))),
        );
      }
      if (view.blind) this.body.append(this.button("Next window", view.onNext, s === "revealed"));
    }
    this.raw.append(h("details", { class: "det-tech" },
      h("summary", {}, `Raw message values (${w.n} × ${this.ds.nFeatures})`), featureTable(this.ds, w)));
    this.map.redraw();
  }

  private button(label: string, action: () => void, primary: boolean): HTMLElement {
    const b = h("button", { type: "button", class: `btn det-quiz-btn${primary ? " primary" : ""}` }, label);
    b.addEventListener("click", action);
    return b;
  }
}

/** A bar from 0 to the highest score with the threshold mark and this window's score. */
function scoreMeter(score: number, theta: number, max: number): HTMLElement {
  const top = Math.max(max, theta * 1.5, 1e-9);
  const at = (v: number): string => `${Math.min(100, (v / top) * 100).toFixed(1)}%`;
  return h("div", { class: "det-meter", title: `score ${score} · threshold ${theta}` },
    h("div", { class: "det-meter-bar" },
      h("div", { class: "det-meter-fill", style: `width: ${at(score)}` }),
      h("div", { class: "det-meter-theta", style: `left: ${at(theta)}` })),
    h("div", { class: "det-meter-labels" },
      h("span", {}, `unusualness ${fmt(score)}`),
      h("span", {}, `alarm above ${fmt(theta)}`)),
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// Overall results: one small table in plain words.

export function resultsPanel(ds: DetectionDataset): HTMLElement {
  const which = ds.metricsRows("full_test_1416") ? "full_test_1416" : "sample";
  const rows = ds.metricsRows(which) ?? [];
  const benign = rows.find((r) => r.cls === "Benign")?.metrics;
  const fpr = ds.manifest.metrics.benign_fpr_at_theta ?? benign?.flag_rate;
  const body = h("tbody");
  for (const r of rows) {
    if (r.cls === "Benign") continue;
    const beats = beatsLengthRule(r.metrics);
    body.append(h("tr", {},
      h("td", {}, detClassChip(r.cls)),
      h("td", {}, pct(r.metrics?.flag_rate)),
      h("td", { title: String(r.metrics?.auroc_vs_benign ?? "") }, fmt(r.metrics?.auroc_vs_benign, 2)),
      h("td", { title: String(r.metrics?.length_baseline_auroc ?? "") }, fmt(r.metrics?.length_baseline_auroc, 2)),
      h("td", { class: beats ? "det-yes" : "det-no" }, beats === null ? "—" : beats ? "✓ yes" : "✗ no"),
    ));
  }
  const n = rows.reduce((acc, r) => acc + (typeof r.metrics?.n === "number" ? r.metrics.n : 0), 0);
  return panel("How well does it work?", `${fmtInt(n)} test windows`,
    h("div", { class: "enc-body" },
      h("table", { class: "enc-table det-results" },
        h("thead", {}, h("tr", {},
          h("th", {}, "attack"),
          h("th", { title: "share of this attack's windows the model raises an alarm for" }, "caught"),
          h("th", { title: "AUROC against normal traffic: 1 = perfect, 0.5 = coin flip" }, "model"),
          h("th", { title: "AUROC of the simple rule 'short window = attack'" }, "length rule"),
          h("th", {}, "model better?"))),
        body),
      h("p", { class: "det-note" },
        h("b", {}, `False alarms on normal traffic: ${pct(fpr)}.`),
        " “model” and “length rule” are AUROC scores: 1 = perfect, 0.5 = coin flip. ",
        "Where the length rule wins, the model has not learned more than “short window = attack”."),
    ));
}

// ---------------------------------------------------------------------------------------------------------------------
// Folded away: score chart and how-it-works.

/** Score histogram per class on one axis, the alarm threshold as a dashed line, the selected window as a marker. */
export class HistogramPanel {
  readonly el: HTMLElement;
  private readonly canvas: StaticCanvas;
  private marker: DetectionWindow | null = null;

  constructor(private readonly ds: DetectionDataset) {
    this.canvas = new StaticCanvas((ctx, w, hgt) => this.draw(ctx, w, hgt), "enc-canvas det-hist-canvas");
    this.canvas.canvas.style.height = `${Math.max(140, 44 * ds.classes().length + 34)}px`;
    this.el = fold("Score chart", "how unusual each class looks · dashed line = alarm threshold", this.canvas.canvas);
    this.el.addEventListener("toggle", () => this.canvas.redraw());
  }

  setMarker(w: DetectionWindow | null): void {
    this.marker = w;
    this.canvas.redraw();
  }

  private draw(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    const theta = this.ds.manifest.method.theta;
    const rows = this.ds.classes().map((cls) => ({ cls, scores: this.ds.scores(cls) }));
    ScoreHistogram.draw(ctx, width, height, {
      rows,
      range: sharedRange(this.ds.scores(), theta),
      theta,
      nBins: 40,
      marker: this.marker ? { cls: this.marker.class, score: this.marker.score } : null,
    });
  }
}

/** How it works, in short, plus every technical detail and the exact numbers (exporter values, verbatim). */
export function aboutPanel(ds: DetectionDataset): HTMLElement {
  const m = ds.manifest;
  const md = m.model;
  const me = m.method;
  return fold("How it works", "model, method, exact numbers, caveats",
    h("div", { class: "enc-body det-about" },
      h("ol", { class: "det-how" },
        h("li", {}, `The trained encoder (${md.run_id}) turns each window into a short summary vector — it never saw labels.`),
        h("li", {}, `A window's "unusualness" = how far it is from its ${me.k} closest neighbours among ${fmtInt(me.bank_size)} training windows.`),
        h("li", {}, `Above the threshold (${fmt(me.theta)}) the model raises an alarm. The threshold also comes from unlabeled data, so it is not tuned.`),
        h("li", {}, `The windows come from the test split of scenario group ${m.scope.group}; the true class is used only to check the answer.`)),
      h("div", { class: "enc-hist-label" }, "Caveats"),
      h("ul", { class: "det-caveats" }, ...m.caveats.map((c) => h("li", {}, c))),
      h("div", { class: "enc-hist-label" }, "Exact numbers per class (full test split of the group, then the shown sample)"),
      exactTable(ds, "full_test_1416"),
      exactTable(ds, "sample"),
      h("div", { class: "enc-hist-label" }, "Model and method"),
      h("dl", { class: "enc-kv" },
        ...kv("checkpoint", `${md.run_id} · ${md.checkpoint} · epoch ${fmtMetric(md.epoch)}`),
        ...kv("sha256", h("code", { class: "det-id" }, md.sha256 ?? "—")),
        ...kv("method", me.name),
        ...kv("threshold rule", me.theta_rule),
        ...kv("scope", `${m.scope.split} · group ${m.scope.group} · ${m.scope.note ?? ""}`),
        ...kv("exported", `${m.created ?? "—"} · git ${m.git_commit ?? "—"} · seed ${me.seed}`)),
    ));
}

const EXACT_COLS: readonly { key: string; label: string }[] = [
  { key: "n", label: "n" },
  { key: "auroc_vs_benign", label: "AUROC vs benign" },
  { key: "length_baseline_auroc", label: "length-only AUROC" },
  { key: "flag_rate", label: "flag rate" },
  { key: "mean_rows", label: "mean rows" },
];

function exactTable(ds: DetectionDataset, which: "sample" | "full_test_1416"): HTMLElement {
  const rows = ds.metricsRows(which);
  if (!rows) return h("p", { class: "enc-dim" }, `${which}: not computed`);
  return h("table", { class: "enc-table det-exact" },
    h("thead", {}, h("tr", {}, h("th", {}, which === "sample" ? "shown sample" : "full test split"),
      ...EXACT_COLS.map((c) => h("th", {}, c.label)))),
    h("tbody", {}, ...rows.map((r) => h("tr", {},
      h("td", {}, r.cls),
      ...EXACT_COLS.map((c) => h("td", {}, fmtMetric(r.metrics?.[c.key]))))))
  );
}

/** n × features table in raw units, each column shaded min → max over the window's rows. */
function featureTable(ds: DetectionDataset, w: DetectionWindow): HTMLElement {
  const f = ds.nFeatures;
  const rows = ds.rows(w);
  const feats = ds.manifest.features;
  const lo = new Array<number>(f).fill(Infinity);
  const hi = new Array<number>(f).fill(-Infinity);
  for (let t = 0; t < w.n; t++) {
    for (let j = 0; j < f; j++) {
      const v = rows[t * f + j];
      if (!Number.isFinite(v)) continue;
      lo[j] = Math.min(lo[j], v);
      hi[j] = Math.max(hi[j], v);
    }
  }
  const table = h("table", { class: "enc-table det-feature-table" },
    h("thead", {}, h("tr", {}, h("th", {}, "t"),
      ...feats.map((name) => h("th", { title: name }, shortFeature(name), h("br"), h("span", { class: "det-unit" }, ds.unit(name) || "·"))))),
  );
  const body = h("tbody");
  for (let t = 0; t < w.n; t++) {
    const tr = h("tr", {}, h("td", {}, String(t)));
    for (let j = 0; j < f; j++) {
      const v = rows[t * f + j];
      const span = hi[j] - lo[j];
      const td = h("td", { title: `${feats[j]} = ${v}` }, fmtValue(v));
      if (Number.isFinite(v) && span > 0) td.style.background = sequential(((v - lo[j]) / span) * 0.6);
      tr.append(td);
    }
    body.append(tr);
  }
  table.append(body);
  return h("div", { class: "enc-body" }, table);
}

function fmtValue(v: number): string {
  if (!Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(3);
}
