import "../encoder/encoder.css";
import "./detection.css";
import { h } from "../encoder/ui/dom";
import { DetectionDataMissing, DetectionDataset } from "./core/detection_dataset";
import { DEFAULT_DETECTION_ROUTE, formatDetectionRoute, parseDetectionRoute, type DetectionRoute } from "./core/route";
import { pickRandom, shuffled, type QuizStage } from "./core/quiz";
import type { DetectionWindow } from "./core/types";
import { DetailPanel, HistogramPanel, aboutPanel, resultsPanel } from "./ui/panels";
import { DetectionSidebar } from "./ui/sidebar";

/**
 * The "Detection (model-all)" tab: label-free kNN anomaly scores of the pretrained encoder on test windows
 * of group 1416. State lives in the URL hash (`#detection?…`).
 */
export class DetectionApp {
  private route: DetectionRoute = { ...DEFAULT_DETECTION_ROUTE };
  private readonly sidebar: DetectionSidebar;
  private readonly histogram: HistogramPanel;
  private readonly detail: DetailPanel;
  private list: readonly DetectionWindow[] = [];
  private current: DetectionWindow | null = null;
  private quizStage: QuizStage = "hidden";
  private readonly revealed = new Set<string>();
  private readonly tally = { right: 0, total: 0 };
  private readonly quizOrder: readonly DetectionWindow[];
  private readonly quizNumbers = new Map<string, number>();

  /** Loads the detection bundle and builds the tab inside `root`; shows how to create it when missing. */
  static async mount(root: HTMLElement, baseUrl = "data/detection/"): Promise<DetectionApp | null> {
    root.replaceChildren(h("div", { class: "enc-loading" }, "loading detection sample…"));
    try {
      return new DetectionApp(root, await DetectionDataset.load(baseUrl));
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (err instanceof DetectionDataMissing) {
        root.replaceChildren(h("div", { class: "enc-empty det-missing" },
          h("p", {}, h("b", {}, "No detection sample yet."), " Run ", h("code", {}, "npm run sim:detect"),
            " from the repo root (it scores test windows with ", h("code", {}, "model-all"), " and writes ",
            h("code", {}, "simulation/public/data/detection/"), "), then reload."),
          h("p", { class: "enc-src" }, msg)));
        return null;
      }
      root.replaceChildren(h("div", { class: "enc-loading error" }, `failed to load ${baseUrl}: ${msg}`));
      throw err;
    }
  }

  constructor(root: HTMLElement, private readonly ds: DetectionDataset) {
    this.quizOrder = shuffled(ds.windows, 0);
    this.quizOrder.forEach((w, k) => this.quizNumbers.set(w.id, k + 1));
    const side = h("aside", { class: "sidebar enc-sidebar" });
    const stage = h("main", { class: "enc-stage" });
    this.histogram = new HistogramPanel(ds);
    this.detail = new DetailPanel(ds);
    stage.append(h("div", { class: "enc-view det-view" }, this.detail.el, resultsPanel(ds), this.histogram.el, aboutPanel(ds)));
    root.replaceChildren(h("div", { class: "enc-layout" }, side, stage));
    this.sidebar = new DetectionSidebar(side, ds, {
      onRoute: (patch) => this.navigate(patch),
      onCopyLink: () => this.copyLink(),
      onRandom: () => this.pickRandom(),
      quizNumber: (w) => this.quizNumbers.get(w.id) ?? 0,
    });
    window.addEventListener("hashchange", () => this.fromHash());
    window.addEventListener("popstate", () => this.fromHash());
    window.addEventListener("keydown", (e) => this.onKey(e));
    this.fromHash();
  }

  private fromHash(): void {
    const r = parseDetectionRoute(location.hash);
    if (!r) return;
    this.route = r;
    this.render();
  }

  private navigate(patch: Partial<DetectionRoute>): void {
    this.route = { ...this.route, ...patch };
    this.render();
  }

  private render(): void {
    const r = this.route;
    this.list = r.blind ? this.quizOrder : this.ds.query({ cls: r.cls, flag: r.flag, sort: r.sort });
    // A window opened by id (URL) is shown even if the filter hides it. In test mode the URL holds only the
    // test number (`win=q12`): the window id names the scenario and attack code, which would give the class away.
    let w = r.win ? this.byKey(r.win) : null;
    if (!w) w = this.list[0] ?? null;
    r.win = w ? (r.blind ? `q${this.quizNumbers.get(w.id) ?? 0}` : w.id) : "";
    if (w?.id !== this.current?.id) this.quizStage = w && this.revealed.has(w.id) ? "revealed" : "hidden";
    this.current = w;
    const hash = formatDetectionRoute(r);
    if (location.hash !== hash) history.replaceState(null, "", hash);
    this.sidebar.render(r, this.list, w, this.tally);
    // The histogram marker is coloured by class, so in test mode it appears only after the reveal.
    this.histogram.setMarker(r.blind && this.quizStage !== "revealed" ? null : w);
    // Explore shows every step at once; Test yourself walks through look → the model decides → the answer.
    this.detail.show(w, {
      number: w ? this.quizNumbers.get(w.id) ?? 0 : 0,
      stage: r.blind ? this.quizStage : "revealed",
      blind: r.blind,
      onDetect: () => this.setStage("detected"),
      onReveal: () => this.reveal(),
      onNext: () => this.pickRandom(),
    });
  }

  /** A window from a URL key: a window id, or `q<n>` = test number n. */
  private byKey(key: string): DetectionWindow | null {
    const m = /^q(\d+)$/.exec(key);
    if (m) return this.quizOrder[Number(m[1]) - 1] ?? null;
    return this.ds.byId(key) ?? null;
  }

  private setStage(stage: QuizStage): void {
    this.quizStage = stage;
    this.render();
  }

  private reveal(): void {
    const w = this.current;
    if (w && !this.revealed.has(w.id)) {
      this.revealed.add(w.id);
      this.tally.total += 1;
      if (w.flag === (w.label === 1)) this.tally.right += 1;
    }
    this.setStage("revealed");
  }

  private pickRandom(): void {
    const unseen = this.list.filter((w) => !this.revealed.has(w.id) && w.id !== this.current?.id);
    const w = pickRandom(unseen.length ? unseen : this.list);
    if (w) this.navigate({ win: w.id });
  }

  private onKey(e: KeyboardEvent): void {
    if (document.body.dataset.tab !== "detection") return;
    const target = e.target as HTMLElement | null;
    if (target && ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)) return;
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    if (this.list.length === 0) return;
    const i = this.current ? this.list.findIndex((w) => w.id === this.current?.id) : -1;
    const next = e.key === "ArrowDown" ? Math.min(this.list.length - 1, i + 1) : Math.max(0, i - 1);
    e.preventDefault();
    this.navigate({ win: this.list[next].id });
  }

  private async copyLink(): Promise<boolean> {
    try {
      await navigator.clipboard.writeText(location.href);
      return true;
    } catch {
      return false;
    }
  }
}
