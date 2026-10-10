import { clear, fmtInt, h } from "../../encoder/ui/dom";
import type { DetectionDataset, FlagFilter } from "../core/detection_dataset";
import { friendlyClass } from "../core/plain";
import type { DetectionRoute } from "../core/route";
import type { DetectionWindow } from "../core/types";
import { detClassChip } from "./panels";

export interface DetectionSidebarCallbacks {
  onRoute(patch: Partial<DetectionRoute>): void;
  onCopyLink(): Promise<boolean>;
  onRandom(): void;
  /** Number of a window in the shuffled list. */
  quizNumber(w: DetectionWindow): number;
}

/** Running tally of the user's test rounds (model right / answers shown). */
export interface QuizTally {
  right: number;
  total: number;
}

const FLAG_OPTIONS: readonly { value: FlagFilter; label: string }[] = [
  { value: "all", label: "all" },
  { value: "flagged", label: "alarm" },
  { value: "unflagged", label: "no alarm" },
];

const MAX_LIST = 800;

/** Left column: the mode switch, then either the test controls or the explore filters, then the window list. */
export class DetectionSidebar {
  constructor(
    private readonly root: HTMLElement,
    private readonly ds: DetectionDataset,
    private readonly cb: DetectionSidebarCallbacks,
  ) {}

  render(r: DetectionRoute, list: readonly DetectionWindow[], current: DetectionWindow | null, tally: QuizTally): void {
    const scrollTop = this.root.querySelector<HTMLElement>(".ident-list")?.scrollTop ?? 0;
    clear(this.root);
    this.root.append(this.modeBlock(r, tally), this.listBlock(list, current, r.blind));
    if (!r.blind) this.root.append(this.linkBlock());
    const listEl = this.root.querySelector<HTMLElement>(".ident-list");
    if (listEl) {
      listEl.scrollTop = scrollTop;
      listEl.querySelector<HTMLElement>(".ident.on")?.scrollIntoView({ block: "nearest" });
    }
  }

  private modeBlock(r: DetectionRoute, tally: QuizTally): HTMLElement {
    const seg = h("div", { class: "seg det-mode" });
    for (const [blind, label] of [[true, "Test yourself"], [false, "Explore"]] as const) {
      const b = h("button", { type: "button", class: r.blind === blind ? "on" : "" }, label);
      b.addEventListener("click", () =>
        this.cb.onRoute(blind ? { blind, win: "" } : { blind, cls: "all", flag: "all", win: "" }));
      seg.append(b);
    }
    const block = h("section", { class: "side-block" }, seg);
    if (r.blind) {
      const random = h("button", { type: "button", class: "btn det-big-btn" }, "🎲  Random window");
      random.addEventListener("click", () => this.cb.onRandom());
      block.append(random,
        h("div", { class: "det-tally" }, tally.total
          ? h("span", {}, "Model right on ", h("b", {}, `${tally.right} of ${tally.total}`), " windows so far")
          : h("span", {}, "Pick a window, guess, then let the model decide.")));
    } else {
      block.append(this.filters(r));
    }
    return block;
  }

  private filters(r: DetectionRoute): HTMLElement {
    const clsSeg = h("div", { class: "seg det-wrap" });
    for (const c of ["all", ...this.ds.classes()]) {
      const n = this.ds.query({ cls: c }).length;
      const b = h("button", { type: "button", class: r.cls === c ? "on" : "" },
        c === "all" ? "all" : friendlyClass(c), h("span", { class: "n" }, String(n)));
      b.addEventListener("click", () => this.cb.onRoute({ cls: c, win: "" }));
      clsSeg.append(b);
    }
    const flagSeg = h("div", { class: "seg" });
    for (const o of FLAG_OPTIONS) {
      const b = h("button", { type: "button", class: r.flag === o.value ? "on" : "" }, o.label);
      b.addEventListener("click", () => this.cb.onRoute({ flag: o.value, win: "" }));
      flagSeg.append(b);
    }
    return h("div", { class: "det-filters" }, h("label", {}, "Class"), clsSeg, h("label", {}, "Model's call"), flagSeg);
  }

  private listBlock(list: readonly DetectionWindow[], current: DetectionWindow | null, blind: boolean): HTMLElement {
    const el = h("div", { class: "ident-list" });
    for (const w of list.slice(0, MAX_LIST)) {
      const on = current?.id === w.id ? " on" : "";
      const msgs = `${w.n} msg${w.n === 1 ? "" : "s"}`;
      // Test mode lists only the number and length: no score, alarm, id or class.
      const item = blind
        ? h("button", { type: "button", class: `ident${on}` },
            h("span", { class: "enc-li-main" }, `Window #${this.cb.quizNumber(w)}`), h("span", { class: "meta" }, msgs))
        : h("button", { type: "button", class: `ident${on}`, title: w.id },
            h("span", { class: `det-dot ${w.flag ? "on" : "off"}`, title: w.flag ? "alarm" : "no alarm" }),
            h("span", { class: "enc-li-main" }, `Window #${this.cb.quizNumber(w)}`),
            h("span", { class: "meta" }, msgs),
            detClassChip(w.class));
      item.addEventListener("click", () => this.cb.onRoute({ win: w.id }));
      el.append(item);
    }
    const count = list.length > MAX_LIST ? `${MAX_LIST} of ${fmtInt(list.length)}` : fmtInt(list.length);
    return h("section", { class: "side-block grow" },
      h("h2", {}, "Windows", h("span", { class: "count" }, count)),
      el,
      h("p", { class: "hint" }, "↑ / ↓ to move through the list"));
  }

  private linkBlock(): HTMLElement {
    const copy = h("button", { type: "button", class: "btn" }, "Copy link to this view");
    copy.addEventListener("click", () => {
      void this.cb.onCopyLink().then((ok) => {
        copy.textContent = ok ? "Link copied" : "Copy failed — use the address bar";
        setTimeout(() => (copy.textContent = "Copy link to this view"), 1600);
      });
    });
    return h("section", { class: "side-block" }, copy);
  }
}
