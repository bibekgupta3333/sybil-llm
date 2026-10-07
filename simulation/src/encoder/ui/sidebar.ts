import type { EncoderDataset } from "../core/encoder_dataset";
import type { GroupKind, GroupSummary } from "../core/grouping";
import type { EncoderRoute } from "../core/route";
import { CLASS_NAMES, SPLITS, type WindowMeta } from "../core/types";
import { clear, fmtInt, h, shortId } from "./dom";
import { EVAL_ONLY_BADGE, SPLIT_LABEL, classChip } from "./labels";

export interface SidebarModel {
  route: EncoderRoute;
  pool: readonly WindowMeta[];
  groups: readonly GroupSummary[];
  /** Windows listed under "Windows" (pool in single mode, group members in group mode). */
  list: readonly WindowMeta[];
  current: WindowMeta | null;
}

export interface SidebarCallbacks {
  onRoute(patch: Partial<EncoderRoute>): void;
  onCurated(windowId: string): void;
  onCopyLink(): Promise<boolean>;
}

const GROUP_KINDS: readonly { kind: GroupKind; label: string; title: string }[] = [
  { kind: "link", label: "link", title: "One (receiver, sender pseudonym) link: its crops of 64 in order" },
  { kind: "sender", label: "sender vehicle", title: "All windows of one physical sender. For GridSybil ghosts this uses the true sender id — privileged information (F13)." },
  { kind: "receiver", label: "receiver", title: "All windows heard by one receiver file (within the sample)" },
  { kind: "batch", label: "batch of 32", title: "A seeded random batch of 32 windows from the current filter" },
];

const MAX_LIST = 600;

/** Left column of the encoder tab: filters, mode, groups, window list, presenter shortcuts, toggles. */
export class EncoderSidebar {
  constructor(
    private readonly root: HTMLElement,
    private readonly ds: EncoderDataset,
    private readonly cb: SidebarCallbacks,
  ) {}

  render(m: SidebarModel): void {
    const scrollTops = [...this.root.querySelectorAll<HTMLElement>(".ident-list")].map((el) => el.scrollTop);
    clear(this.root);
    const r = m.route;
    this.root.append(
      this.filtersBlock(r),
      this.viewBlock(r, m),
      this.windowsBlock(r, m),
      this.curatedBlock(r, m.current),
      this.togglesBlock(r),
    );
    const lists = [...this.root.querySelectorAll<HTMLElement>(".ident-list")];
    lists.forEach((el, i) => {
      el.scrollTop = scrollTops[i] ?? 0;
      el.querySelector<HTMLElement>(".ident.on")?.scrollIntoView({ block: "nearest" });
    });
  }

  private filtersBlock(r: EncoderRoute): HTMLElement {
    const splitSeg = h("div", { class: "seg" });
    for (const s of SPLITS) {
      const n = this.ds.filter({ splits: [s] }).length;
      const b = h("button", { type: "button", class: r.split === s ? "on" : "", title: `${n} sampled windows` },
        SPLIT_LABEL[s], h("span", { class: "n" }, String(n)));
      b.addEventListener("click", () => this.cb.onRoute({ split: s, run: "all", key: "", win: "" }));
      splitSeg.append(b);
    }
    const scenSeg = h("div", { class: "seg" });
    for (const [value, label] of [["all", "all"], ["0709", "0709"], ["1416", "1416"]] as const) {
      const b = h("button", { type: "button", class: r.scenario === value ? "on" : "" }, label);
      b.addEventListener("click", () => this.cb.onRoute({ scenario: value, run: "all", key: "", win: "" }));
      scenSeg.append(b);
    }
    const runs = this.ds.runs(r.split, r.scenario === "all" ? undefined : r.scenario);
    const runSel = h("select", { "aria-label": "run" }, h("option", { value: "all" }, `all runs (${runs.length})`));
    for (const run of runs) runSel.append(h("option", { value: run }, run));
    runSel.value = runs.includes(r.run) ? r.run : "all";
    runSel.addEventListener("change", () => this.cb.onRoute({ run: runSel.value, key: "", win: "" }));

    const clsSeg = h("div", { class: "seg" });
    for (const c of ["all", ...CLASS_NAMES] as const) {
      const b = h("button", { type: "button", class: r.cls === c ? "on" : "", disabled: !r.labels }, c);
      b.addEventListener("click", () => this.cb.onRoute({ cls: c, key: "", win: "" }));
      clsSeg.append(b);
    }
    return h("section", { class: "side-block" },
      h("h2", {}, "Split", h("span", { class: "count" }, "sampled windows")),
      splitSeg,
      h("label", {}, "Scenario group"),
      scenSeg,
      h("label", {}, "Run", runSel),
      h("label", { title: EVAL_ONLY_BADGE }, "Class filter ", h("span", { class: "enc-eval-tag" }, "labels: evaluation only")),
      clsSeg,
    );
  }

  private viewBlock(r: EncoderRoute, m: SidebarModel): HTMLElement {
    const modeSeg = h("div", { class: "seg" });
    for (const mode of ["single", "group"] as const) {
      const b = h("button", { type: "button", class: r.mode === mode ? "on" : "" }, mode === "single" ? "Single window" : "Group");
      b.addEventListener("click", () => this.cb.onRoute({ mode }));
      modeSeg.append(b);
    }
    const block = h("section", { class: "side-block" }, h("h2", {}, "View"), modeSeg);
    if (r.mode !== "group") return block;

    const kindSeg = h("div", { class: "seg enc-kind" });
    for (const g of GROUP_KINDS) {
      const b = h("button", { type: "button", class: r.group === g.kind ? "on" : "", title: g.title }, g.label);
      b.addEventListener("click", () => this.cb.onRoute({ group: g.kind, key: "" }));
      kindSeg.append(b);
    }
    block.append(h("label", {}, "Group by"), kindSeg);
    if (r.group === "sender") {
      block.append(h("p", { class: "hint enc-warn-hint" },
        "Privileged grouping: GridSybil ghosts all broadcast under pseudonym 1 (F13); grouping them by true sender uses simulator ground truth the receiver never has."));
    }
    if (r.group === "batch") {
      const seed = Number(r.key.startsWith("batch:") ? r.key.slice(6) : 0) || 0;
      const reshuffle = h("button", { type: "button", class: "btn" }, `Reshuffle (seed ${seed} → ${seed + 1})`);
      reshuffle.addEventListener("click", () => this.cb.onRoute({ key: `batch:${seed + 1}`, win: "" }));
      block.append(h("p", { class: "hint" }, `Seeded random batch of 32 from ${fmtInt(m.pool.length)} filtered windows (seed ${seed}).`), reshuffle);
      return block;
    }
    const list = h("div", { class: "ident-list enc-group-list" });
    for (const g of m.groups.slice(0, MAX_LIST)) {
      const on = g.key === r.key;
      const item = h("button", { type: "button", class: `ident${on ? " on" : ""}`, title: g.key },
        h("span", { class: "enc-li-main" }, g.label),
        h("span", { class: "meta" }, `${g.size} win`),
        r.labels && g.classes.GridSybil > 0 ? classChip("GridSybil") : null,
        r.labels && g.classes.Benign > 0 ? classChip("Benign") : null,
      );
      item.addEventListener("click", () => this.cb.onRoute({ key: g.key, win: "" }));
      list.append(item);
    }
    block.append(
      h("h2", { class: "enc-sub-h" }, "Groups", h("span", { class: "count" }, m.groups.length > MAX_LIST ? `${MAX_LIST} of ${m.groups.length}` : String(m.groups.length))),
      list,
    );
    return block;
  }

  private windowsBlock(r: EncoderRoute, m: SidebarModel): HTMLElement {
    const list = h("div", { class: "ident-list" });
    for (const w of m.list.slice(0, MAX_LIST)) {
      const on = m.current?.id === w.id && r.mode === "single";
      const item = h("button", { type: "button", class: `ident${on ? " on" : ""}`, title: w.id },
        h("span", { class: "enc-li-main" }, shortId(w.id, 30)),
        h("span", { class: "meta" }, `n=${w.n} · ${w.k + 1}/${w.K}`),
        r.labels ? classChip(w.label_name) : null,
      );
      item.addEventListener("click", () => this.cb.onRoute({ mode: "single", win: w.id }));
      list.append(item);
    }
    const count = m.list.length > MAX_LIST ? `${MAX_LIST} of ${fmtInt(m.list.length)}` : fmtInt(m.list.length);
    return h("section", { class: "side-block grow" },
      h("h2", {}, r.mode === "group" ? "Windows in group" : "Windows", h("span", { class: "count" }, count)),
      list,
      m.list.length > MAX_LIST ? h("p", { class: "hint" }, "Narrow with run / scenario to list the rest.") : null,
    );
  }

  private curatedBlock(r: EncoderRoute, current: WindowMeta | null): HTMLElement {
    const block = h("section", { class: "side-block" },
      h("h2", {}, "Presenter shortcuts", h("span", { class: "count" }, "illustrative")));
    for (const c of this.ds.manifest.curated) {
      const exists = !!this.ds.byId(c.window_id);
      const on = r.mode === "single" && current?.id === c.window_id;
      const b = h("button", { type: "button", class: `enc-curated${on ? " on" : ""}`, disabled: !exists, title: `rule: ${c.rule}` },
        h("span", { class: "enc-curated-name" }, c.name),
        h("span", { class: "enc-badge" }, "illustrative"),
        h("span", { class: "enc-curated-rule" }, `rule: ${c.rule}`),
      );
      b.addEventListener("click", () => this.cb.onCurated(c.window_id));
      block.append(b);
    }
    if (this.ds.manifest.curated.length === 0) block.append(h("p", { class: "hint" }, "No curated windows in the manifest."));
    return block;
  }

  private togglesBlock(r: EncoderRoute): HTMLElement {
    const raw = h("input", { type: "checkbox" });
    raw.checked = r.raw;
    raw.addEventListener("change", () => this.cb.onRoute({ raw: raw.checked }));
    const labels = h("input", { type: "checkbox" });
    labels.checked = r.labels;
    labels.addEventListener("change", () => this.cb.onRoute(labels.checked ? { labels: true } : { labels: false, cls: "all" }));
    const copy = h("button", { type: "button", class: "btn" }, "Copy link to this view");
    copy.addEventListener("click", () => {
      void this.cb.onCopyLink().then((ok) => {
        copy.textContent = ok ? "Link copied" : "Copy failed — use the address bar";
        setTimeout(() => (copy.textContent = "Copy link to this view"), 1600);
      });
    });
    return h("section", { class: "side-block" },
      h("h2", {}, "Display"),
      h("label", { class: "check" }, raw, "Raw units (default: normalised, what the encoder sees)"),
      h("label", { class: "check" }, labels, "Show labels ", h("span", { class: "enc-eval-tag" }, "evaluation only")),
      copy,
      h("p", { class: "hint" }, "Keys: ↑ / ↓ previous / next window in the list."),
    );
  }
}
