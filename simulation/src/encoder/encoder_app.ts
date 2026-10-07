import "./encoder.css";
import { EncoderDataset } from "./core/encoder_dataset";
import { Grouper, type GroupSummary } from "./core/grouping";
import { PredictionStore } from "./core/predictions";
import { DEFAULT_ROUTE, formatRoute, parseRoute, type EncoderRoute } from "./core/route";
import type { WindowMeta } from "./core/types";
import { h } from "./ui/dom";
import { GroupView } from "./ui/group_view";
import { EncoderSidebar } from "./ui/sidebar";
import { SingleView } from "./ui/single_view";

/**
 * The "Encoder input (T = 64)" tab: what the encoder reads, how it is split, where the mask is.
 * State lives in the URL hash (`#encoder?…`) so the presenter can jump to a prepared view.
 */
export class EncoderApp {
  private route: EncoderRoute = { ...DEFAULT_ROUTE };
  private readonly grouper: Grouper;
  private readonly sidebar: EncoderSidebar;
  private readonly single: SingleView;
  private readonly group: GroupView;
  private readonly stage: HTMLElement;
  private list: readonly WindowMeta[] = [];
  private current: WindowMeta | null = null;

  /** Loads the sample + predictions and builds the tab inside `root`. */
  static async mount(root: HTMLElement, baseUrl = "data/encoder/"): Promise<EncoderApp> {
    root.replaceChildren(h("div", { class: "enc-loading" }, "loading encoder-input sample…"));
    try {
      const [ds, preds] = await Promise.all([EncoderDataset.load(baseUrl), PredictionStore.load(baseUrl)]);
      return new EncoderApp(root, ds, preds);
    } catch (err) {
      root.replaceChildren(h("div", { class: "enc-loading error" },
        `failed to load ${baseUrl}: ${err instanceof Error ? err.message : String(err)}`));
      throw err;
    }
  }

  constructor(root: HTMLElement, private readonly ds: EncoderDataset, preds: PredictionStore) {
    this.grouper = new Grouper(ds);
    const side = h("aside", { class: "sidebar enc-sidebar" });
    this.stage = h("main", { class: "enc-stage" });
    const singleRoot = h("div", { class: "enc-view" });
    const groupRoot = h("div", { class: "enc-view" });
    this.stage.append(singleRoot, groupRoot);
    root.replaceChildren(h("div", { class: "enc-layout" }, side, this.stage));
    this.sidebar = new EncoderSidebar(side, ds, {
      onRoute: (patch) => this.navigate(patch),
      onCurated: (id) => this.openCurated(id),
      onCopyLink: () => this.copyLink(),
    });
    this.single = new SingleView(singleRoot, ds, preds);
    this.group = new GroupView(groupRoot, ds, this.grouper, (id) => this.navigate({ mode: "single", win: id }));
    window.addEventListener("hashchange", () => this.fromHash());
    window.addEventListener("popstate", () => this.fromHash());
    window.addEventListener("keydown", (e) => this.onKey(e));
    this.fromHash();
  }

  private fromHash(): void {
    const r = parseRoute(location.hash);
    if (!r) return;
    this.route = r;
    this.render();
  }

  private navigate(patch: Partial<EncoderRoute>): void {
    this.route = { ...this.route, ...patch };
    this.render();
  }

  private openCurated(id: string): void {
    const w = this.ds.byId(id);
    if (!w) return;
    this.navigate({ split: w.split, scenario: "all", run: "all", cls: "all", mode: "single", win: id });
  }

  private render(): void {
    const r = this.route;
    const pool = this.ds.filter({
      splits: [r.split],
      scenarios: r.scenario === "all" ? undefined : [r.scenario],
      runs: r.run === "all" ? undefined : [r.run],
      classes: r.cls === "all" || !r.labels ? undefined : [r.cls],
    });
    let groups: GroupSummary[] = [];
    let members: WindowMeta[] = [];
    if (r.mode === "group") {
      groups = this.grouper.groups(r.group, pool);
      if (r.group === "batch") {
        if (!r.key.startsWith("batch:")) r.key = "batch:0";
      } else if (!groups.some((g) => g.key === r.key)) {
        r.key = groups[0]?.key ?? "";
      }
      members = r.key ? this.grouper.members(r.group, r.key, pool) : [];
    }
    this.list = r.mode === "group" ? members : pool;
    // A window opened by id (curated, URL, group card) is shown even if the filter would hide it.
    let w = r.win ? this.ds.byId(r.win) ?? null : null;
    if (r.mode === "single" && !w) w = pool[0] ?? null;
    if (r.mode === "single") r.win = w?.id ?? "";
    this.current = w;

    const hash = formatRoute(r);
    if (location.hash !== hash) history.replaceState(null, "", hash);
    this.sidebar.render({ route: r, pool, groups, list: this.list, current: w });
    const [singleRoot, groupRoot] = [...this.stage.children] as HTMLElement[];
    singleRoot.hidden = r.mode !== "single";
    groupRoot.hidden = r.mode !== "group";
    if (r.mode === "single") {
      this.single.show(w, r.raw, r.labels);
    } else {
      const label = r.group === "batch"
        ? `seed ${r.key.slice(6)} · ${members.length} windows`
        : groups.find((g) => g.key === r.key)?.label ?? r.key;
      this.group.show({ kind: r.group, key: r.key, label, members, pool, raw: r.raw, labels: r.labels });
    }
  }

  private onKey(e: KeyboardEvent): void {
    if (document.body.dataset.tab !== "encoder") return;
    const target = e.target as HTMLElement | null;
    if (target && ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)) return;
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    if (this.list.length === 0) return;
    const i = this.current ? this.list.findIndex((w) => w.id === this.current?.id) : -1;
    const next = e.key === "ArrowDown" ? Math.min(this.list.length - 1, i + 1) : Math.max(0, i - 1);
    e.preventDefault();
    this.navigate({ mode: "single", win: this.list[next].id });
  }

  private async copyLink(): Promise<boolean> {
    const url = location.href;
    try {
      await navigator.clipboard.writeText(url);
      return true;
    } catch {
      return false;
    }
  }
}
