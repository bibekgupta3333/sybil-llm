import type { EncoderDataset } from "../core/encoder_dataset";
import { Grouper, type GroupKind } from "../core/grouping";
import { GroupStats, type GroupStatsResult } from "../core/stats";
import type { WindowMeta } from "../core/types";
import { LengthHistogram } from "../render/length_histogram";
import { StaticCanvas } from "../render/static_canvas";
import { TensorHeatmap } from "../render/tensor_heatmap";
import { clear, fmtInt, fmtPct, h, shortId } from "./dom";
import { EVAL_ONLY_BADGE, classChip, isSharedGhostPseudonym } from "./labels";
import { caveatsBanner, fullDatasetPanel, panel } from "./panels";

const MAX_CARDS = 32;
const STATS_BATCH = 256;

export interface GroupViewModel {
  kind: GroupKind;
  key: string;
  label: string;
  members: readonly WindowMeta[];
  pool: readonly WindowMeta[];
  raw: boolean;
  labels: boolean;
}

/** Group stage: stats, length histogram, link crop timeline, up to 32 mini cards (click → single view). */
export class GroupView {
  private canvases: StaticCanvas[] = [];

  constructor(
    private readonly root: HTMLElement,
    private readonly ds: EncoderDataset,
    private readonly grouper: Grouper,
    private readonly onOpen: (windowId: string) => void,
  ) {}

  show(m: GroupViewModel): void {
    this.canvases.forEach((c) => c.dispose());
    this.canvases = [];
    clear(this.root);
    this.root.append(caveatsBanner(this.ds));
    if (m.members.length === 0) {
      this.root.append(h("div", { class: "enc-empty" }, "No group selected or the filter is empty."));
      return;
    }
    const stats = GroupStats.of(this.ds, m.members);
    const privileged = m.kind === "sender" && m.members.some((w) => isSharedGhostPseudonym(w.scenario, w.sender_pseudo));
    const head = h("div", { class: "enc-title" },
      h("span", { class: "enc-title-id" }, `${kindLabel(m.kind)} · ${m.label}`),
      privileged ? h("span", { class: "enc-badge enc-badge-warn", title: "F13: ghosts share pseudonym 1; the true sender id is simulator ground truth" }, "privileged grouping") : null,
    );
    this.root.append(head, this.statsPanel(`${kindLabel(m.kind)} statistics`, stats, m.labels, m.kind === "link" ? this.timeline(m.members) : null));
    if (m.kind === "batch") {
      const seed = Number(m.key.startsWith("batch:") ? m.key.slice(6) : 0) || 0;
      const big = this.grouper.batch(m.pool, STATS_BATCH, seed);
      this.root.append(this.statsPanel(`Batch of ${STATS_BATCH} · statistics only`, GroupStats.of(this.ds, big), m.labels, null,
        `${big.length} windows drawn with seed ${seed} from ${fmtInt(m.pool.length)} filtered sample windows`));
    }
    this.root.append(this.cardsPanel(m), fullDatasetPanel(this.ds, m.labels));
  }

  private statsPanel(title: string, s: GroupStatsResult, labels: boolean, extra: HTMLElement | null, caption = ""): HTMLElement {
    const tile = (k: string, v: string, sub = ""): HTMLElement =>
      h("div", { class: "enc-tile" }, h("div", { class: "enc-tile-k" }, k), h("div", { class: "enc-tile-v" }, v), sub ? h("div", { class: "enc-tile-s" }, sub) : null);
    const hist = new StaticCanvas(() => undefined, "enc-canvas enc-hist-canvas");
    this.canvases.push(hist);
    const body = h("div", { class: "enc-body" },
      h("div", { class: "enc-tiles" },
        tile("windows", fmtInt(s.count)),
        labels ? tile("class mix", `${s.benign} B · ${s.gridsybil} G`, EVAL_ONLY_BADGE) : null,
        tile("padding", fmtPct(s.paddingShare), `${fmtInt(s.realRows)} real rows of ${fmtInt(s.count * this.ds.seqLen)}`),
        tile("mean length", s.meanLength.toFixed(1), "real rows per window"),
      ),
      extra,
      h("div", { class: "enc-hist-label" }, "Window length n (real rows), 64 bins"),
      hist.canvas,
    );
    const p = panel(title, caption, body);
    queueMicrotask(() => hist.setDraw((ctx, w, hh) => LengthHistogram.draw(ctx, w, hh, s.lengthHistogram)));
    return p;
  }

  /** Crop timeline of a link, e.g. 64 | 64 | 22 (segment width ∝ n). */
  private timeline(ws: readonly WindowMeta[]): HTMLElement {
    const K = ws[0]?.K ?? ws.length;
    const bar = h("div", { class: "enc-timeline" });
    for (const w of ws) {
      const seg = h("button", { type: "button", class: "enc-seg", title: `crop ${w.k + 1} of ${K} · n = ${w.n} · ${w.id}` }, String(w.n));
      seg.style.flex = `${w.n} 0 0`;
      seg.addEventListener("click", () => this.onOpen(w.id));
      bar.append(seg);
    }
    const note = ws.length === K ? `all ${K} crops of this link are in the sample` : `${ws.length} of ${K} crops in the current filter`;
    return h("div", { class: "enc-timeline-wrap" },
      h("div", { class: "enc-hist-label" }, `Link split into crops of 64 messages: ${ws.map((w) => w.n).join(" | ")} — ${note}`),
      bar);
  }

  private cardsPanel(m: GroupViewModel): HTMLElement {
    const grid = h("div", { class: "enc-cardgrid" });
    const feats = this.ds.manifest.features;
    for (const w of m.members.slice(0, MAX_CARDS)) {
      const c = new StaticCanvas(() => undefined, "enc-mini-canvas");
      this.canvases.push(c);
      const card = h("button", { type: "button", class: "enc-mini", title: `${w.id}\n${w.scenario} · ${w.run}\nreceiver ${w.receiver} · sender ${w.sender} (pseudonym ${w.sender_pseudo})` },
        c.canvas,
        h("div", { class: "enc-mini-meta" },
          h("span", {}, `n=${w.n}`),
          m.labels ? classChip(w.label_name) : null),
        h("div", { class: "enc-mini-id" }, shortId(w.id, 22)),
      );
      card.addEventListener("click", () => this.onOpen(w.id));
      grid.append(card);
      const matrix = this.ds.matrix(w, m.raw);
      const mask = this.ds.mask(w);
      queueMicrotask(() => c.setDraw((ctx, cw, ch) => TensorHeatmap.drawMini(ctx, cw, ch, { matrix, mask, features: feats, units: [], raw: m.raw })));
    }
    const caption = m.members.length > MAX_CARDS ? `first ${MAX_CARDS} of ${m.members.length}` : `${m.members.length} windows`;
    return panel("Windows · 64 rows (top → bottom) × 13 features + mask", caption, h("div", { class: "enc-body" }, grid,
      h("p", { class: "enc-src" }, "Hatched = padding (mask 0). Click a card to open it.")));
  }
}

function kindLabel(kind: GroupKind): string {
  return kind === "link" ? "Link" : kind === "sender" ? "Sender vehicle" : kind === "receiver" ? "Receiver" : "Random batch";
}
