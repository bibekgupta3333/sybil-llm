import type { ClassMetrics, DetectionManifest, DetectionWindow, MetricsByClass } from "./types";
import { orderClasses } from "./types";

export type FlagFilter = "all" | "flagged" | "unflagged";
export type SortOrder = "score_desc" | "score_asc" | "index";

export interface WindowQuery {
  /** A class name or "all". */
  readonly cls?: string;
  readonly flag?: FlagFilter;
  readonly sort?: SortOrder;
}

/** Claimed vs receiver positions of one window, in metres, plus range and the joint bounding box. */
export interface WindowPositions {
  readonly claimed: [number, number][];
  readonly receiver: [number, number][];
  readonly range: number[];
  readonly bbox: [number, number, number, number];
}

/** Summary of window lengths (real rows) in a set of windows. */
export interface LengthSummary {
  readonly count: number;
  readonly min: number;
  readonly median: number;
  readonly max: number;
  readonly mean: number;
}

/** One row of the metrics table: the class and its metrics exactly as written, or null if absent. */
export interface MetricsRow {
  readonly cls: string;
  readonly metrics: ClassMetrics | null;
}

/**
 * The exported detection sample: manifest + raw-unit real rows. Validates the row offsets against `x.f32`
 * (windows must be concatenated in order) so a stale or truncated bundle fails loudly instead of drawing
 * the wrong rows.
 */
export class DetectionDataset {
  readonly windows: readonly DetectionWindow[];
  readonly nFeatures: number;
  readonly nRows: number;
  private readonly index = new Map<string, DetectionWindow>();

  private constructor(
    readonly manifest: DetectionManifest,
    private readonly x: Float32Array,
  ) {
    if (manifest.schema !== 1) throw new Error(`unsupported detection schema ${manifest.schema} (expected 1)`);
    this.windows = manifest.windows;
    this.nFeatures = manifest.features.length;
    if (this.nFeatures < 1) throw new Error("manifest lists no features");
    const nRows = x.length / this.nFeatures;
    if (!Number.isInteger(nRows)) {
      throw new Error(`x.f32 length ${x.length} is not a multiple of ${this.nFeatures} features`);
    }
    this.nRows = nRows;
    let expected = 0;
    for (const w of this.windows) {
      if (!Number.isInteger(w.n) || w.n < 1) throw new Error(`window ${w.id}: n = ${w.n} must be ≥ 1`);
      if (w.n > manifest.model.seq_len) throw new Error(`window ${w.id}: n = ${w.n} exceeds seq_len ${manifest.model.seq_len}`);
      if (w.offset !== expected) throw new Error(`window ${w.id}: offset ${w.offset}, expected ${expected} (rows must be concatenated in order)`);
      if (this.index.has(w.id)) throw new Error(`duplicate window id ${w.id}`);
      this.index.set(w.id, w);
      expected += w.n;
    }
    if (expected !== nRows) throw new Error(`windows cover ${expected} rows but x.f32 holds ${nRows}`);
  }

  /** Fetches `manifest.json` + `x.f32`. Throws `DetectionDataMissing` when the bundle is absent. */
  static async load(baseUrl = "data/detection/"): Promise<DetectionDataset> {
    const mres = await fetch(`${baseUrl}manifest.json`);
    if (!mres.ok) throw new DetectionDataMissing(`manifest.json: HTTP ${mres.status}`);
    let manifest: DetectionManifest;
    try {
      // The Vite dev server answers a missing file with index.html (HTTP 200), so a parse error means "missing".
      manifest = JSON.parse(await mres.text()) as DetectionManifest;
    } catch {
      throw new DetectionDataMissing("manifest.json is not JSON (file missing?)");
    }
    const xres = await fetch(`${baseUrl}x.f32`);
    if (!xres.ok) throw new DetectionDataMissing(`x.f32: HTTP ${xres.status}`);
    return new DetectionDataset(manifest, new Float32Array(await xres.arrayBuffer()));
  }

  static fromParts(manifest: DetectionManifest, x: Float32Array): DetectionDataset {
    return new DetectionDataset(manifest, x);
  }

  byId(id: string): DetectionWindow | undefined {
    return this.index.get(id);
  }

  /** Classes present in the windows or the metrics, in display order. */
  classes(): string[] {
    return orderClasses([...this.windows.map((w) => w.class), ...Object.keys(this.manifest.metrics.sample ?? {})]);
  }

  /** Unit of a feature ("" if the manifest gives none). */
  unit(feature: string): string {
    return this.manifest.units[feature] ?? "";
  }

  /** The window's n × features real rows, raw units (zero-copy view). */
  rows(w: DetectionWindow): Float32Array {
    const f = this.nFeatures;
    return this.x.subarray(w.offset * f, (w.offset + w.n) * f);
  }

  /** One feature's values over the window's rows; null if the manifest has no such feature. */
  column(w: DetectionWindow, feature: string): number[] | null {
    const j = this.manifest.features.indexOf(feature);
    if (j < 0) return null;
    const r = this.rows(w);
    const out: number[] = [];
    for (let t = 0; t < w.n; t++) out.push(r[t * this.nFeatures + j]);
    return out;
  }

  /** Claimed vs receiver positions; null when the position features are not exported. */
  positions(w: DetectionWindow): WindowPositions | null {
    const cx = this.column(w, "claimed_pos_x");
    const cy = this.column(w, "claimed_pos_y");
    const rx = this.column(w, "rx_pos_x");
    const ry = this.column(w, "rx_pos_y");
    if (!cx || !cy || !rx || !ry) return null;
    const claimed = cx.map((x, t): [number, number] => [x, cy[t]]);
    const receiver = rx.map((x, t): [number, number] => [x, ry[t]]);
    const range = this.column(w, "range") ?? claimed.map(([x, y], t) => Math.hypot(x - receiver[t][0], y - receiver[t][1]));
    const xs = [...cx, ...rx].filter(Number.isFinite);
    const ys = [...cy, ...ry].filter(Number.isFinite);
    const bbox: [number, number, number, number] = xs.length && ys.length
      ? [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]
      : [0, 0, 1, 1];
    return { claimed, receiver, range, bbox };
  }

  /** Windows matching the class / flag filter, sorted (score ties broken by sample index). */
  query(q: WindowQuery = {}): DetectionWindow[] {
    const cls = q.cls ?? "all";
    const flag = q.flag ?? "all";
    const out = this.windows.filter(
      (w) =>
        (cls === "all" || w.class === cls) &&
        (flag === "all" || (flag === "flagged" ? w.flag : !w.flag)),
    );
    const sort = q.sort ?? "score_desc";
    if (sort === "index") return out.sort((a, b) => a.i - b.i);
    const sign = sort === "score_desc" ? -1 : 1;
    return out.sort((a, b) => sign * (a.score - b.score) || a.i - b.i);
  }

  /** Scores of one class's windows (or all windows). */
  scores(cls = "all"): number[] {
    return this.windows.filter((w) => cls === "all" || w.class === cls).map((w) => w.score);
  }

  /** Highest score among the shown windows (0 when there are none). */
  scoreMax(): number {
    return this.windows.reduce((m, w) => Math.max(m, w.score), 0);
  }

  /** Length (real rows) summary of one class's sampled windows; null if the class has none. */
  lengths(cls: string): LengthSummary | null {
    const ns = this.windows.filter((w) => w.class === cls).map((w) => w.n).sort((a, b) => a - b);
    if (ns.length === 0) return null;
    const mid = ns.length >> 1;
    const median = ns.length % 2 ? ns[mid] : (ns[mid - 1] + ns[mid]) / 2;
    return { count: ns.length, min: ns[0], median, max: ns[ns.length - 1], mean: ns.reduce((a, b) => a + b, 0) / ns.length };
  }

  /** Metrics table rows for the sample or the full split; null when that block was not written. */
  metricsRows(which: "sample" | "full_test_1416"): MetricsRow[] | null {
    const block: MetricsByClass | null | undefined = this.manifest.metrics[which];
    if (!block) return null;
    return orderClasses([...this.classes(), ...Object.keys(block)]).map((cls) => ({ cls, metrics: block[cls] ?? null }));
  }
}

/** The detection bundle is not there (exporter not run yet). */
export class DetectionDataMissing extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DetectionDataMissing";
  }
}

/** A metric exactly as the exporter wrote it: numbers verbatim, null / missing as "—". */
export function fmtMetric(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isFinite(v) ? String(v) : "—";
  if (typeof v === "boolean") return v ? "true" : "false";
  return String(v);
}
