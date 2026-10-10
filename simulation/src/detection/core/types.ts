/**
 * Data contract of `public/data/detection/` (written by `scripts/export_detection_sample.py`, schema 1).
 *
 * `x.f32` holds little-endian float32 real rows in raw units (metres, m/s, …), `n × features` per window,
 * concatenated in `windows` order; `offset` is the window's first row.
 */

/** Per-class metrics as the exporter wrote them. Values are shown verbatim; null = not computed. */
export interface ClassMetrics {
  readonly n?: number | null;
  readonly auroc_vs_benign?: number | null;
  readonly flag_rate?: number | null;
  readonly length_baseline_auroc?: number | null;
  readonly mean_rows?: number | null;
  readonly [key: string]: unknown;
}

export type MetricsByClass = Readonly<Record<string, ClassMetrics>>;

export interface DetectionMetrics {
  readonly sample: MetricsByClass;
  readonly full_test_1416?: MetricsByClass | null;
  readonly benign_fpr_at_theta?: number | null;
  readonly [key: string]: unknown;
}

export interface ModelInfo {
  readonly run_id: string;
  readonly checkpoint: string;
  readonly sha256?: string;
  readonly epoch?: number;
  readonly best_val_joint?: number;
  readonly dataset?: string;
  readonly seq_len: number;
  readonly d_model?: number;
}

export interface MethodInfo {
  readonly name: string;
  readonly k: number;
  readonly bank_size: number;
  readonly theta: number;
  readonly theta_rule: string;
  readonly seed: number;
}

export interface ScopeInfo {
  readonly split: string;
  readonly group: string;
  readonly note?: string;
  readonly per_class?: number;
}

/** One scored test window. `label` / `class` are evaluation only: read by the exporter after scoring. */
export interface DetectionWindow {
  readonly i: number;
  readonly id: string;
  readonly scenario: string;
  readonly class: string;
  readonly label: number;
  readonly n: number;
  readonly offset: number;
  readonly score: number;
  readonly flag: boolean;
  readonly rank_pct: number;
}

export interface DetectionManifest {
  readonly schema: number;
  readonly created?: string;
  readonly git_commit?: string;
  readonly model: ModelInfo;
  readonly method: MethodInfo;
  readonly scope: ScopeInfo;
  readonly features: readonly string[];
  readonly units: Readonly<Record<string, string>>;
  readonly metrics: DetectionMetrics;
  readonly caveats: readonly string[];
  readonly windows: readonly DetectionWindow[];
}

/** Display order of the classes; any other class in the manifest follows, sorted by name. */
export const CLASS_ORDER: readonly string[] = [
  "Benign",
  "GridSybil",
  "DataReplaySybil",
  "DoSRandomSybil",
  "DoSDisruptiveSybil",
];

export const CLASS_COLORS: Readonly<Record<string, string>> = {
  Benign: "#16A34A",
  GridSybil: "#DC2626",
  DataReplaySybil: "#8B5CF6",
  DoSRandomSybil: "#F97316",
  DoSDisruptiveSybil: "#2563EB",
};

export function classColor(cls: string): string {
  return CLASS_COLORS[cls] ?? "#64748B";
}

/** Sort classes in `CLASS_ORDER`, unknown ones after it alphabetically. */
export function orderClasses(classes: Iterable<string>): string[] {
  const rank = (c: string): number => {
    const i = CLASS_ORDER.indexOf(c);
    return i < 0 ? CLASS_ORDER.length : i;
  };
  return [...new Set(classes)].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
}
