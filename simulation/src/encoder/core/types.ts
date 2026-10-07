/** Data contract of `public/data/encoder/` (written by `scripts/export_encoder_sample.py`). */

export type Split = "train" | "pretrain_val" | "val" | "test";
export type ClassName = "Benign" | "GridSybil";

export const SPLITS: readonly Split[] = ["train", "pretrain_val", "val", "test"];
export const CLASS_NAMES: readonly ClassName[] = ["Benign", "GridSybil"];

/** Per-split totals over the FULL encoder input (not the sample). */
export interface SplitSummary {
  readonly windows: number;
  readonly benign: number;
  readonly gridsybil: number;
  readonly messages: number;
  readonly padding_share: number;
}

export interface FullDatasetSummary {
  readonly windows: number;
  readonly messages: number;
  readonly padding_share: number;
  readonly sender_vehicles: number;
  readonly senders_in_multiple_splits: number;
  readonly per_split: Readonly<Partial<Record<Split, SplitSummary>>>;
  readonly length_only_auc_val: number;
  readonly length_only_auc_source: string;
}

export interface SampleSummary {
  readonly rule: string;
  readonly n_windows: number;
  readonly n_rows: number;
  readonly per_split_class: Readonly<Partial<Record<Split, Partial<Record<ClassName, number>>>>>;
}

/** A presenter window chosen by an explicit, recorded rule ("illustrative", not representative). */
export interface CuratedWindow {
  readonly name: string;
  readonly rule: string;
  readonly window_id: string;
}

/** One sampled window. `row` indexes the first of its `n` real rows in `x.f32`. */
export interface WindowMeta {
  readonly i: number;
  readonly id: string;
  readonly split: Split;
  readonly scenario: string;
  readonly run: string;
  readonly receiver_file: string;
  readonly receiver: number;
  readonly sender: number;
  readonly sender_pseudo: number;
  /** `scenario|run|receiver_file|sender|sender_pseudo`. */
  readonly link: string;
  /** Crop index within the link (0-based). */
  readonly k: number;
  /** Number of crops of this link in the full dataset. */
  readonly K: number;
  /** Real (unpadded) rows, 1..seq_len. */
  readonly n: number;
  readonly label: number;
  readonly label_name: ClassName;
  readonly row: number;
  readonly sender_splits_full: readonly Split[];
  readonly raw_trace: string;
}

export interface EncoderManifest {
  readonly version: number;
  readonly created_utc: string;
  readonly git_commit: string;
  readonly seed: number;
  readonly source: string;
  readonly seq_len: number;
  readonly features: readonly string[];
  readonly units: readonly string[];
  readonly norm: { readonly mean: readonly number[]; readonly std: readonly number[] };
  readonly full_dataset: FullDatasetSummary;
  readonly sample: SampleSummary;
  readonly curated: readonly CuratedWindow[];
  readonly windows: readonly WindowMeta[];
}

/** One trained model's description in `predictions/index.json`. */
export interface ModelCard {
  readonly name: string;
  readonly file: string;
  readonly checkpoint_sha256: string;
  readonly seed: number;
  readonly trained_on_splits: readonly Split[];
  readonly threshold_from: string;
  readonly threshold: number;
  readonly score_meaning: string;
  readonly created_utc: string;
}

export interface PredictionIndex {
  readonly models: readonly ModelCard[];
}

export interface PredictionEntry {
  readonly score: number;
  readonly pred: 0 | 1 | null;
}

/** "0709" / "1416" from a scenario folder name such as `GridSybil_1416`; "" if absent. */
export function scenarioGroup(scenario: string): string {
  const m = /_(\d{4})$/.exec(scenario);
  return m ? m[1] : "";
}
