import type { DetectionManifest, DetectionWindow } from "./types";

/** Synthetic fixture for the detection core tests. Not shipped in the app bundle. Values are made up. */

export const DET_FEATURES = [
  "claimed_pos_x", "claimed_pos_y", "rx_pos_x", "rx_pos_y", "claimed_vel_x", "claimed_vel_y",
  "rx_vel_x", "rx_vel_y", "claimed_acl_x", "claimed_acl_y", "range", "bearing", "log_dtau",
] as const;

export const DET_UNITS: Record<string, string> = {
  claimed_pos_x: "m", claimed_pos_y: "m", rx_pos_x: "m", rx_pos_y: "m",
  claimed_vel_x: "m/s", claimed_vel_y: "m/s", rx_vel_x: "m/s", rx_vel_y: "m/s",
  claimed_acl_x: "m/s²", claimed_acl_y: "m/s²", range: "m", bearing: "rad", log_dtau: "log s",
};

export interface SyntheticDetWindow {
  readonly n: number;
  readonly cls: string;
  readonly score: number;
  readonly flag?: boolean;
}

/** Row value of feature j at row t of window i: `1000 i + 10 t + j` (easy to assert offsets). */
export function fixtureValue(i: number, t: number, j: number): number {
  return 1000 * i + 10 * t + j;
}

export function buildDetectionFixture(specs: readonly SyntheticDetWindow[], theta = 0.5): { manifest: DetectionManifest; x: Float32Array } {
  const f = DET_FEATURES.length;
  const nRows = specs.reduce((s, w) => s + w.n, 0);
  const x = new Float32Array(nRows * f);
  const windows: DetectionWindow[] = [];
  let offset = 0;
  specs.forEach((s, i) => {
    for (let t = 0; t < s.n; t++) for (let j = 0; j < f; j++) x[(offset + t) * f + j] = fixtureValue(i, t, j);
    windows.push({
      i,
      id: `w${String(i).padStart(3, "0")}`,
      scenario: `${s.cls}_1416`,
      class: s.cls,
      label: s.cls === "Benign" ? 0 : 1,
      n: s.n,
      offset,
      score: s.score,
      flag: s.flag ?? s.score >= theta,
      rank_pct: 0,
    });
    offset += s.n;
  });
  const manifest: DetectionManifest = {
    schema: 1,
    created: "2026-10-10T00:00:00Z",
    git_commit: "abc1234",
    model: { run_id: "model-all", checkpoint: "best.pt", sha256: "00", epoch: 10, best_val_joint: 0.3773, dataset: "all", seq_len: 24, d_model: 128 },
    method: { name: "kNN cosine anomaly score (label-free)", k: 10, bank_size: 100, theta, theta_rule: "train-score p95", seed: 0 },
    scope: { split: "test", group: "1416", note: "0709 excluded until F14 is decided", per_class: 3 },
    features: [...DET_FEATURES],
    units: DET_UNITS,
    metrics: {
      sample: {
        Benign: { n: 2, auroc_vs_benign: null, flag_rate: 0.5, length_baseline_auroc: null, mean_rows: 2.5 },
        GridSybil: { n: 2, auroc_vs_benign: 0.75, flag_rate: 0.5, length_baseline_auroc: 0.625, mean_rows: 3 },
      },
      full_test_1416: null,
      benign_fpr_at_theta: 0.05,
    },
    caveats: ["synthetic fixture"],
    windows,
  };
  return { manifest, x };
}

/** Five windows, two classes plus one DoS window with no metrics entry. */
export function standardDetectionFixture(): { manifest: DetectionManifest; x: Float32Array } {
  return buildDetectionFixture([
    { n: 2, cls: "Benign", score: 0.1 },
    { n: 3, cls: "GridSybil", score: 0.9 },
    { n: 1, cls: "DoSRandomSybil", score: 0.6 },
    { n: 3, cls: "Benign", score: 0.7 },
    { n: 3, cls: "GridSybil", score: 0.2 },
  ]);
}
