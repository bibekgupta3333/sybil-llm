import type { ClassName, EncoderManifest, Split, WindowMeta } from "./types";

/** Synthetic fixtures shared by the encoder core unit tests. Not shipped in the app bundle. */

export const ENC_FEATURES = [
  "claimed_pos_x", "claimed_pos_y", "rx_pos_x", "rx_pos_y", "claimed_vel_x", "claimed_vel_y",
  "rx_vel_x", "rx_vel_y", "claimed_acl_x", "claimed_acl_y", "range", "bearing", "log_dtau",
] as const;

export const ENC_UNITS = ["m", "m", "m", "m", "m/s", "m/s", "m/s", "m/s", "m/s²", "m/s²", "m", "rad", "log s"];

/** Normalisation used by the fixture: mean 100·(j+1), std (j+1). */
export const FIXTURE_MEAN = ENC_FEATURES.map((_, j) => 100 * (j + 1));
export const FIXTURE_STD = ENC_FEATURES.map((_, j) => j + 1);

export interface SyntheticWindow {
  readonly n: number;
  readonly split?: Split;
  readonly cls?: ClassName;
  readonly scenario?: string;
  readonly run?: string;
  readonly receiverFile?: string;
  readonly receiver?: number;
  readonly sender?: number;
  readonly senderPseudo?: number;
  readonly k?: number;
  readonly K?: number;
  /** Normalised value of feature j at real row t; default `t + j / 100`. */
  readonly fill?: (t: number, j: number) => number;
}

export function defaultFill(t: number, j: number): number {
  return t + j / 100;
}

export function buildEncoderFixture(specs: readonly SyntheticWindow[]): { manifest: EncoderManifest; x: Float32Array } {
  const f = ENC_FEATURES.length;
  const nRows = specs.reduce((s, w) => s + w.n, 0);
  const x = new Float32Array(nRows * f);
  const windows: WindowMeta[] = [];
  let row = 0;
  specs.forEach((s, i) => {
    const scenario = s.scenario ?? "GridSybil_0709";
    const run = s.run ?? "run_a";
    const receiverFile = s.receiverFile ?? `${scenario}/${run}/traceJSON-10-9-A0-0-0.json`;
    const sender = s.sender ?? 100 + i;
    const senderPseudo = s.senderPseudo ?? 1000 + sender;
    const cls = s.cls ?? "Benign";
    const split = s.split ?? "test";
    for (let t = 0; t < s.n; t++) for (let j = 0; j < f; j++) x[(row + t) * f + j] = (s.fill ?? defaultFill)(t, j);
    windows.push({
      i, id: `w${String(i).padStart(3, "0")}`, split, scenario, run, receiver_file: receiverFile,
      receiver: s.receiver ?? 10, sender, sender_pseudo: senderPseudo,
      link: `${scenario}|${run}|${receiverFile}|${sender}|${senderPseudo}`,
      k: s.k ?? 0, K: s.K ?? 1, n: s.n, label: cls === "Benign" ? 0 : 16, label_name: cls, row,
      sender_splits_full: [split], raw_trace: `data/VeReMi-Dataset/${receiverFile}`,
    });
    row += s.n;
  });
  const manifest: EncoderManifest = {
    version: 1, created_utc: "2026-10-07T00:00:00Z", git_commit: "fixture", seed: 0,
    source: "synthetic", seq_len: 64, features: [...ENC_FEATURES], units: ENC_UNITS,
    norm: { mean: FIXTURE_MEAN, std: FIXTURE_STD },
    full_dataset: {
      windows: specs.length, messages: nRows, padding_share: 1 - nRows / (64 * specs.length),
      sender_vehicles: new Set(windows.map((w) => w.sender)).size, senders_in_multiple_splits: 0,
      per_split: {}, length_only_auc_val: 0.531, length_only_auc_source: "synthetic",
    },
    sample: { rule: "all synthetic windows", n_windows: specs.length, n_rows: nRows, per_split_class: {} },
    curated: windows.length ? [{ name: "first", rule: "smallest id", window_id: windows[0].id }] : [],
    windows,
  };
  return { manifest, x };
}

/** Two links of one GridSybil ghost + benign windows across splits and scenarios. */
export function standardFixture() {
  return buildEncoderFixture([
    { n: 64, cls: "GridSybil", sender: 7, senderPseudo: 1, k: 0, K: 3, receiver: 10 },
    { n: 64, cls: "GridSybil", sender: 7, senderPseudo: 1, k: 1, K: 3, receiver: 10 },
    { n: 22, cls: "GridSybil", sender: 7, senderPseudo: 1, k: 2, K: 3, receiver: 10 },
    { n: 5, cls: "Benign", sender: 8, receiver: 10 },
    { n: 1, cls: "Benign", sender: 9, receiver: 11, receiverFile: "GridSybil_0709/run_a/traceJSON-11.json" },
    { n: 30, cls: "Benign", split: "train", scenario: "DoSRandomSybil_1416", run: "run_b", sender: 20 },
    { n: 12, cls: "GridSybil", split: "val", scenario: "GridSybil_1416", run: "run_c", sender: 21, senderPseudo: 1 },
  ]);
}
