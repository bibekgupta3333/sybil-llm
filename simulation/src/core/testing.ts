import type { ClassLabel, IdentityRecord, Manifest } from "./manifest";

/** Synthetic fixtures shared by the core unit tests. Not shipped in the app bundle. */

export const FEATURES = [
  "pos_x", "pos_y", "spd_x", "spd_y", "acl_x", "acl_y", "hed_x", "hed_y",
  "dt", "dpos_x", "dpos_y", "dspd_x", "dspd_y",
] as const;

export interface SyntheticIdentity {
  readonly label: ClassLabel;
  readonly nSteps: number;
  /** Fills one step's 13 features; defaults to a straight 12 m/s drive along +x at 1 Hz. */
  readonly fill?: (step: number, row: Float32Array) => void;
  readonly sender?: number;
  readonly tags?: readonly string[];
  readonly subfolder?: string;
  readonly group?: "0709" | "1416";
  readonly dataSource?: "prepared" | "raw_veremi";
  readonly pairedId?: number | null;
  readonly benignMatchId?: number | null;
  readonly overlap?: readonly [number, number] | null;
  readonly attackTrace?: "fake_pseudonym" | "all_pseudonyms" | null;
  readonly broadcastPseudonyms?: number | null;
}

export function straightDrive(step: number, row: Float32Array): void {
  const speed = 12;
  const dt = step === 0 ? 0 : 1;
  row[0] = 100 + speed * step; // pos_x
  row[1] = 500; // pos_y
  row[2] = speed; // spd_x
  row[3] = 0;
  row[4] = 0;
  row[5] = 0;
  row[6] = 1; // hed_x
  row[7] = 0;
  row[8] = dt;
  row[9] = step === 0 ? 0 : speed * dt; // dpos_x
  row[10] = 0;
  row[11] = 0;
  row[12] = 0;
}

export function buildFixture(specs: readonly SyntheticIdentity[], t0 = 25200) {
  const f = FEATURES.length;
  const total = specs.reduce((n, s) => n + s.nSteps, 0);
  const steps = new Float32Array(total * f);
  const times = new Float64Array(total);
  const identities: IdentityRecord[] = [];
  let offset = 0;
  specs.forEach((spec, id) => {
    const row = new Float32Array(f);
    let t = t0 + id * 1000;
    for (let i = 0; i < spec.nSteps; i++) {
      row.fill(0);
      (spec.fill ?? straightDrive)(i, row);
      steps.set(row, (offset + i) * f);
      t += row[8];
      times[offset + i] = t;
    }
    const nWindows = Math.floor((spec.nSteps - 20) / 10) + 1;
    identities.push({
      id, sender_uid: `Fam_0709_run_${1000 + id}`, label: spec.label,
      label_int: spec.label === "Benign" ? 0 : 4, family: "GridSybil", group: spec.group ?? "0709", subfolder: spec.subfolder ?? "run",
      sender_pseudo: 1000 + id, sender: spec.sender ?? id, pseudonyms_total: 1, pseudonyms_surviving: 1,
      tags: spec.tags ?? [], row_offset: offset, n_steps: spec.nSteps, n_windows: nWindows,
      x_windows_rows: Array.from({ length: nWindows }, (_, k) => k), t_start: times[offset],
      t_end: times[offset + spec.nSteps - 1],
      data_source: spec.dataSource ?? "prepared", paired_identity_id: spec.pairedId ?? null,
      benign_match_id: spec.benignMatchId ?? null, overlap: spec.overlap ?? null,
      attack_trace: spec.attackTrace ?? null, broadcast_pseudonyms: spec.broadcastPseudonyms ?? null,
    });
    offset += spec.nSteps;
  });
  const manifest: Manifest = {
    version: 1, seed: 0, window_size: 20, stride: 10, feature_cols: FEATURES,
    norm_mean: Array(f).fill(0), norm_std: Array(f).fill(1),
    label_to_int: { Benign: 0, GridSybil: 4 }, class_palette: { Benign: "#16A34A", GridSybil: "#DC2626" },
    road_bbox: [0, 0, 1400, 1400], total_steps: total,
    binaries: {
      steps: { file: "steps.f32", dtype: "float32", shape: [total, f], sha256: "" },
      times: { file: "times.f64", dtype: "float64", shape: [total], sha256: "" },
    },
    identities,
  };
  return { manifest, steps, times };
}
