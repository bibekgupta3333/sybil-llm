/** Types mirroring `simulation/public/data/manifest.json` written by scripts/export_simulation_sample.py. */

export type ClassLabel =
  | "Benign"
  | "DataReplaySybil"
  | "DoSDisruptiveSybil"
  | "DoSRandomSybil"
  | "GridSybil";

/** Axis-aligned box in SUMO planar metres. */
export interface BBox {
  readonly minX: number;
  readonly minY: number;
  readonly maxX: number;
  readonly maxY: number;
}

export interface BinaryInfo {
  readonly file: string;
  readonly dtype: "float32" | "float64";
  readonly shape: readonly number[];
  readonly sha256: string;
}

/** One exported identity: a pseudonym's stitched step sequence plus provenance. */
export interface IdentityRecord {
  readonly id: number;
  readonly sender_uid: string;
  readonly label: ClassLabel;
  readonly label_int: number;
  readonly family: string;
  readonly group: "0709" | "1416";
  readonly subfolder: string;
  readonly sender_pseudo: number;
  /** Physical SUMO vehicle id resolved from the ground-truth ledger; null if unresolved. */
  readonly sender: number | null;
  readonly pseudonyms_total: number | null;
  readonly pseudonyms_surviving: number;
  readonly tags: readonly string[];
  readonly row_offset: number;
  readonly n_steps: number;
  readonly n_windows: number;
  readonly x_windows_rows: readonly number[];
  readonly t_start: number;
  readonly t_end: number;
  /** "prepared": from X_windows.npy (ground-truth position). "raw_veremi": a fabricated broadcast
   *  observed in a neighbor's raw log — never part of X_windows, never seen in training. */
  readonly data_source: "prepared" | "raw_veremi";
  /** The id of this identity's real/fabricated counterpart for the same physical vehicle, if any. */
  readonly paired_identity_id: number | null;
  /** On a raw_veremi identity: a Benign identity from the same run that drives at the same time. */
  readonly benign_match_id: number | null;
  /** [t0, t1] simulation seconds shared by this identity and its benign match. */
  readonly overlap: readonly [number, number] | null;
  /** On a raw_veremi identity: one fake pseudonym (GridSybil) or every pseudonym heard from the sender. */
  readonly attack_trace: "fake_pseudonym" | "all_pseudonyms" | null;
  /** How many pseudonyms the broadcast trace spans. */
  readonly broadcast_pseudonyms: number | null;
}

export interface Manifest {
  readonly version: number;
  readonly seed: number;
  readonly window_size: number;
  readonly stride: number;
  readonly feature_cols: readonly string[];
  readonly norm_mean: readonly number[];
  readonly norm_std: readonly number[];
  readonly label_to_int: Readonly<Record<string, number>>;
  readonly class_palette: Readonly<Record<string, string>>;
  /** [minX, minY, maxX, maxY] of all benign steps — the honest road extent. */
  readonly road_bbox: readonly [number, number, number, number];
  readonly total_steps: number;
  readonly binaries: { readonly steps: BinaryInfo; readonly times: BinaryInfo };
  readonly identities: readonly IdentityRecord[];
}

/** Column positions of the 13 kinematic features, resolved once from `feature_cols`. */
export class FeatureIndex {
  readonly posX: number;
  readonly posY: number;
  readonly spdX: number;
  readonly spdY: number;
  readonly aclX: number;
  readonly aclY: number;
  readonly hedX: number;
  readonly hedY: number;
  readonly dt: number;
  readonly dposX: number;
  readonly dposY: number;
  readonly dspdX: number;
  readonly dspdY: number;
  readonly count: number;

  constructor(readonly names: readonly string[]) {
    const at = (name: string): number => {
      const i = names.indexOf(name);
      if (i < 0) throw new Error(`manifest.feature_cols is missing "${name}"`);
      return i;
    };
    this.posX = at("pos_x");
    this.posY = at("pos_y");
    this.spdX = at("spd_x");
    this.spdY = at("spd_y");
    this.aclX = at("acl_x");
    this.aclY = at("acl_y");
    this.hedX = at("hed_x");
    this.hedY = at("hed_y");
    this.dt = at("dt");
    this.dposX = at("dpos_x");
    this.dposY = at("dpos_y");
    this.dspdX = at("dspd_x");
    this.dspdY = at("dspd_y");
    this.count = names.length;
  }
}
