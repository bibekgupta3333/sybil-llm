import type { EncoderDataset } from "./encoder_dataset";
import type { WindowMeta } from "./types";

export interface GroupStatsResult {
  readonly count: number;
  readonly benign: number;
  readonly gridsybil: number;
  /** Padding rows / (count × seq_len); 0 for an empty group. */
  readonly paddingShare: number;
  readonly realRows: number;
  /** Bin j counts windows with n = j + 1 (seq_len bins). */
  readonly lengthHistogram: number[];
  /** Mean n; 0 for an empty group. */
  readonly meanLength: number;
}

export class GroupStats {
  static of(ds: EncoderDataset, ws: readonly WindowMeta[]): GroupStatsResult {
    const T = ds.seqLen;
    const hist = new Array<number>(T).fill(0);
    let benign = 0;
    let gridsybil = 0;
    let realRows = 0;
    for (const w of ws) {
      if (w.label_name === "Benign") benign++;
      else gridsybil++;
      realRows += w.n;
      hist[Math.min(T, Math.max(1, w.n)) - 1]++;
    }
    const count = ws.length;
    return {
      count,
      benign,
      gridsybil,
      paddingShare: count ? 1 - realRows / (count * T) : 0,
      realRows,
      lengthHistogram: hist,
      meanLength: count ? realRows / count : 0,
    };
  }
}
