import type { EncoderDataset } from "./encoder_dataset";
import type { WindowMeta } from "./types";

export type Point = [number, number];

export interface WindowGeometryResult {
  /** Sender-claimed positions (m), one per real row. */
  readonly claimed: Point[];
  /** Receiver's own positions (m), one per real row. */
  readonly receiver: Point[];
  /** Receiver-to-claimed distance (m), one per real row. */
  readonly range: number[];
  /** [minx, miny, maxx, maxy] over claimed and receiver points; zeros if empty. */
  readonly bbox: [number, number, number, number];
}

/** Spatial view of a window in raw units (metres), real rows only. */
export class WindowGeometry {
  static of(ds: EncoderDataset, w: WindowMeta): WindowGeometryResult {
    const cx = ds.featureIndex("claimed_pos_x");
    const cy = ds.featureIndex("claimed_pos_y");
    const rx = ds.featureIndex("rx_pos_x");
    const ry = ds.featureIndex("rx_pos_y");
    const rg = ds.featureIndex("range");
    const raw = ds.rawRows(w);
    const f = ds.nFeatures;
    const claimed: Point[] = [];
    const receiver: Point[] = [];
    const range: number[] = [];
    let minx = Infinity;
    let miny = Infinity;
    let maxx = -Infinity;
    let maxy = -Infinity;
    for (let i = 0; i < w.n; i++) {
      const o = i * f;
      const c: Point = [raw[o + cx], raw[o + cy]];
      const r: Point = [raw[o + rx], raw[o + ry]];
      claimed.push(c);
      receiver.push(r);
      range.push(raw[o + rg]);
      for (const [x, y] of [c, r]) {
        minx = Math.min(minx, x);
        miny = Math.min(miny, y);
        maxx = Math.max(maxx, x);
        maxy = Math.max(maxy, y);
      }
    }
    const bbox: [number, number, number, number] = w.n ? [minx, miny, maxx, maxy] : [0, 0, 0, 0];
    return { claimed, receiver, range, bbox };
  }
}
