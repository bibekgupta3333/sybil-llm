/** Score histograms: one shared x range so the classes and θ are comparable. */

export interface Range {
  readonly lo: number;
  readonly hi: number;
}

/**
 * The range covering every finite score and θ. A degenerate range (all equal) is widened by ±0.5 so bins
 * have a width; no finite values gives [0, 1].
 */
export function sharedRange(scores: Iterable<number>, theta?: number): Range {
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of scores) {
    if (!Number.isFinite(s)) continue;
    if (s < lo) lo = s;
    if (s > hi) hi = s;
  }
  if (theta !== undefined && Number.isFinite(theta)) {
    lo = Math.min(lo, theta);
    hi = Math.max(hi, theta);
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return { lo: 0, hi: 1 };
  if (hi === lo) return { lo: lo - 0.5, hi: hi + 0.5 };
  return { lo, hi };
}

/**
 * Counts per bin of `nBins` equal-width bins over [lo, hi]. Bins are half-open [a, b) except the last, which
 * includes `hi`. Values outside the range are clamped into the end bins; non-finite values are skipped.
 */
export function binScores(scores: Iterable<number>, range: Range, nBins: number): number[] {
  if (!Number.isInteger(nBins) || nBins < 1) throw new Error(`nBins must be a positive integer, got ${nBins}`);
  const counts = new Array<number>(nBins).fill(0);
  const width = (range.hi - range.lo) / nBins;
  for (const s of scores) {
    if (!Number.isFinite(s)) continue;
    let b = width > 0 ? Math.floor((s - range.lo) / width) : 0;
    if (b < 0) b = 0;
    if (b >= nBins) b = nBins - 1;
    counts[b] += 1;
  }
  return counts;
}

/** Position of `value` in [0, 1] across the range (clamped). */
export function rangeFraction(value: number, range: Range): number {
  const span = range.hi - range.lo;
  if (span <= 0) return 0.5;
  return Math.max(0, Math.min(1, (value - range.lo) / span));
}
