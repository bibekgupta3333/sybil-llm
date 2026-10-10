import { describe, expect, it } from "vitest";

import { binScores, rangeFraction, sharedRange } from "./histogram";

describe("score histogram", () => {
  it("covers every score and θ", () => {
    expect(sharedRange([0.2, 0.5, 0.4])).toEqual({ lo: 0.2, hi: 0.5 });
    expect(sharedRange([0.2, 0.5], 0.9)).toEqual({ lo: 0.2, hi: 0.9 });
    expect(sharedRange([0.3, Number.NaN, Infinity], 0.1)).toEqual({ lo: 0.1, hi: 0.3 });
    expect(sharedRange([0.5, 0.5])).toEqual({ lo: 0, hi: 1 });
    expect(sharedRange([])).toEqual({ lo: 0, hi: 1 });
  });

  it("bins half-open with the upper edge in the last bin", () => {
    const r = { lo: 0, hi: 1 };
    expect(binScores([0, 0.24, 0.25, 0.5, 0.99, 1], r, 4)).toEqual([2, 1, 1, 2]);
    expect(binScores([-5, 7, Number.NaN], r, 4)).toEqual([1, 0, 0, 1]);
    const counts = binScores([0.1, 0.2, 0.3, 0.9], r, 10);
    expect(counts.reduce((a, b) => a + b, 0)).toBe(4);
    expect(counts[9]).toBe(1);
    expect(() => binScores([0.1], r, 0)).toThrow();
  });

  it("maps a value into [0, 1] across the range", () => {
    expect(rangeFraction(0.5, { lo: 0, hi: 2 })).toBe(0.25);
    expect(rangeFraction(-1, { lo: 0, hi: 2 })).toBe(0);
    expect(rangeFraction(3, { lo: 0, hi: 2 })).toBe(1);
    expect(rangeFraction(1, { lo: 1, hi: 1 })).toBe(0.5);
  });
});
