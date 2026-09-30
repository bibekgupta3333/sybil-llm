import { describe, expect, it } from "vitest";

import { Dataset } from "./dataset";
import { buildFixture } from "./testing";

describe("Identity.windowSlice", () => {
  const fx = buildFixture([{ label: "GridSybil", nSteps: 45, sender: 3 }]);
  const ds = Dataset.fromParts(fx.manifest, fx.steps, fx.times);
  const identity = ds.identity(0); // windowCount = floor((45-20)/10)+1 = 3

  it("slices exactly the 20 steps of the requested window", () => {
    const sliced = identity.windowSlice(1); // steps [10, 30)
    expect(sliced.stepCount).toBe(20);
    expect(sliced.windowCount).toBe(1);
    expect(sliced.posX(0)).toBe(identity.posX(10));
    expect(sliced.posX(19)).toBe(identity.posX(29));
    expect(sliced.times[0]).toBe(identity.times[10]);
  });

  it("carries the original window's X_windows row and label", () => {
    const sliced = identity.windowSlice(2);
    expect(sliced.record.x_windows_rows).toEqual([identity.record.x_windows_rows[2]]);
    expect(sliced.record.n_windows).toBe(1);
    expect(sliced.label).toBe(identity.label);
  });

  it("timeSlice keeps only steps inside [t0, t1], snapped to a window boundary", () => {
    const fx2 = buildFixture([{ label: "Benign", nSteps: 60 }]);
    const long = Dataset.fromParts(fx2.manifest, fx2.steps, fx2.times).identity(0);
    const t = long.times;
    const sliced = long.timeSlice(t[13], t[55]);   // step 13 snaps forward to step 20
    expect(sliced.times[0]).toBe(t[20]);
    expect(sliced.times[sliced.stepCount - 1]).toBe(t[55]);
    expect(sliced.posX(0)).toBe(long.posX(20));
    expect(sliced.windowCount).toBe(2);            // 36 steps -> windows at 0 and 10
    expect(sliced.record.x_windows_rows).toEqual(long.record.x_windows_rows.slice(2, 4));
  });

  it("timeSlice returns the identity unchanged when under one window would remain", () => {
    const t = identity.times;
    expect(identity.timeSlice(t[5], t[15])).toBe(identity);
  });

  it("rejects an out-of-range window index", () => {
    expect(() => identity.windowSlice(-1)).toThrow(RangeError);
    expect(() => identity.windowSlice(3)).toThrow(RangeError);
  });
});
