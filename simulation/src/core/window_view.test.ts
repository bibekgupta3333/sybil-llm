import { describe, expect, it } from "vitest";

import { Dataset } from "./dataset";
import { buildFixture } from "./testing";
import { WindowView } from "./window_view";

describe("WindowView", () => {
  const fx = buildFixture([{ label: "Benign", nSteps: 57 }]);
  const ds = Dataset.fromParts(fx.manifest, fx.steps, fx.times);
  const ident = ds.identity(0);

  it("slices window k as steps [10k, 10k+20) with the trailing tail excluded", () => {
    const view = new WindowView(ident, { mean: fx.manifest.norm_mean, std: fx.manifest.norm_std });
    expect(view.count).toBe(4);
    const w2 = view.raw(2);
    expect(w2.length).toBe(20 * 13);
    expect(w2[0]).toBe(ident.posX(20));
    expect(w2[19 * 13]).toBe(ident.posX(39));
    expect(() => view.raw(4)).toThrow(RangeError);
  });

  it("normalizes with train-split stats and round-trips, guarding std == 0", () => {
    const mean = Array(13).fill(0);
    const std = Array(13).fill(2);
    mean[0] = 100;
    std[5] = 0; // acl_y is constant in the fixture → std 0 must not divide by zero
    const view = new WindowView(ident, { mean, std });
    const z = view.normalized(0);
    expect(z[0]).toBeCloseTo((ident.posX(0) - 100) / 2);
    expect(z[5]).toBe(0);
    expect(Number.isFinite(z[5])).toBe(true);
    const back = view.denormalize(z);
    expect(back[0]).toBeCloseTo(ident.posX(0));
  });
});
