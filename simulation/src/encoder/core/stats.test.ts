import { describe, expect, it } from "vitest";

import { EncoderDataset } from "./encoder_dataset";
import { GroupStats } from "./stats";
import { standardFixture } from "./testing";

describe("GroupStats", () => {
  const fx = standardFixture();
  const ds = EncoderDataset.fromParts(fx.manifest, fx.x);

  it("counts classes, padding and lengths", () => {
    const s = GroupStats.of(ds, ds.windows);
    const real = 64 + 64 + 22 + 5 + 1 + 30 + 12;
    expect(s).toMatchObject({ count: 7, benign: 3, gridsybil: 4, realRows: real });
    expect(s.paddingShare).toBeCloseTo(1 - real / (7 * 64));
    expect(s.meanLength).toBeCloseTo(real / 7);
    expect(s.lengthHistogram.length).toBe(64);
    expect(s.lengthHistogram[63]).toBe(2);
    expect(s.lengthHistogram[0]).toBe(1);
  });

  it("handles an empty group", () => {
    expect(GroupStats.of(ds, [])).toMatchObject({ count: 0, paddingShare: 0, meanLength: 0 });
  });
});
