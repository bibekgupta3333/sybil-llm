import { describe, expect, it } from "vitest";

import { EncoderDataset } from "./encoder_dataset";
import { WindowGeometry } from "./geometry";
import { FIXTURE_MEAN, FIXTURE_STD, buildEncoderFixture } from "./testing";

describe("WindowGeometry", () => {
  it("returns raw claimed / receiver positions, range and bbox for real rows only", () => {
    const fx = buildEncoderFixture([{ n: 3, fill: (t, j) => (j === 0 ? t : j === 10 ? 2 : 0) }]);
    const ds = EncoderDataset.fromParts(fx.manifest, fx.x);
    const g = WindowGeometry.of(ds, ds.windows[0]);
    expect(g.claimed.length).toBe(3);
    expect(g.claimed[2][0]).toBeCloseTo(2 * FIXTURE_STD[0] + FIXTURE_MEAN[0]);
    expect(g.claimed[0][1]).toBeCloseTo(FIXTURE_MEAN[1]);
    expect(g.receiver[1]).toEqual([FIXTURE_MEAN[2], FIXTURE_MEAN[3]]);
    expect(g.range[0]).toBeCloseTo(2 * FIXTURE_STD[10] + FIXTURE_MEAN[10]);
    expect(g.bbox).toEqual([FIXTURE_MEAN[0], FIXTURE_MEAN[1], FIXTURE_MEAN[2], FIXTURE_MEAN[3]]); // receiver x (300) exceeds claimed x (≤ 102)
  });
});
