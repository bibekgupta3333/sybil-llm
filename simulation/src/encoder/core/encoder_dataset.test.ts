import { describe, expect, it } from "vitest";

import { EncoderDataset } from "./encoder_dataset";
import { FIXTURE_MEAN, FIXTURE_STD, buildEncoderFixture, standardFixture } from "./testing";

describe("EncoderDataset", () => {
  const fx = standardFixture();
  const ds = EncoderDataset.fromParts(fx.manifest, fx.x);

  it("returns each window's own real rows", () => {
    const w = ds.byId("w002")!;
    const rows = ds.realRows(w);
    expect(rows.length).toBe(22 * 13);
    expect(rows[0]).toBeCloseTo(0);
    expect(rows[13 * 21 + 12]).toBeCloseTo(21.12);
    expect(ds.byId("missing")).toBeUndefined();
  });

  it("rebuilds the 64-row tensor with zero padding and the mask", () => {
    const w = ds.byId("w003")!; // n = 5
    const m = ds.matrix(w, false);
    expect(m.length).toBe(64 * 13);
    expect(m[4 * 13 + 1]).toBeCloseTo(4.01);
    expect([...m.subarray(5 * 13)].every((v) => v === 0)).toBe(true);
    const mask = ds.mask(w);
    expect([...mask].reduce((a, b) => a + b, 0)).toBe(5);
    expect(mask[4]).toBe(1);
    expect(mask[5]).toBe(0);
  });

  it("derives raw units from train mean/std and marks padding NaN", () => {
    const w = ds.byId("w003")!;
    const m = ds.matrix(w, true);
    expect(m[2 * 13 + 3]).toBeCloseTo((2 + 0.03) * FIXTURE_STD[3] + FIXTURE_MEAN[3], 3);
    expect(Number.isNaN(m[5 * 13])).toBe(true);
    expect(Number.isNaN(m[64 * 13 - 1])).toBe(true);
  });

  it("filters by split, scenario name or time group, run and class", () => {
    expect(ds.filter({}).length).toBe(7);
    expect(ds.filter({ splits: ["test"] }).length).toBe(5);
    expect(ds.filter({ scenarios: ["1416"] }).map((w) => w.id)).toEqual(["w005", "w006"]);
    expect(ds.filter({ scenarios: ["GridSybil_1416"] }).map((w) => w.id)).toEqual(["w006"]);
    expect(ds.filter({ classes: ["GridSybil"], splits: ["test"] }).length).toBe(3);
    expect(ds.filter({ runs: ["run_b"] }).length).toBe(1);
  });

  it("lists runs and scenarios", () => {
    expect(ds.scenarios()).toEqual(["DoSRandomSybil_1416", "GridSybil_0709", "GridSybil_1416"]);
    expect(ds.runs()).toEqual(["run_a", "run_b", "run_c"]);
    expect(ds.runs("test")).toEqual(["run_a"]);
    expect(ds.runs(undefined, "1416")).toEqual(["run_b", "run_c"]);
  });

  it("rejects inconsistent parts", () => {
    expect(() => EncoderDataset.fromParts(fx.manifest, fx.x.subarray(13))).toThrow();
    expect(() => EncoderDataset.fromParts(fx.manifest, fx.x.subarray(1))).toThrow();
    const bad = buildEncoderFixture([{ n: 3 }]);
    const tooLong = { ...bad.manifest, windows: [{ ...bad.manifest.windows[0], n: 65 }] };
    expect(() => EncoderDataset.fromParts(tooLong, new Float32Array(65 * 13))).toThrow();
  });
});
