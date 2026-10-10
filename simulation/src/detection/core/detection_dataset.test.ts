import { describe, expect, it } from "vitest";

import { DetectionDataset, fmtMetric } from "./detection_dataset";
import { DET_FEATURES, buildDetectionFixture, fixtureValue, standardDetectionFixture } from "./testing";
import { orderClasses } from "./types";

describe("DetectionDataset", () => {
  const fx = standardDetectionFixture();
  const ds = DetectionDataset.fromParts(fx.manifest, fx.x);
  const f = DET_FEATURES.length;

  it("parses the manifest and indexes windows", () => {
    expect(ds.windows.length).toBe(5);
    expect(ds.nFeatures).toBe(13);
    expect(ds.nRows).toBe(12);
    expect(ds.byId("w003")?.class).toBe("Benign");
    expect(ds.byId("missing")).toBeUndefined();
    expect(ds.unit("range")).toBe("m");
    expect(ds.unit("nope")).toBe("");
  });

  it("slices each window's own rows from the row offset", () => {
    const w = ds.byId("w003")!; // offset 6, n = 3
    expect(w.offset).toBe(6);
    const r = ds.rows(w);
    expect(r.length).toBe(3 * f);
    expect(r[0]).toBe(fixtureValue(3, 0, 0));
    expect(r[2 * f + 12]).toBe(fixtureValue(3, 2, 12));
    expect(ds.column(ds.byId("w002")!, "bearing")).toEqual([fixtureValue(2, 0, 11)]);
    expect(ds.column(w, "missing_feature")).toBeNull();
  });

  it("builds claimed vs receiver positions with a joint bounding box", () => {
    const p = ds.positions(ds.byId("w001")!)!;
    expect(p.claimed).toEqual([[1000, 1001], [1010, 1011], [1020, 1021]]);
    expect(p.receiver).toEqual([[1002, 1003], [1012, 1013], [1022, 1023]]);
    expect(p.range).toEqual([1010, 1020, 1030]);
    expect(p.bbox).toEqual([1000, 1001, 1022, 1023]);
  });

  it("filters by class and flag and sorts by score", () => {
    expect(ds.query().map((w) => w.id)).toEqual(["w001", "w003", "w002", "w004", "w000"]);
    expect(ds.query({ sort: "score_asc" }).map((w) => w.id)).toEqual(["w000", "w004", "w002", "w003", "w001"]);
    expect(ds.query({ sort: "index" }).map((w) => w.i)).toEqual([0, 1, 2, 3, 4]);
    expect(ds.query({ cls: "Benign" }).map((w) => w.id)).toEqual(["w003", "w000"]);
    expect(ds.query({ flag: "flagged" }).map((w) => w.id)).toEqual(["w001", "w003", "w002"]);
    expect(ds.query({ cls: "GridSybil", flag: "unflagged" }).map((w) => w.id)).toEqual(["w004"]);
  });

  it("lists classes in display order, including metric-only and unknown classes", () => {
    expect(ds.classes()).toEqual(["Benign", "GridSybil", "DoSRandomSybil"]);
    expect(orderClasses(["Zeta", "DoSDisruptiveSybil", "Alpha", "Benign"])).toEqual(["Benign", "DoSDisruptiveSybil", "Alpha", "Zeta"]);
  });

  it("summarises window lengths per class", () => {
    expect(ds.lengths("Benign")).toEqual({ count: 2, min: 2, median: 2.5, max: 3, mean: 2.5 });
    expect(ds.lengths("DoSRandomSybil")).toEqual({ count: 1, min: 1, median: 1, max: 1, mean: 1 });
    expect(ds.lengths("DataReplaySybil")).toBeNull();
  });

  it("returns metrics rows verbatim, null for missing classes and absent blocks", () => {
    const rows = ds.metricsRows("sample")!;
    expect(rows.map((r) => r.cls)).toEqual(["Benign", "GridSybil", "DoSRandomSybil"]);
    expect(rows[1].metrics?.auroc_vs_benign).toBe(0.75);
    expect(rows[2].metrics).toBeNull();
    expect(ds.metricsRows("full_test_1416")).toBeNull();
    expect(fmtMetric(0.7123456)).toBe("0.7123456");
    expect(fmtMetric(null)).toBe("—");
    expect(fmtMetric(undefined)).toBe("—");
    expect(fmtMetric(Number.NaN)).toBe("—");
    expect(fmtMetric(300)).toBe("300");
  });

  it("rejects inconsistent bundles", () => {
    expect(() => DetectionDataset.fromParts(fx.manifest, fx.x.subarray(f))).toThrow(/rows/);
    expect(() => DetectionDataset.fromParts(fx.manifest, fx.x.subarray(1))).toThrow(/multiple/);
    const ws = fx.manifest.windows;
    const shifted = { ...fx.manifest, windows: [ws[0], { ...ws[1], offset: ws[1].offset + 1 }, ...ws.slice(2)] };
    expect(() => DetectionDataset.fromParts(shifted, fx.x)).toThrow(/offset/);
    const dup = { ...fx.manifest, windows: [ws[0], { ...ws[1], id: ws[0].id }, ...ws.slice(2)] };
    expect(() => DetectionDataset.fromParts(dup, fx.x)).toThrow(/duplicate/);
    expect(() => DetectionDataset.fromParts({ ...fx.manifest, schema: 2 }, fx.x)).toThrow(/schema/);
    const long = buildDetectionFixture([{ n: 25, cls: "Benign", score: 0 }]);
    expect(() => DetectionDataset.fromParts(long.manifest, long.x)).toThrow(/seq_len/);
  });
});
