import { describe, expect, it } from "vitest";

import { EncoderDataset } from "./encoder_dataset";
import { Grouper } from "./grouping";
import { buildEncoderFixture, standardFixture } from "./testing";

describe("Grouper", () => {
  const fx = standardFixture();
  const ds = EncoderDataset.fromParts(fx.manifest, fx.x);
  const g = new Grouper(ds);
  const pool = ds.windows;

  it("groups by link, largest first, members ordered by crop index", () => {
    const groups = g.groups("link", pool);
    expect(groups[0].size).toBe(3);
    expect(groups[0].classes).toEqual({ Benign: 0, GridSybil: 3 });
    const shuffled = [pool[2], pool[0], pool[1], pool[3]];
    expect(g.members("link", groups[0].key, shuffled).map((w) => w.k)).toEqual([0, 1, 2]);
    expect(groups.reduce((s, x) => s + x.size, 0)).toBe(pool.length);
  });

  it("groups senders within a time group and receivers by trace file", () => {
    const senders = g.groups("sender", pool);
    expect(senders[0]).toMatchObject({ key: "0709|7", size: 3 });
    const recv = g.groups("receiver", pool);
    const r10 = recv.find((r) => r.key.endsWith("traceJSON-10-9-A0-0-0.json") && r.key.startsWith("GridSybil_0709"))!;
    expect(r10.size).toBe(4);
    expect(g.members("receiver", r10.key, pool).map((w) => w.sender)).toEqual([7, 7, 7, 8]);
  });

  it("draws a deterministic seeded batch independent of pool order", () => {
    const big = buildEncoderFixture(Array.from({ length: 100 }, (_, i) => ({ n: 1 + (i % 64) })));
    const bds = EncoderDataset.fromParts(big.manifest, big.x);
    const bg = new Grouper(bds);
    const a = bg.batch(bds.windows, 32, 0).map((w) => w.id);
    const b = bg.batch([...bds.windows].reverse(), 32, 0).map((w) => w.id);
    expect(a).toEqual(b);
    expect(new Set(a).size).toBe(32);
    expect(bg.batch(bds.windows, 32, 1).map((w) => w.id)).not.toEqual(a);
    expect(bg.members("batch", "batch:0", bds.windows).map((w) => w.id)).toEqual(a);
    expect(bg.groups("batch", bds.windows)[0].size).toBe(32);
    expect(g.batch(pool).length).toBe(pool.length);
  });
});
