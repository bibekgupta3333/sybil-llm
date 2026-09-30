import { describe, expect, it } from "vitest";

import { Dataset } from "./dataset";
import { buildFixture } from "./testing";

describe("Dataset", () => {
  const fx = buildFixture([
    { label: "Benign", nSteps: 50 },
    { label: "GridSybil", nSteps: 20, sender: 7 },
    { label: "GridSybil", nSteps: 30, sender: 7 },
  ]);
  const ds = Dataset.fromParts(fx.manifest, fx.steps, fx.times);

  it("exposes zero-copy identity views at the right offsets", () => {
    const a = ds.identity(0);
    const c = ds.identity(2);
    expect(a.stepCount).toBe(50);
    expect(a.posX(0)).toBe(100);
    expect(a.posX(49)).toBe(100 + 12 * 49);
    expect(c.record.row_offset).toBe(70);
    expect(c.posX(0)).toBe(100); // its own first step, not identity 0's
    expect(c.times[0]).toBeGreaterThan(a.times[49]);
    expect(ds.identities().reduce((n, i) => n + i.stepCount, 0)).toBe(fx.manifest.total_steps);
  });

  it("derives window counts from the stride/window contract", () => {
    expect(ds.identity(0).windowCount).toBe(4); // steps 0..49 → starts 0,10,20,30
    expect(ds.identity(1).windowCount).toBe(1);
    expect(ds.identity(0).windowIndexOf(35)).toBe(3);
    expect(ds.identity(0).windowIndexOf(49)).toBe(3); // clamped to the last window
  });

  it("groups attacker pseudonyms by physical sender", () => {
    const groups = ds.senderGroups();
    expect(groups.size).toBe(1);
    expect([...groups.values()][0].map((i) => i.id)).toEqual([1, 2]);
  });

  it("rejects binaries that do not tile the identities", () => {
    const bad = { ...fx.manifest, total_steps: fx.manifest.total_steps + 1 };
    expect(() => Dataset.fromParts(bad, fx.steps, fx.times)).toThrow(/steps binary/);
    const gap = { ...fx.manifest, identities: fx.manifest.identities.map((r, i) => (i === 1 ? { ...r, row_offset: 51 } : r)) };
    expect(() => Dataset.fromParts(gap, fx.steps, fx.times)).toThrow(/row_offset/);
  });
});

describe("Dataset.pairedAttackers", () => {
  const fx = buildFixture([
    { label: "Benign", nSteps: 30 },
    { label: "GridSybil", nSteps: 20, sender: 7, dataSource: "prepared", pairedId: 2 },
    { label: "GridSybil", nSteps: 20, sender: 7, dataSource: "raw_veremi", pairedId: 1 },
    { label: "DataReplaySybil", nSteps: 20, sender: 9 }, // no pair
  ]);
  const ds = Dataset.fromParts(fx.manifest, fx.steps, fx.times);

  it("lists only physical vehicles with a resolved prepared/raw-VeReMi pair", () => {
    const pairs = ds.pairedAttackers();
    expect(pairs).toHaveLength(1);
    expect(pairs[0]).toMatchObject({ sender: 7, realId: 1, attackId: 2, label: "GridSybil" });
  });

  it("never pairs benign identities or unpaired attackers", () => {
    const pairs = ds.pairedAttackers();
    expect(pairs.some((p) => p.realId === 0 || p.attackId === 0)).toBe(false);
    expect(pairs.some((p) => p.realId === 3 || p.attackId === 3)).toBe(false);
  });

  it("caches the result across calls", () => {
    expect(ds.pairedAttackers()).toBe(ds.pairedAttackers());
  });
});

describe("Dataset.benignAttackMatches", () => {
  // buildFixture offsets each identity's clock by id*1000 s, so give both the same overlap window
  // expressed in each one's own time: benign (id 0) starts 25200, attack (id 1) starts 26200.
  const fx = buildFixture([
    { label: "Benign", nSteps: 3000 },
    { label: "GridSybil", nSteps: 60, dataSource: "raw_veremi", benignMatchId: 0, overlap: [26200, 26259] },
    { label: "GridSybil", nSteps: 40 },
  ]);
  const ds = Dataset.fromParts(fx.manifest, fx.steps, fx.times);

  it("pairs each matched attack with its benign vehicle, both clipped to the overlap", () => {
    const matches = ds.benignAttackMatches();
    expect(matches).toHaveLength(1);
    const m = matches[0];
    expect(m.attack.record.id).toBe(1);
    expect(m.benign.record.id).toBe(0);
    expect(m.benign.times[0]).toBeGreaterThanOrEqual(26200);
    expect(m.benign.times[m.benign.stepCount - 1]).toBeLessThanOrEqual(26259);
    expect(m.benign.stepCount).toBeLessThan(ds.identity(0).stepCount);
  });

  it("returns the same clipped objects on every call", () => {
    expect(ds.benignAttackMatches()[0].benign).toBe(ds.benignAttackMatches()[0].benign);
  });
});

describe("Dataset multi-attack views", () => {
  // buildFixture puts identity i on its own clock starting at 25200 + 1000*i, so identities 1-3 all
  // overlap the long benign (id 0, which runs 25200..28199), and identity 4 is in another run.
  const fx = buildFixture([
    { label: "Benign", nSteps: 3000, subfolder: "runA" },
    { label: "GridSybil", nSteps: 80, dataSource: "raw_veremi", subfolder: "runA", benignMatchId: 0, overlap: [26200, 26279] },
    { label: "GridSybil", nSteps: 60, dataSource: "raw_veremi", subfolder: "runA" },
    { label: "GridSybil", nSteps: 10, dataSource: "raw_veremi", subfolder: "runA" },  // too short to qualify
    { label: "GridSybil", nSteps: 80, dataSource: "raw_veremi", subfolder: "runB" },  // different run
  ]);
  const ds = Dataset.fromParts(fx.manifest, fx.steps, fx.times);

  it("lists only attack traces from the same run that overlap long enough, longest first", () => {
    expect(ds.attacksOverlapping(0).map((a) => a.id)).toEqual([1, 2]);
  });

  it("with no extras, returns the match's own clipped pair", () => {
    const match = ds.benignAttackMatches()[0];
    const view = ds.benignMultiAttackView(match, []);
    expect(view.benign).toBe(match.benign);
    expect(view.attacks).toEqual([match.attack]);
  });

  it("with extras, clips everything to benign ∩ (combined attack span) and caches the result", () => {
    const match = ds.benignAttackMatches()[0];
    const view = ds.benignMultiAttackView(match, [2]);
    expect(view.attacks.map((a) => a.id)).toEqual([1, 2]);
    const [t0, t1] = view.window;
    expect(t0).toBe(26200);                   // earliest selected attack start (id 1)
    expect(t1).toBe(ds.identity(2).record.t_end);  // latest selected attack end (id 2)
    expect(view.benign.times[0]).toBeGreaterThanOrEqual(t0);
    expect(view.benign.times[view.benign.stepCount - 1]).toBeLessThanOrEqual(t1);
    expect(ds.benignMultiAttackView(match, [2])).toBe(view);
  });
});
