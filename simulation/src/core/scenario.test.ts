import { describe, expect, it } from "vitest";

import { Dataset } from "./dataset";
import { scenarioKey } from "./scenario";
import { buildFixture } from "./testing";

describe("Dataset.scenarios", () => {
  const fx = buildFixture([
    { label: "Benign", nSteps: 30, subfolder: "runA", group: "0709" },
    { label: "GridSybil", nSteps: 20, sender: 7, subfolder: "runA", group: "0709" },
    { label: "GridSybil", nSteps: 20, sender: 8, subfolder: "runA", group: "0709" },
    { label: "Benign", nSteps: 20, subfolder: "runB", group: "1416" },
  ]);
  const ds = Dataset.fromParts(fx.manifest, fx.steps, fx.times);

  it("groups identities by (subfolder, group) and splits benign vs attacker", () => {
    const scenarios = ds.scenarios();
    expect(scenarios).toHaveLength(2);
    const runA = ds.scenario(scenarioKey("runA", "0709"));
    expect(runA).toBeDefined();
    expect(runA!.benignIds).toEqual([0]);
    expect(runA!.attackerIds).toEqual([1, 2]);
    const runB = ds.scenario(scenarioKey("runB", "1416"));
    expect(runB!.benignIds).toEqual([3]);
    expect(runB!.attackerIds).toEqual([]);
  });

  it("unions member bboxes and is cached across calls", () => {
    const runA = ds.scenario(scenarioKey("runA", "0709"))!;
    // straightDrive moves along +x from pos_x=100; the longest member (30 steps) reaches furthest.
    expect(runA.bbox.minX).toBeLessThanOrEqual(100);
    expect(runA.bbox.maxX).toBeGreaterThanOrEqual(100 + 12 * 29);
    expect(ds.scenarios()).toBe(ds.scenarios()); // same array instance: cached
  });

  it("returns undefined for an unknown key", () => {
    expect(ds.scenario("nope::0709")).toBeUndefined();
  });
});
