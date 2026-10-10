import { describe, expect, it } from "vitest";

import { modelIsRight, modelSaysAttack, outcome, pickRandom, shuffled } from "./quiz";
import type { DetectionWindow } from "./types";

const win = (flag: boolean, label: number): DetectionWindow =>
  ({ i: 0, id: "x", scenario: "s", class: label ? "GridSybil" : "Benign", label, n: 3, offset: 0, score: 0.1, flag, rank_pct: 50 }) as DetectionWindow;

describe("quiz", () => {
  it("reads the model's call from the flag", () => {
    expect(modelSaysAttack(win(true, 0))).toBe(true);
    expect(modelSaysAttack(win(false, 1))).toBe(false);
  });

  it("scores the call against the label", () => {
    expect(outcome(win(true, 1))).toBe("true positive");
    expect(outcome(win(true, 0))).toBe("false positive");
    expect(outcome(win(false, 0))).toBe("true negative");
    expect(outcome(win(false, 1))).toBe("false negative");
    expect(modelIsRight(win(true, 1))).toBe(true);
    expect(modelIsRight(win(false, 1))).toBe(false);
  });

  it("shuffles deterministically and keeps every item", () => {
    const xs = Array.from({ length: 50 }, (_, i) => i);
    const a = shuffled(xs, 0);
    expect(a).toEqual(shuffled(xs, 0));
    expect([...a].sort((p, q) => p - q)).toEqual(xs);
    expect(a).not.toEqual(xs);
  });

  it("picks from the list or returns null", () => {
    expect(pickRandom([], () => 0.5)).toBeNull();
    expect(pickRandom(["a", "b", "c"], () => 0.99)).toBe("c");
    expect(pickRandom(["a", "b", "c"], () => 0)).toBe("a");
  });
});
