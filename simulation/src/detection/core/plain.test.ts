import { describe, expect, it } from "vitest";

import { beatsLengthRule, fmt, friendlyClass, outcomePhrase, pct } from "./plain";

describe("plain-language helpers", () => {
  it("names classes for people and keeps unknown names", () => {
    expect(friendlyClass("DoSRandomSybil")).toBe("DoS Random");
    expect(friendlyClass("Benign")).toBe("Normal traffic");
    expect(friendlyClass("Other")).toBe("Other");
  });

  it("formats numbers and shares", () => {
    expect(fmt(0.1867558181285857)).toBe("0.187");
    expect(fmt(null)).toBe("—");
    expect(pct(0.12107)).toBe("12%");
    expect(pct(undefined)).toBe("—");
  });

  it("phrases outcomes", () => {
    expect(outcomePhrase("false negative")).toBe("attack missed");
    expect(outcomePhrase("false positive")).toBe("false alarm");
  });

  it("compares the model with the length-only rule", () => {
    expect(beatsLengthRule({ auroc_vs_benign: 0.61, length_baseline_auroc: 0.52 })).toBe(true);
    expect(beatsLengthRule({ auroc_vs_benign: 0.14, length_baseline_auroc: 0.96 })).toBe(false);
    expect(beatsLengthRule({ auroc_vs_benign: null })).toBeNull();
  });
});
