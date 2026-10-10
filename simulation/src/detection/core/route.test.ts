import { describe, expect, it } from "vitest";

import { DEFAULT_DETECTION_ROUTE, formatDetectionRoute, parseDetectionRoute } from "./route";

describe("detection route", () => {
  it("parses only detection hashes", () => {
    expect(parseDetectionRoute("#encoder?split=test")).toBeNull();
    expect(parseDetectionRoute("")).toBeNull();
    expect(parseDetectionRoute("#detectionX")).toBeNull();
    expect(parseDetectionRoute("#detection")).toEqual(DEFAULT_DETECTION_ROUTE);
  });

  it("round-trips and drops defaults", () => {
    const r = { cls: "DoSRandomSybil", flag: "flagged" as const, sort: "score_asc" as const, win: "a|b c", blind: false };
    expect(parseDetectionRoute(formatDetectionRoute(r))).toEqual(r);
    expect(formatDetectionRoute({ ...DEFAULT_DETECTION_ROUTE })).toBe("#detection");
    expect(formatDetectionRoute({ ...DEFAULT_DETECTION_ROUTE, blind: false })).toBe("#detection?blind=0");
  });

  it("falls back to defaults on unknown values", () => {
    expect(parseDetectionRoute("#detection?flag=maybe&sort=random")).toEqual(DEFAULT_DETECTION_ROUTE);
  });
});
