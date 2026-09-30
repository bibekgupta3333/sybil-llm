import { describe, expect, it } from "vitest";

import { clampSidebarWidth } from "./sidebar_resizer";

describe("clampSidebarWidth", () => {
  it("passes through a value already inside the usable range", () => {
    expect(clampSidebarWidth(400)).toBe(400);
  });

  it("clamps below the minimum", () => {
    expect(clampSidebarWidth(50)).toBe(260);
  });

  it("clamps above the maximum", () => {
    expect(clampSidebarWidth(10_000)).toBe(640);
  });

  it("rounds to the nearest pixel", () => {
    expect(clampSidebarWidth(400.6)).toBe(401);
  });
});
