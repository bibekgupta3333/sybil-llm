import { describe, expect, it } from "vitest";

import { DEFAULT_ROUTE, formatRoute, parseRoute } from "./route";

describe("route", () => {
  it("ignores non-encoder hashes", () => {
    expect(parseRoute("")).toBeNull();
    expect(parseRoute("#window")).toBeNull();
    expect(parseRoute("#encoderx")).toBeNull();
  });

  it("applies defaults to a bare #encoder", () => {
    expect(parseRoute("#encoder")).toEqual(DEFAULT_ROUTE);
    expect(parseRoute("#encoder?split=bogus&raw=maybe")).toEqual(DEFAULT_ROUTE);
  });

  it("parses the documented example and round-trips", () => {
    const r = parseRoute("#encoder?split=val&scenario=1416&mode=group&group=sender&key=0709%7C7&win=w001&raw=1&labels=0")!;
    expect(r).toMatchObject({ split: "val", scenario: "1416", mode: "group", group: "sender", key: "0709|7",
      win: "w001", raw: true, labels: false, run: "all", cls: "all" });
    expect(parseRoute(formatRoute(r))).toEqual(r);
    const tricky = { ...DEFAULT_ROUTE, key: "GridSybil_0709|run a|f.json|7|1", cls: "GridSybil" as const };
    expect(parseRoute(formatRoute(tricky))).toEqual(tricky);
  });
});
