import type { GroupKind } from "./grouping";
import type { ClassName, Split } from "./types";
import { SPLITS } from "./types";

/**
 * View state encoded in the URL hash, e.g.
 * `#encoder?split=test&scenario=all&mode=single&group=link&key=…&win=<id>&raw=0&labels=1`.
 */
export interface EncoderRoute {
  split: Split;
  /** "all", a time group ("0709" / "1416") or a scenario folder name. */
  scenario: string;
  /** "all" or a run name. */
  run: string;
  cls: ClassName | "all";
  mode: "single" | "group";
  group: GroupKind;
  /** Group key ("" = none selected). */
  key: string;
  /** Window id ("" = none selected). */
  win: string;
  raw: boolean;
  labels: boolean;
}

/** Defaults: test split, all scenarios/runs/classes, single mode, link grouping, normalised values, labels on. */
export const DEFAULT_ROUTE: Readonly<EncoderRoute> = {
  split: "test",
  scenario: "all",
  run: "all",
  cls: "all",
  mode: "single",
  group: "link",
  key: "",
  win: "",
  raw: false,
  labels: true,
};

const GROUPS: readonly GroupKind[] = ["link", "sender", "receiver", "batch"];

function pick<T extends string>(value: string | null, allowed: readonly T[], fallback: T): T {
  return value !== null && (allowed as readonly string[]).includes(value) ? (value as T) : fallback;
}

function flag(value: string | null, fallback: boolean): boolean {
  if (value === "1" || value === "true") return true;
  if (value === "0" || value === "false") return false;
  return fallback;
}

/** Parses `#encoder[?…]`; returns null for any other hash. Unknown or invalid values fall back to defaults. */
export function parseRoute(hash: string): EncoderRoute | null {
  const h = hash.startsWith("#") ? hash.slice(1) : hash;
  const q = h.indexOf("?");
  const path = q < 0 ? h : h.slice(0, q);
  if (path !== "encoder") return null;
  const p = new URLSearchParams(q < 0 ? "" : h.slice(q + 1));
  const d = DEFAULT_ROUTE;
  return {
    split: pick(p.get("split"), SPLITS, d.split),
    scenario: p.get("scenario") || d.scenario,
    run: p.get("run") || d.run,
    cls: pick(p.get("cls"), ["all", "Benign", "GridSybil"] as const, d.cls),
    mode: pick(p.get("mode"), ["single", "group"] as const, d.mode),
    group: pick(p.get("group"), GROUPS, d.group),
    key: p.get("key") ?? d.key,
    win: p.get("win") ?? d.win,
    raw: flag(p.get("raw"), d.raw),
    labels: flag(p.get("labels"), d.labels),
  };
}

/** Formats a route as `#encoder?…`; `parseRoute(formatRoute(r))` round-trips. */
export function formatRoute(r: EncoderRoute): string {
  const p = new URLSearchParams();
  p.set("split", r.split);
  p.set("scenario", r.scenario);
  p.set("run", r.run);
  p.set("cls", r.cls);
  p.set("mode", r.mode);
  p.set("group", r.group);
  if (r.key) p.set("key", r.key);
  if (r.win) p.set("win", r.win);
  p.set("raw", r.raw ? "1" : "0");
  p.set("labels", r.labels ? "1" : "0");
  return `#encoder?${p.toString()}`;
}
