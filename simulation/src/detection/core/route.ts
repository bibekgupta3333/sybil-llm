import type { FlagFilter, SortOrder } from "./detection_dataset";

/** URL state of the detection tab: `#detection?cls=…&flag=…&sort=…&win=…&blind=0`. */
export interface DetectionRoute {
  cls: string;
  flag: FlagFilter;
  sort: SortOrder;
  win: string;
  /** "Test yourself" mode: the true class (and anything that gives it away) stays hidden until revealed. */
  blind: boolean;
}

export const DETECTION_PREFIX = "#detection";

export const DEFAULT_DETECTION_ROUTE: Readonly<DetectionRoute> = { cls: "all", flag: "all", sort: "score_desc", win: "", blind: true };

const FLAGS: readonly FlagFilter[] = ["all", "flagged", "unflagged"];
const SORTS: readonly SortOrder[] = ["score_desc", "score_asc", "index"];

/** Parses a location hash; null if it is not a detection hash. Unknown values fall back to the defaults. */
export function parseDetectionRoute(hash: string): DetectionRoute | null {
  if (!hash.startsWith(DETECTION_PREFIX)) return null;
  const rest = hash.slice(DETECTION_PREFIX.length);
  if (rest !== "" && !rest.startsWith("?")) return null;
  const p = new URLSearchParams(rest.slice(1));
  const flag = p.get("flag") as FlagFilter | null;
  const sort = p.get("sort") as SortOrder | null;
  return {
    cls: p.get("cls") || DEFAULT_DETECTION_ROUTE.cls,
    flag: flag && FLAGS.includes(flag) ? flag : DEFAULT_DETECTION_ROUTE.flag,
    sort: sort && SORTS.includes(sort) ? sort : DEFAULT_DETECTION_ROUTE.sort,
    win: p.get("win") ?? "",
    blind: p.get("blind") !== "0",
  };
}

/** Formats a route; default values are left out so the plain tab link stays `#detection`. */
export function formatDetectionRoute(r: DetectionRoute): string {
  const p = new URLSearchParams();
  if (r.cls !== DEFAULT_DETECTION_ROUTE.cls) p.set("cls", r.cls);
  if (r.flag !== DEFAULT_DETECTION_ROUTE.flag) p.set("flag", r.flag);
  if (r.sort !== DEFAULT_DETECTION_ROUTE.sort) p.set("sort", r.sort);
  if (r.win) p.set("win", r.win);
  if (!r.blind) p.set("blind", "0");
  const qs = p.toString();
  return qs ? `${DETECTION_PREFIX}?${qs}` : DETECTION_PREFIX;
}
