import type { ClassMetrics } from "./types";

/** Short, readable class names for the UI (the data keeps the dataset names). */
const NAMES: Readonly<Record<string, string>> = {
  Benign: "Normal traffic",
  GridSybil: "Grid Sybil",
  DataReplaySybil: "Data Replay",
  DoSRandomSybil: "DoS Random",
  DoSDisruptiveSybil: "DoS Disruptive",
};

export function friendlyClass(cls: string): string {
  return NAMES[cls] ?? cls;
}

/** A number for people: `digits` decimals, "—" when missing. */
export function fmt(v: unknown, digits = 3): string {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";
}

/** A share as a whole percent ("12%"), "—" when missing. */
export function pct(v: unknown): string {
  return typeof v === "number" && Number.isFinite(v) ? `${Math.round(v * 100)}%` : "—";
}

/** Plain name of a prediction outcome. */
export function outcomePhrase(o: "true positive" | "false positive" | "true negative" | "false negative"): string {
  return {
    "true positive": "attack caught",
    "false positive": "false alarm",
    "true negative": "correctly left alone",
    "false negative": "attack missed",
  }[o];
}

/** Verdict on one attack class: does the model separate it from normal traffic better than "short window = attack"? */
export function beatsLengthRule(m: ClassMetrics | null | undefined): boolean | null {
  const a = m?.auroc_vs_benign;
  const b = m?.length_baseline_auroc;
  if (typeof a !== "number" || typeof b !== "number") return null;
  return a > b;
}
