import type { DetectionWindow } from "./types";

/** How far a "Test yourself" round has gone for the selected window. */
export type QuizStage = "hidden" | "detected" | "revealed";

/** The model's call for a window: flagged (score ≥ θ) means "attack". */
export function modelSaysAttack(w: DetectionWindow): boolean {
  return w.flag;
}

/** True when the model's call matches the evaluation label (label 1 = attack, 0 = benign). */
export function modelIsRight(w: DetectionWindow): boolean {
  return modelSaysAttack(w) === (w.label === 1);
}

/** Outcome name in the usual confusion-matrix terms. */
export function outcome(w: DetectionWindow): "true positive" | "false positive" | "true negative" | "false negative" {
  const attack = w.label === 1;
  if (modelSaysAttack(w)) return attack ? "true positive" : "false positive";
  return attack ? "false negative" : "true negative";
}

/** A seeded-free random pick from a list (UI only); null when the list is empty. */
export function pickRandom<T>(list: readonly T[], rand: () => number = Math.random): T | null {
  if (list.length === 0) return null;
  return list[Math.min(list.length - 1, Math.floor(rand() * list.length))];
}

/** Deterministic shuffle (mulberry32), so test mode never lists windows in their class-grouped sample order. */
export function shuffled<T>(list: readonly T[], seed = 0): T[] {
  let a = seed >>> 0;
  const rand = (): number => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = [...list];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}
