import type { ClassName, Split } from "../core/types";
import { h } from "./dom";

export const SPLIT_LABEL: Record<Split, string> = {
  train: "train",
  pretrain_val: "pretrain val",
  val: "val",
  test: "test",
};

export const CLASS_COLOR: Record<ClassName, string> = {
  Benign: "var(--green)",
  GridSybil: "var(--red)",
};

/** Class chip (only rendered when labels are shown). */
export function classChip(cls: ClassName): HTMLElement {
  const chip = h("span", { class: "chip" }, cls);
  chip.style.setProperty("--c", CLASS_COLOR[cls]);
  return chip;
}

/** The ghost pseudonym shared by every GridSybil attacker in a run (F13). */
export function isSharedGhostPseudonym(scenario: string, pseudo: number): boolean {
  return scenario.startsWith("GridSybil") && pseudo === 1;
}

export const EVAL_ONLY_BADGE = "evaluation only — the encoder never sees labels";
