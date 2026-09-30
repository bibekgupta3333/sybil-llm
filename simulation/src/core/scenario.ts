import type { BBox } from "./manifest";

/** One VeReMi run (`subfolder` × time `group`): its benign/attacker identity ids and combined bbox. */
export interface Scenario {
  readonly key: string;
  readonly subfolder: string;
  readonly group: "0709" | "1416";
  readonly benignIds: readonly number[];
  readonly attackerIds: readonly number[];
  /** Union of every member identity's position bbox. */
  readonly bbox: BBox;
}

export function scenarioKey(subfolder: string, group: string): string {
  return `${subfolder}::${group}`;
}
