import type { EncoderDataset } from "./encoder_dataset";
import type { ClassName, WindowMeta } from "./types";
import { scenarioGroup } from "./types";

export type GroupKind = "link" | "sender" | "receiver" | "batch";

export interface GroupSummary {
  readonly key: string;
  readonly label: string;
  readonly size: number;
  readonly classes: Readonly<Record<ClassName, number>>;
}

/** Deterministic PRNG (mulberry32). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Groups windows of a pool.
 *
 * - `link`: one receiver hearing one sender pseudonym (its crops in order).
 * - `sender`: one physical vehicle within a time group (0709 / 1416), the unit of the split (D8).
 *   For GridSybil ghosts this grouping uses the true sender and is privileged (F13).
 * - `receiver`: one receiver trace file.
 * - `batch`: a seeded random batch; its key is `batch:<seed>`.
 */
export class Grouper {
  static readonly BATCH_SIZE = 32;

  constructor(private readonly ds: EncoderDataset) {}

  keyOf(kind: Exclude<GroupKind, "batch">, w: WindowMeta): string {
    switch (kind) {
      case "link":
        return w.link;
      case "sender":
        return `${scenarioGroup(w.scenario)}|${w.sender}`;
      case "receiver":
        return `${w.scenario}|${w.run}|${w.receiver_file}`;
    }
  }

  groups(kind: GroupKind, pool: readonly WindowMeta[]): GroupSummary[] {
    if (kind === "batch") {
      const members = this.batch(pool);
      return members.length ? [{ key: "batch:0", label: `random batch of ${members.length} (seed 0)`,
        size: members.length, classes: countClasses(members) }] : [];
    }
    const byKey = new Map<string, WindowMeta[]>();
    for (const w of pool) {
      const key = this.keyOf(kind, w);
      const list = byKey.get(key);
      if (list) list.push(w);
      else byKey.set(key, [w]);
    }
    const out: GroupSummary[] = [];
    for (const [key, ws] of byKey) {
      out.push({ key, label: labelOf(kind, ws[0]), size: ws.length, classes: countClasses(ws) });
    }
    return out.sort((a, b) => b.size - a.size || (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
  }

  members(kind: GroupKind, key: string, pool: readonly WindowMeta[]): WindowMeta[] {
    if (kind === "batch") {
      const seed = Number(key.startsWith("batch:") ? key.slice(6) : 0);
      return this.batch(pool, Grouper.BATCH_SIZE, Number.isFinite(seed) ? seed : 0);
    }
    const ws = pool.filter((w) => this.keyOf(kind, w) === key);
    const cmp = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);
    if (kind === "link") return ws.sort((a, b) => a.k - b.k || a.i - b.i);
    if (kind === "sender") {
      return ws.sort((a, b) => cmp(a.receiver_file, b.receiver_file) || cmp(a.link, b.link) || a.k - b.k || a.i - b.i);
    }
    return ws.sort((a, b) => a.sender - b.sender || cmp(a.link, b.link) || a.k - b.k || a.i - b.i);
  }

  /** `size` windows drawn without replacement with a seeded shuffle; independent of pool order. */
  batch(pool: readonly WindowMeta[], size = Grouper.BATCH_SIZE, seed = 0): WindowMeta[] {
    const ws = [...pool].sort((a, b) => a.i - b.i);
    const rand = mulberry32(seed);
    for (let i = ws.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [ws[i], ws[j]] = [ws[j], ws[i]];
    }
    return ws.slice(0, Math.min(size, ws.length));
  }

  get dataset(): EncoderDataset {
    return this.ds;
  }
}

function countClasses(ws: readonly WindowMeta[]): Record<ClassName, number> {
  const c: Record<ClassName, number> = { Benign: 0, GridSybil: 0 };
  for (const w of ws) c[w.label_name]++;
  return c;
}

function labelOf(kind: Exclude<GroupKind, "batch">, w: WindowMeta): string {
  switch (kind) {
    case "link":
      return `receiver ${w.receiver} ← pseudonym ${w.sender_pseudo} (${w.scenario})`;
    case "sender":
      return `vehicle ${w.sender} (${scenarioGroup(w.scenario)})`;
    case "receiver":
      return `receiver ${w.receiver} (${w.scenario}, run ${w.run})`;
  }
}
