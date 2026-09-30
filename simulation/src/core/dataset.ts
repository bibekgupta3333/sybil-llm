import { Identity } from "./identity";
import { FeatureIndex, type BBox, type ClassLabel, type IdentityRecord, type Manifest } from "./manifest";
import { scenarioKey, type Scenario } from "./scenario";
import { Viewport } from "./viewport";

/** One physical vehicle with both a prepared (ground-truth) identity and a raw-VeReMi companion. */
export interface PairedAttacker {
  readonly key: string;
  readonly sender: number | null;
  readonly subfolder: string;
  readonly group: "0709" | "1416";
  readonly label: ClassLabel;
  readonly realId: number;
  readonly attackId: number;
}

/** One-line observed pattern per attack type, from docs/research-notes/data_understanding/attack-taxonomy.md. */
export const ATTACK_SIGNATURES: Readonly<Record<string, string>> = {
  GridSybil: "Several ghost pseudonyms clustered around one point; position and claimed speed disagree.",
  DataReplaySybil: "Stale positions re-broadcast (frozen or repeated) while other fields keep updating.",
  DoSRandomSybil: "Incoherent position jumps at an elevated message rate.",
  DoSDisruptiveSybil: "Mostly single-shot disposable pseudonyms with bursty, irregular timing.",
};

/**
 * A fabricated broadcast and a Benign vehicle driving in the same run at the same time, both already
 * clipped to their shared time window. The two are different vehicles; neither is derived from the other.
 */
export interface BenignAttackMatch {
  readonly key: string;
  readonly subfolder: string;
  readonly group: "0709" | "1416";
  readonly label: ClassLabel;
  /** Raw dataset folder the pair comes from, e.g. `DoSRandomSybil_1416/VeReMi_50400_...`. */
  readonly folder: string;
  readonly attackTrace: "fake_pseudonym" | "all_pseudonyms";
  readonly pseudonymCount: number;
  readonly benign: Identity;
  readonly attack: Identity;
  readonly overlap: readonly [number, number];
}

/**
 * One benign vehicle against one or more attack traces from the same run, all clipped to one shared
 * window: the benign vehicle's span intersected with the selected attacks' combined span.
 */
export interface BenignMultiAttackView {
  readonly benign: Identity;
  readonly attacks: readonly Identity[];
  readonly window: readonly [number, number];
}

/** Steps of `identity` inside `[t0, t1]`. */
function stepsWithin(identity: Identity, t0: number, t1: number): number {
  let n = 0;
  for (let i = 0; i < identity.stepCount; i++) if (identity.times[i] >= t0 && identity.times[i] <= t1) n++;
  return n;
}

/**
 * The exported sample: manifest + two binaries, exposed as zero-copy `Identity` views.
 *
 * `load()` fetches from a base URL (the Vite `public/data` folder); `fromParts()` builds one from
 * in-memory arrays for tests. Both validate that identity offsets tile the binaries exactly.
 */
export class Dataset {
  readonly features: FeatureIndex;
  private readonly identityCache = new Map<number, Identity>();
  private scenarioCache: Scenario[] | null = null;
  private pairedCache: PairedAttacker[] | null = null;
  private matchCache: BenignAttackMatch[] | null = null;
  private overlapCache = new Map<number, Identity[]>();
  private viewCache = new Map<string, BenignMultiAttackView>();

  private constructor(
    readonly manifest: Manifest,
    private readonly steps: Float32Array,
    private readonly times: Float64Array,
  ) {
    this.features = new FeatureIndex(manifest.feature_cols);
    Dataset.validate(manifest, steps, times, this.features.count);
  }

  static fromParts(manifest: Manifest, steps: Float32Array, times: Float64Array): Dataset {
    return new Dataset(manifest, steps, times);
  }

  static async load(baseUrl = "data/"): Promise<Dataset> {
    const manifest = (await (await fetch(`${baseUrl}manifest.json`)).json()) as Manifest;
    const [stepsBuf, timesBuf] = await Promise.all([
      fetch(`${baseUrl}${manifest.binaries.steps.file}`).then((r) => r.arrayBuffer()),
      fetch(`${baseUrl}${manifest.binaries.times.file}`).then((r) => r.arrayBuffer()),
    ]);
    await Dataset.verifySha(stepsBuf, manifest.binaries.steps.sha256, "steps");
    await Dataset.verifySha(timesBuf, manifest.binaries.times.sha256, "times");
    return new Dataset(manifest, new Float32Array(stepsBuf), new Float64Array(timesBuf));
  }

  get identityCount(): number {
    return this.manifest.identities.length;
  }

  get records(): readonly IdentityRecord[] {
    return this.manifest.identities;
  }

  identity(id: number): Identity {
    const cached = this.identityCache.get(id);
    if (cached) return cached;
    const rec = this.manifest.identities[id];
    if (!rec) throw new RangeError(`no identity with id ${id}`);
    const f = this.features.count;
    const ident = new Identity(
      rec,
      this.steps.subarray(rec.row_offset * f, (rec.row_offset + rec.n_steps) * f),
      this.times.subarray(rec.row_offset, rec.row_offset + rec.n_steps),
      this.features,
      this.manifest.stride,
      this.manifest.window_size,
    );
    this.identityCache.set(id, ident);
    return ident;
  }

  identities(): Identity[] {
    return this.manifest.identities.map((r) => this.identity(r.id));
  }

  /** All identities sharing a physical vehicle within one run — the Sybil multiplicity groups. */
  senderGroups(minSize = 2): Map<string, Identity[]> {
    const groups = new Map<string, Identity[]>();
    for (const ident of this.identities()) {
      if (!ident.isAttacker) continue;
      const list = groups.get(ident.senderKey) ?? [];
      list.push(ident);
      groups.set(ident.senderKey, list);
    }
    for (const [key, list] of groups) if (list.length < minSize) groups.delete(key);
    return groups;
  }

  /** Groups identities by (subfolder, group) — one VeReMi run — computed once and cached. */
  scenarios(): Scenario[] {
    if (this.scenarioCache) return this.scenarioCache;
    interface Building {
      subfolder: string;
      group: "0709" | "1416";
      benign: number[];
      attackers: number[];
      bbox: BBox;
    }
    const byKey = new Map<string, Building>();
    for (const ident of this.identities()) {
      const r = ident.record;
      const key = scenarioKey(r.subfolder, r.group);
      const box = ident.bbox();
      const entry = byKey.get(key);
      if (!entry) {
        byKey.set(key, { subfolder: r.subfolder, group: r.group, benign: [], attackers: [], bbox: box });
      } else {
        entry.bbox = Viewport.union(entry.bbox, box);
      }
      (ident.isAttacker ? byKey.get(key)!.attackers : byKey.get(key)!.benign).push(ident.id);
    }
    this.scenarioCache = [...byKey.entries()].map(([key, e]) => ({
      key,
      subfolder: e.subfolder,
      group: e.group,
      benignIds: e.benign,
      attackerIds: e.attackers,
      bbox: e.bbox,
    }));
    return this.scenarioCache;
  }

  scenario(key: string): Scenario | undefined {
    return this.scenarios().find((s) => s.key === key);
  }

  /**
   * Every attack trace from the same run folder that overlaps benign identity `benignId` by at least
   * `minSteps` steps on both sides -- the candidates for "several attacks against one benign vehicle".
   * Sorted by overlap, longest first. Cached per benign id.
   */
  attacksOverlapping(benignId: number, minSteps = 30): Identity[] {
    const cached = this.overlapCache.get(benignId);
    if (cached) return cached;
    const benign = this.identity(benignId);
    const b = benign.record;
    const scored: { attack: Identity; seconds: number }[] = [];
    for (const attack of this.identities()) {
      const r = attack.record;
      if (r.data_source !== "raw_veremi" || r.subfolder !== b.subfolder || r.group !== b.group) continue;
      const t0 = Math.max(b.t_start, r.t_start);
      const t1 = Math.min(b.t_end, r.t_end);
      if (t1 <= t0 || stepsWithin(benign, t0, t1) < minSteps || stepsWithin(attack, t0, t1) < minSteps) continue;
      scored.push({ attack, seconds: t1 - t0 });
    }
    const result = scored.sort((x, y) => y.seconds - x.seconds).map((s) => s.attack);
    this.overlapCache.set(benignId, result);
    return result;
  }

  /**
   * `match` plus any extra attack traces (ids from `attacksOverlapping`), clipped to one shared window.
   * With no extras this is exactly the match's own clipped pair. Cached, so the same selection always
   * returns the same Identity objects (the renderers compare by reference).
   */
  benignMultiAttackView(match: BenignAttackMatch, extraAttackIds: readonly number[]): BenignMultiAttackView {
    if (extraAttackIds.length === 0) return { benign: match.benign, attacks: [match.attack], window: match.overlap };
    const key = `${match.key}|${[...extraAttackIds].sort((a, b) => a - b).join(",")}`;
    const cached = this.viewCache.get(key);
    if (cached) return cached;
    const benign = this.identity(match.benign.record.id);
    const attacks = [this.identity(match.attack.record.id), ...extraAttackIds.map((id) => this.identity(id))];
    const t0 = Math.max(benign.record.t_start, Math.min(...attacks.map((a) => a.record.t_start)));
    const t1 = Math.min(benign.record.t_end, Math.max(...attacks.map((a) => a.record.t_end)));
    const view: BenignMultiAttackView = {
      benign: benign.timeSlice(t0, t1),
      attacks: attacks.map((a) => a.timeSlice(t0, t1)),
      window: [t0, t1],
    };
    this.viewCache.set(key, view);
    return view;
  }

  colorFor(label: string): string {
    return this.manifest.class_palette[label] ?? "#64748B";
  }

  /**
   * Physical vehicles that have both a prepared (ground-truth) identity and a raw-VeReMi
   * (fabricated-broadcast) companion — the pairs the "Map compare" mode picks from. Cached like
   * `scenarios()`.
   */
  pairedAttackers(): PairedAttacker[] {
    if (this.pairedCache) return this.pairedCache;
    const pairs: PairedAttacker[] = [];
    for (const ident of this.identities()) {
      if (ident.dataSource !== "prepared" || ident.pairedId === null) continue;
      const r = ident.record;
      pairs.push({
        key: `${r.subfolder}/${r.sender ?? "?"}`,
        sender: r.sender,
        subfolder: r.subfolder,
        group: r.group,
        label: r.label,
        realId: ident.id,
        attackId: ident.pairedId,
      });
    }
    this.pairedCache = pairs;
    return this.pairedCache;
  }

  /**
   * Every fabricated broadcast with a time-matched Benign vehicle, both clipped to their overlap.
   * Clipped identities are built once and cached, so the same objects come back on every call.
   */
  benignAttackMatches(): BenignAttackMatch[] {
    if (this.matchCache) return this.matchCache;
    const matches: BenignAttackMatch[] = [];
    for (const attack of this.identities()) {
      const r = attack.record;
      if (r.benign_match_id === null || r.overlap === null) continue;
      const [t0, t1] = r.overlap;
      matches.push({
        key: `${attack.id}`,
        subfolder: r.subfolder,
        group: r.group,
        label: r.label,
        folder: `${r.family}_${r.group}/${r.subfolder}`,
        attackTrace: r.attack_trace ?? "fake_pseudonym",
        pseudonymCount: r.broadcast_pseudonyms ?? 1,
        benign: this.identity(r.benign_match_id).timeSlice(t0, t1),
        attack: attack.timeSlice(t0, t1),
        overlap: r.overlap,
      });
    }
    this.matchCache = matches;
    return this.matchCache;
  }

  private static validate(manifest: Manifest, steps: Float32Array, times: Float64Array, featureCount: number): void {
    if (steps.length !== manifest.total_steps * featureCount) {
      throw new Error(`steps binary has ${steps.length} values, manifest implies ${manifest.total_steps * featureCount}`);
    }
    if (times.length !== manifest.total_steps) {
      throw new Error(`times binary has ${times.length} values, manifest says ${manifest.total_steps}`);
    }
    let expectedOffset = 0;
    for (const rec of manifest.identities) {
      if (rec.row_offset !== expectedOffset) {
        throw new Error(`identity ${rec.id}: row_offset ${rec.row_offset} does not follow the previous identity (${expectedOffset})`);
      }
      expectedOffset += rec.n_steps;
    }
    if (expectedOffset !== manifest.total_steps) {
      throw new Error(`identities cover ${expectedOffset} steps, manifest says ${manifest.total_steps}`);
    }
  }

  private static async verifySha(buf: ArrayBuffer, expected: string, name: string): Promise<void> {
    if (typeof crypto === "undefined" || !crypto.subtle) return; // non-secure context: skip, keep working
    const digest = await crypto.subtle.digest("SHA-256", buf);
    const hex = Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
    if (hex !== expected) throw new Error(`${name} binary sha256 mismatch: manifest ${expected}, file ${hex}`);
  }
}
