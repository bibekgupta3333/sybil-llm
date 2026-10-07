import type { ClassName, EncoderManifest, Split, WindowMeta } from "./types";
import { scenarioGroup } from "./types";

/**
 * Window filter. Every field is optional; an absent or empty list matches everything.
 * A `scenarios` entry matches a full scenario name (`GridSybil_1416`) or its time group (`1416`).
 */
export interface WindowFilter {
  splits?: readonly Split[];
  scenarios?: readonly string[];
  runs?: readonly string[];
  classes?: readonly ClassName[];
}

function matches<T>(list: readonly T[] | undefined, value: T): boolean {
  return !list || list.length === 0 || list.includes(value);
}

/**
 * The exported encoder-input sample: manifest + real rows (`x.f32`, normalised float32).
 *
 * Padding is not stored; `matrix()` / `mask()` rebuild the 64-row tensor the encoder reads
 * (real rows first, then padding). Raw units are derived as `z * std + mean` (train statistics).
 */
export class EncoderDataset {
  readonly windows: readonly WindowMeta[];
  readonly nFeatures: number;
  readonly seqLen: number;
  private readonly index = new Map<string, WindowMeta>();

  private constructor(
    readonly manifest: EncoderManifest,
    private readonly x: Float32Array,
  ) {
    this.windows = manifest.windows;
    this.nFeatures = manifest.features.length;
    this.seqLen = manifest.seq_len;
    const nRows = x.length / this.nFeatures;
    if (!Number.isInteger(nRows)) {
      throw new Error(`x.f32 length ${x.length} is not a multiple of ${this.nFeatures} features`);
    }
    if (manifest.norm.mean.length !== this.nFeatures || manifest.norm.std.length !== this.nFeatures) {
      throw new Error("norm.mean / norm.std must have one entry per feature");
    }
    for (const w of this.windows) {
      if (w.n < 1 || w.n > this.seqLen) throw new Error(`window ${w.id}: n = ${w.n} outside 1..${this.seqLen}`);
      if (w.row < 0 || w.row + w.n > nRows) throw new Error(`window ${w.id}: rows exceed x.f32 (${nRows} rows)`);
      if (this.index.has(w.id)) throw new Error(`duplicate window id ${w.id}`);
      this.index.set(w.id, w);
    }
  }

  static async load(baseUrl = "data/encoder/"): Promise<EncoderDataset> {
    const mres = await fetch(`${baseUrl}manifest.json`);
    if (!mres.ok) throw new Error(`manifest.json: HTTP ${mres.status}`);
    const manifest = (await mres.json()) as EncoderManifest;
    const xres = await fetch(`${baseUrl}x.f32`);
    if (!xres.ok) throw new Error(`x.f32: HTTP ${xres.status}`);
    return new EncoderDataset(manifest, new Float32Array(await xres.arrayBuffer()));
  }

  static fromParts(manifest: EncoderManifest, x: Float32Array): EncoderDataset {
    return new EncoderDataset(manifest, x);
  }

  byId(id: string): WindowMeta | undefined {
    return this.index.get(id);
  }

  /** Index of a feature by name; throws if the manifest does not list it. */
  featureIndex(name: string): number {
    const j = this.manifest.features.indexOf(name);
    if (j < 0) throw new Error(`unknown feature ${name}`);
    return j;
  }

  /** The window's n × 13 real rows, normalised (zero-copy view). */
  realRows(w: WindowMeta): Float32Array {
    const f = this.nFeatures;
    return this.x.subarray(w.row * f, (w.row + w.n) * f);
  }

  /** The window's real rows in raw units (`z * std + mean`). */
  rawRows(w: WindowMeta): Float32Array {
    const z = this.realRows(w);
    const out = new Float32Array(z.length);
    const { mean, std } = this.manifest.norm;
    const f = this.nFeatures;
    for (let i = 0; i < z.length; i++) out[i] = z[i] * std[i % f] + mean[i % f];
    return out;
  }

  /** The seq_len × 13 tensor; padding rows are 0 (normalised) or NaN (raw). */
  matrix(w: WindowMeta, raw: boolean): Float32Array {
    const out = new Float32Array(this.seqLen * this.nFeatures);
    out.set(raw ? this.rawRows(w) : this.realRows(w));
    if (raw) out.fill(Number.NaN, w.n * this.nFeatures);
    return out;
  }

  /** 1 for the n real rows, 0 for padding. */
  mask(w: WindowMeta): Uint8Array {
    const m = new Uint8Array(this.seqLen);
    m.fill(1, 0, w.n);
    return m;
  }

  filter(f: WindowFilter): WindowMeta[] {
    return this.windows.filter(
      (w) =>
        matches(f.splits, w.split) &&
        (!f.scenarios || f.scenarios.length === 0 ||
          f.scenarios.includes(w.scenario) || f.scenarios.includes(scenarioGroup(w.scenario))) &&
        matches(f.runs, w.run) &&
        matches(f.classes, w.label_name),
    );
  }

  /** Sorted distinct runs, optionally restricted to a split and a scenario (name or time group). */
  runs(split?: Split, scenario?: string): string[] {
    const ws = this.filter({ splits: split ? [split] : undefined, scenarios: scenario ? [scenario] : undefined });
    return [...new Set(ws.map((w) => w.run))].sort();
  }

  /** Sorted distinct scenario folder names. */
  scenarios(): string[] {
    return [...new Set(this.windows.map((w) => w.scenario))].sort();
  }
}
