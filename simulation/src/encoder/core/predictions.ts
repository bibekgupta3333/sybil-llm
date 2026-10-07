import type { ModelCard, PredictionEntry, PredictionIndex, Split } from "./types";

/**
 * Model scores per window (`predictions/index.json` + one file per model).
 * A missing or empty index yields a store with no models (encoder untrained).
 */
export class PredictionStore {
  private constructor(
    private readonly cards: readonly ModelCard[],
    private readonly scores: ReadonlyMap<string, Readonly<Record<string, PredictionEntry>>>,
  ) {}

  static async load(baseUrl = "data/encoder/"): Promise<PredictionStore> {
    let index: PredictionIndex;
    try {
      const res = await fetch(`${baseUrl}predictions/index.json`);
      if (!res.ok) return PredictionStore.empty();
      index = (await res.json()) as PredictionIndex;
    } catch {
      return PredictionStore.empty();
    }
    const scores: Record<string, Record<string, PredictionEntry>> = {};
    for (const card of index.models ?? []) {
      const res = await fetch(`${baseUrl}predictions/${card.file}`);
      if (!res.ok) throw new Error(`predictions/${card.file}: HTTP ${res.status}`);
      scores[card.name] = (await res.json()) as Record<string, PredictionEntry>;
    }
    return PredictionStore.fromParts(index, scores);
  }

  static fromParts(index: PredictionIndex, scores: Readonly<Record<string, Record<string, PredictionEntry>>>): PredictionStore {
    const models = index.models ?? [];
    for (const m of models) {
      if (!(m.name in scores)) throw new Error(`no scores for model ${m.name}`);
    }
    return new PredictionStore(models, new Map(models.map((m) => [m.name, scores[m.name]])));
  }

  static empty(): PredictionStore {
    return new PredictionStore([], new Map());
  }

  models(): readonly ModelCard[] {
    return this.cards;
  }

  score(model: string, windowId: string): PredictionEntry | undefined {
    return this.scores.get(model)?.[windowId];
  }

  /** True if the model was trained on this split (its scores there are not held-out evidence). */
  trainedOn(model: string, split: Split): boolean {
    return this.cards.find((m) => m.name === model)?.trained_on_splits.includes(split) ?? false;
  }
}
