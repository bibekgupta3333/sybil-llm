import { afterEach, describe, expect, it, vi } from "vitest";

import { PredictionStore } from "./predictions";
import type { ModelCard } from "./types";

const card: ModelCard = {
  name: "timesnet_s0", file: "timesnet_s0.json", checkpoint_sha256: "abc", seed: 0,
  trained_on_splits: ["train", "pretrain_val"], threshold_from: "val", threshold: 0.5,
  score_meaning: "higher = more anomalous", created_utc: "2026-10-07T00:00:00Z",
};

describe("PredictionStore", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("is empty for an empty index", () => {
    const s = PredictionStore.fromParts({ models: [] }, {});
    expect(s.models()).toEqual([]);
    expect(s.score("x", "w000")).toBeUndefined();
    expect(s.trainedOn("x", "test")).toBe(false);
  });

  it("looks up scores and training splits", () => {
    const s = PredictionStore.fromParts({ models: [card] }, { timesnet_s0: { w000: { score: 0.9, pred: 1 } } });
    expect(s.score("timesnet_s0", "w000")).toEqual({ score: 0.9, pred: 1 });
    expect(s.score("timesnet_s0", "w001")).toBeUndefined();
    expect(s.trainedOn("timesnet_s0", "train")).toBe(true);
    expect(s.trainedOn("timesnet_s0", "test")).toBe(false);
    expect(() => PredictionStore.fromParts({ models: [card] }, {})).toThrow();
  });

  it("loads over fetch and degrades to empty when the index is missing", async () => {
    const files: Record<string, unknown> = {
      "base/predictions/index.json": { models: [card] },
      "base/predictions/timesnet_s0.json": { w000: { score: 0.1, pred: 0 } },
    };
    vi.stubGlobal("fetch", async (url: string) =>
      url in files ? new Response(JSON.stringify(files[url])) : new Response("", { status: 404 }));
    const s = await PredictionStore.load("base/");
    expect(s.score("timesnet_s0", "w000")?.pred).toBe(0);
    const empty = await PredictionStore.load("missing/");
    expect(empty.models()).toEqual([]);
  });
});
