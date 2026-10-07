import type { EncoderDataset } from "../core/encoder_dataset";
import type { PredictionStore } from "../core/predictions";
import { SPLITS, type WindowMeta } from "../core/types";
import { fmtDistance } from "../render/spatial_map";
import { fmtInt, fmtPct, h } from "./dom";
import { EVAL_ONLY_BADGE, SPLIT_LABEL, classChip, isSharedGhostPseudonym } from "./labels";

/** Panel shell matching the legacy simulator's `.panel` / `.panel-head`. */
export function panel(title: string, caption: string | Node | null, ...body: (Node | null)[]): HTMLElement {
  return h("section", { class: "panel enc-panel" },
    h("div", { class: "panel-head" }, h("span", { class: "panel-title" }, title),
      caption === null ? null : typeof caption === "string" ? h("span", { class: "panel-caption" }, caption) : caption),
    ...body);
}

/** Always-visible caveats: scope of the data, F13, untrained model, sample vs full data. */
export function caveatsBanner(ds: EncoderDataset): HTMLElement {
  const fd = ds.manifest.full_dataset;
  return h("div", { class: "enc-banner", role: "note" },
    h("b", {}, "Caveats"),
    h("span", {}, "Benign + GridSybil only (no DataReplay / DoS yet)."),
    h("span", {}, "GridSybil ghosts share pseudonym 1 in every run (F13)."),
    h("span", {}, "Encoder untrained: no model scores."),
    h("span", {}, `Sample of ${fmtInt(ds.manifest.sample.n_windows)} windows, not the full ${fmtInt(fd.windows)}.`),
  );
}

/** Mask + length: what the mask hides and why length must not be a shortcut. */
export function maskPanel(ds: EncoderDataset, w: WindowMeta): HTMLElement {
  const T = ds.seqLen;
  const bar = h("div", { class: "enc-maskbar", title: `${w.n} real rows, ${T - w.n} padding rows` });
  for (let t = 0; t < T; t++) bar.append(h("span", { class: t < w.n ? "real" : "pad" }));
  const fd = ds.manifest.full_dataset;
  return panel("Mask & length", `${w.n} / ${T} real`,
    h("div", { class: "enc-body" },
      h("div", { class: "enc-big" }, h("b", {}, String(w.n)), ` real rows · ${T - w.n} padding · mask = ${w.n} ones then ${T - w.n} zeros`),
      bar,
      h("p", {}, `The encoder averages only the ${w.n} real row${w.n === 1 ? "" : "s"} (masked mean pooling); losses ignore padding rows.`),
      h("p", {},
        "Length shortcut: window length alone separates the classes with AUC ",
        h("b", {}, fd.length_only_auc_val.toFixed(3)),
        " on val (chance 0.5). An unmasked mean over all 64 rows would leak length into every feature.",
      ),
      h("p", { class: "enc-src" }, `source: ${fd.length_only_auc_source} · full-data padding share ${fmtPct(fd.padding_share)}`),
    ));
}

/** Provenance: every id back to the raw VeReMi trace + split integrity. */
export function provenancePanel(ds: EncoderDataset, w: WindowMeta, labels: boolean): HTMLElement {
  const fd = ds.manifest.full_dataset;
  const splitsOk = w.sender_splits_full.length === 1;
  const kv = (k: string, v: Node | string): HTMLElement[] => [h("dt", {}, k), h("dd", {}, v)];
  const ghost = isSharedGhostPseudonym(w.scenario, w.sender_pseudo);
  return panel("Provenance", `window ${w.i} of the sample`,
    h("div", { class: "enc-body" },
      h("dl", { class: "enc-kv" },
        ...kv("window id", h("code", {}, w.id)),
        ...kv("split", SPLIT_LABEL[w.split]),
        ...kv("scenario · run", `${w.scenario} · ${w.run}`),
        ...kv("receiver file", h("code", {}, w.receiver_file)),
        ...kv("receiver vehicle", String(w.receiver)),
        ...kv("sender vehicle", String(w.sender)),
        ...kv("sender pseudonym", String(w.sender_pseudo)),
        ...kv("link crop", `${w.k + 1} of ${w.K} (crops of 64 messages)`),
        ...kv("raw trace", h("code", {}, w.raw_trace)),
        ...kv("label", labels ? h("span", { class: "enc-inline" }, classChip(w.label_name), ` code ${w.label}`, h("span", { class: "enc-eval-tag" }, "evaluation only")) : "hidden"),
      ),
      h("div", { class: `enc-integrity ${splitsOk ? "ok" : "bad"}` },
        splitsOk
          ? `✓ sender in 1 split over the full dataset (${w.sender_splits_full.map((s) => SPLIT_LABEL[s]).join(", ")})`
          : `✗ sender in ${w.sender_splits_full.length} splits over the full dataset: ${w.sender_splits_full.join(", ")}`),
      h("div", { class: `enc-integrity ${fd.senders_in_multiple_splits === 0 ? "ok" : "bad"}` },
        `senders in more than one split = ${fd.senders_in_multiple_splits} of ${fmtInt(fd.sender_vehicles)} (full dataset)`),
      ghost
        ? h("p", { class: "enc-note" },
          "Pseudonym 1 is shared by every GridSybil ghost in this run (F13). The sender vehicle shown is simulator ground truth; grouping by it is privileged information the receiver does not have.")
        : null,
    ));
}

/** Predictions slot: honest about the untrained encoder; warns about scores on training splits. */
export function detectionPanel(preds: PredictionStore, w: WindowMeta, labels: boolean): HTMLElement {
  const models = preds.models();
  if (models.length === 0) {
    return panel("Detection", "no model yet",
      h("div", { class: "enc-body" },
        h("div", { class: "verdict neutral" }, "No model scores yet (encoder untrained)"),
        h("p", {}, "When a model exists, ", h("code", {}, "predictions/index.json"),
          " lists it with its checkpoint hash, seed, training splits and a threshold chosen on val; each window then shows its anomaly score (higher = more anomalous), the threshold and the decision."),
        h("p", {}, "Scores on a split the model trained on are flagged — they are not held-out evidence."),
      ));
  }
  const rows = models.map((m) => {
    const s = preds.score(m.name, w.id);
    const trained = preds.trainedOn(m.name, w.split);
    return h("div", { class: "enc-model" },
      h("div", { class: "enc-model-head" }, h("b", {}, m.name), h("span", { class: "enc-src" }, `seed ${m.seed} · ${m.checkpoint_sha256.slice(0, 10)}`)),
      s
        ? h("div", {}, `score ${s.score.toFixed(4)} · threshold ${m.threshold.toFixed(4)} (from ${m.threshold_from}) → `,
          h("b", {}, s.pred === null ? "no decision" : s.pred === 1 ? "anomalous" : "normal"),
          labels ? h("span", { class: "enc-src" }, ` · label ${w.label_name}`) : null)
        : h("div", { class: "enc-src" }, "no score for this window"),
      trained ? h("div", { class: "enc-integrity bad" }, `⚠ trained on ${SPLIT_LABEL[w.split]}: not held-out evidence`) : null,
      h("div", { class: "enc-src" }, m.score_meaning),
    );
  });
  return panel("Detection", `${models.length} model${models.length > 1 ? "s" : ""}`, h("div", { class: "enc-body" }, ...rows));
}

/** Full-dataset per-split table from `manifest.full_dataset` (not the sample) + the sample's own counts. */
export function fullDatasetPanel(ds: EncoderDataset, labels: boolean): HTMLElement {
  const fd = ds.manifest.full_dataset;
  const sample = ds.manifest.sample;
  const head = h("tr", {}, h("th", {}, "split"), h("th", {}, "windows"),
    labels ? h("th", {}, "benign") : null, labels ? h("th", {}, "GridSybil") : null,
    h("th", {}, "messages"), h("th", {}, "padding"), h("th", {}, "in sample"));
  const body = SPLITS.map((s) => {
    const p = fd.per_split[s];
    const sc = sample.per_split_class[s] ?? {};
    const inSample = (sc.Benign ?? 0) + (sc.GridSybil ?? 0);
    return h("tr", {},
      h("td", {}, SPLIT_LABEL[s]),
      h("td", {}, p ? fmtInt(p.windows) : "—"),
      labels ? h("td", {}, p ? fmtInt(p.benign) : "—") : null,
      labels ? h("td", {}, p ? fmtInt(p.gridsybil) : "—") : null,
      h("td", {}, p ? fmtInt(p.messages) : "—"),
      h("td", {}, p ? fmtPct(p.padding_share) : "—"),
      h("td", { class: "enc-dim" }, labels ? `${fmtInt(inSample)} (${sc.Benign ?? 0} B / ${sc.GridSybil ?? 0} G)` : fmtInt(inSample)),
    );
  });
  const total = h("tr", { class: "enc-total" },
    h("td", {}, "all"), h("td", {}, fmtInt(fd.windows)),
    labels ? h("td", {}, fmtInt(SPLITS.reduce((a, s) => a + (fd.per_split[s]?.benign ?? 0), 0))) : null,
    labels ? h("td", {}, fmtInt(SPLITS.reduce((a, s) => a + (fd.per_split[s]?.gridsybil ?? 0), 0))) : null,
    h("td", {}, fmtInt(fd.messages)), h("td", {}, fmtPct(fd.padding_share)),
    h("td", { class: "enc-dim" }, fmtInt(sample.n_windows)));
  return panel("Full encoder input · per split", h("span", { class: "enc-badge enc-badge-full" }, "full data, not the sample"),
    h("div", { class: "enc-body" },
      h("table", { class: "enc-table" }, h("thead", {}, head), h("tbody", {}, ...body, total)),
      h("p", { class: "enc-src" },
        `${fmtInt(fd.sender_vehicles)} sender vehicles · senders in more than one split: ${fd.senders_in_multiple_splits} · source ${ds.manifest.source} · export ${ds.manifest.created_utc} @ ${ds.manifest.git_commit.slice(0, 10)} · seed ${ds.manifest.seed}`),
      h("p", { class: "enc-src" }, `Sample rule: ${sample.rule}`),
      labels ? h("p", { class: "enc-src" }, `Class columns: ${EVAL_ONLY_BADGE}.`) : null,
    ));
}

/** Range summary line for the map panel caption. */
export function rangeCaption(range: readonly number[]): string {
  if (range.length === 0) return "";
  let max = 0;
  let sum = 0;
  for (const r of range) {
    if (r > max) max = r;
    sum += r;
  }
  return `max range ${fmtDistance(max)} · mean ${fmtDistance(sum / range.length)}`;
}
