"""Confusion counts and per-check fire counts of the plausibility baseline, additive across receivers and workers.

Implements tracker 4.5. The paper's partition (VeReMi-Extension, Table II): detected genuine = FP, detected
misbehaving = TP, not detected genuine = TN, not detected misbehaving = FN; Accuracy, Precision, Recall, F1 as usual.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection

import numpy as np


@dataclasses.dataclass
class DetectionCounts:
    """Additive counts for one subset.

    Attributes:
        tp, fp, tn, fn: confusion counts over labelled messages.
        unknown: messages whose sender has no trace file in the run (excluded from tp..fn).
        check_fires: check name -> [fires on genuine, fires on misbehaving].
        check_evaluated: check name -> [evaluated on genuine, evaluated on misbehaving] (a check that does not
            apply to a message, e.g. a consistency check on the first message of a pseudonym, is not evaluated).
        only_check_fires: check name -> [genuine, misbehaving] messages where this was the only decision check that fired.
        receivers: number of receiver files.
    """

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    unknown: int = 0
    check_fires: dict[str, list[int]] = dataclasses.field(default_factory=dict)
    check_evaluated: dict[str, list[int]] = dataclasses.field(default_factory=dict)
    only_check_fires: dict[str, list[int]] = dataclasses.field(default_factory=dict)
    receivers: int = 0

    def add_messages(
        self,
        label: np.ndarray,
        flagged: np.ndarray,
        fired: dict[str, np.ndarray],
        evaluated: dict[str, np.ndarray],
        decision: Collection[str] | None = None,
    ) -> None:
        """Adds one receiver's messages.

        Args:
            label: (n,) int, 1 misbehaving, 0 genuine, negative unknown.
            flagged: (n,) bool, the detector's decision.
            fired: check name -> (n,) bool, the check fired (output below the threshold).
            evaluated: check name -> (n,) bool, the check applied to the message.
        """
        known = label >= 0
        mal = known & (label == 1)
        gen = known & (label == 0)
        self.tp += int(np.sum(flagged & mal))
        self.fn += int(np.sum(~flagged & mal))
        self.fp += int(np.sum(flagged & gen))
        self.tn += int(np.sum(~flagged & gen))
        self.unknown += int(np.sum(~known))
        decision = set(fired) if decision is None else set(decision)
        n_fired = np.zeros(label.shape, dtype=np.int64)
        for name, fire in fired.items():
            if name in decision:
                n_fired += fire
        for name, fire in fired.items():
            only = fire & (n_fired == 1) & (name in decision)
            for store, values in (
                (self.check_fires, fire),
                (self.check_evaluated, evaluated[name]),
                (self.only_check_fires, only),
            ):
                pair = store.setdefault(name, [0, 0])
                pair[0] += int(np.sum(values & gen))
                pair[1] += int(np.sum(values & mal))

    def merge(self, other: DetectionCounts) -> None:
        """Adds another counts object into this one."""
        for name in ("tp", "fp", "tn", "fn", "unknown", "receivers"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for mine, theirs in (
            (self.check_fires, other.check_fires),
            (self.check_evaluated, other.check_evaluated),
            (self.only_check_fires, other.only_check_fires),
        ):
            for key, pair in theirs.items():
                acc = mine.setdefault(key, [0, 0])
                acc[0] += pair[0]
                acc[1] += pair[1]

    def scores(self) -> dict[str, float]:
        """Accuracy, precision, recall, F1 (NaN when a denominator is 0)."""
        return confusion_scores(self.tp, self.fp, self.tn, self.fn)

    def check_rates(self) -> dict[str, dict[str, float]]:
        """Per check: fire rate over all genuine / misbehaving messages and over those where it was evaluated."""
        n_gen, n_mal = self.fp + self.tn, self.tp + self.fn
        out: dict[str, dict[str, float]] = {}
        for name, (fire_gen, fire_mal) in self.check_fires.items():
            eval_gen, eval_mal = self.check_evaluated[name]
            only_gen, only_mal = self.only_check_fires[name]
            out[name] = {
                "fire_rate_genuine": _ratio(fire_gen, n_gen),
                "fire_rate_malicious": _ratio(fire_mal, n_mal),
                "fire_rate_genuine_when_evaluated": _ratio(fire_gen, eval_gen),
                "fire_rate_malicious_when_evaluated": _ratio(fire_mal, eval_mal),
                "evaluated_share_genuine": _ratio(eval_gen, n_gen),
                "evaluated_share_malicious": _ratio(eval_mal, n_mal),
                "only_check_fired_genuine": only_gen,
                "only_check_fired_malicious": only_mal,
            }
        return out

    def to_dict(self) -> dict:
        """JSON-ready counts, scores and per-check rates."""
        return {
            "counts": {"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn, "unknown_label": self.unknown},
            "messages": self.tp + self.fp + self.tn + self.fn,
            "genuine": self.fp + self.tn,
            "malicious": self.tp + self.fn,
            "receivers": self.receivers,
            "scores": self.scores(),
            "checks": self.check_rates(),
            "check_counts": {
                name: {
                    "fires": self.check_fires[name],
                    "evaluated": self.check_evaluated[name],
                    "only_fired": self.only_check_fires[name],
                }
                for name in self.check_fires
            },
        }


def confusion_scores(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    """Accuracy, precision, recall and F1 from confusion counts (NaN for a zero denominator)."""
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else float("nan")
    return {"accuracy": _ratio(tp + tn, tp + fp + tn + fn), "precision": precision, "recall": recall, "f1": f1}


def _ratio(num: int, den: int) -> float:
    """num / den, NaN when den is 0."""
    return num / den if den else float("nan")
