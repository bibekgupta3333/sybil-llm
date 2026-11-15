"""Read-only checks of the window cache; writes `<cache_dir>/verify_report.json`.

    .venv-train/bin/python -m baselinepaper.experiments.verify_data

Checks (the letters are the report's keys):

a. the GridSybil windows of the test vehicles re-derived from T32_benign_grid equal `T32_benign/test_gridsybil`
   (same id set, identical masks, max abs difference within the tolerance);
b. no vehicle key is in two of train / val / test, across benign and GridSybil windows;
c. every train / val GridSybil (and benign) vehicle key is in the T32_benign metadata's train / val vehicle list;
d. windows per split (against the T32_benign metadata), composition of every regime's pools, share of 1-3-row
   windows per split;
e. padding rows are zero, masks are non-empty prefix masks, n_messages = mask row count, window ids are unique
   across splits, the missing `log_dtau` cells of the re-normalised GridSybil windows are 0.

Exit code 1 if any check fails.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np

from baselinepaper.experiments import config as cfg
from baselinepaper.experiments.data import LOG_DTAU, REDERIVED_TEST_GRIDSYBIL, EncoderInput, WindowCache, is_prefix_mask

PARTS = {
    "train": (cfg.TRAIN_BENIGN, cfg.TRAIN_GRIDSYBIL),
    "val": (cfg.VAL_BENIGN, cfg.VAL_GRIDSYBIL),
    "test": cfg.TEST_SPLITS,
}


class CacheVerifier:
    """Runs the checks on a built cache."""

    def __init__(self, config: cfg.ProtocolConfig | None = None) -> None:
        self.config = config or cfg.ProtocolConfig()
        self.cache = WindowCache(self.config)
        self.benign_input = EncoderInput(self.config.benign_input_dir)
        self.grid_input = EncoderInput(self.config.grid_input_dir)

    def keys(self, part: str) -> set[str]:
        return set().union(*(set(self.cache.split(s).vehicle_key.tolist()) for s in PARTS[part]))

    def check_rederivation(self) -> dict:
        return self.cache.rederivation_check()

    def check_disjoint_vehicles(self) -> dict:
        keys = {part: self.keys(part) for part in PARTS}
        overlaps = {
            f"{a}&{b}": len(keys[a] & keys[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        }
        return {
            "vehicle_keys": {part: len(k) for part, k in keys.items()},
            "keys_in_two_parts": overlaps,
            "passed": not any(overlaps.values()),
        }

    def check_membership(self) -> dict:
        out, passed = {}, True
        for part, splits in PARTS.items():
            allowed = self.benign_input.vehicle_ids(part)
            for split in splits:
                keys = set(self.cache.split(split).vehicle_key.tolist())
                outside = len(keys - allowed)
                out[split] = {"vehicle_keys": len(keys), "not_in_metadata_list": outside, "metadata_list": part}
                passed &= outside == 0
        rederived = set(self.cache.split(REDERIVED_TEST_GRIDSYBIL).vehicle_key.tolist())
        out[REDERIVED_TEST_GRIDSYBIL] = {
            "vehicle_keys": len(rederived),
            "not_in_metadata_list": len(rederived - self.benign_input.vehicle_ids("test")),
            "metadata_list": "test",
        }
        grid_scenarios = set(self.config.gridsybil_scenarios)
        folders_ok = all(
            set(np.unique(self.cache.split(s).scenario).tolist()) <= grid_scenarios
            for s in (cfg.TRAIN_GRIDSYBIL, cfg.VAL_GRIDSYBIL)
        )
        return {
            "splits": out,
            "gridsybil_windows_only_from_gridsybil_folders": folders_ok,
            "passed": passed and folders_ok,
        }

    def check_counts(self) -> dict:
        metadata_counts = self.benign_input.metadata["counts_by_split"]
        splits, passed = {}, True
        for name in (*cfg.ALL_SPLITS, REDERIVED_TEST_GRIDSYBIL):
            s = self.cache.split(name)
            record = {
                "windows": len(s),
                "messages": int(s.n_messages.sum()),
                "vehicle_keys": len(np.unique(s.vehicle_key)),
                "windows_1_3_rows": int((s.n_messages <= 3).sum()),
                "share_1_3_rows": float((s.n_messages <= 3).mean()),
                "padding_share": float(1 - s.n_messages.sum() / (len(s) * self.config.seq_len)),
                "windows_by_scenario": {k: int(v) for k, v in zip(*np.unique(s.scenario, return_counts=True))},
            }
            source = cfg.BENIGN_INPUT_SPLITS.get(name)
            if source is not None:
                expected = metadata_counts[source]
                record["matches_t32_benign_metadata"] = (
                    record["windows"] == expected["windows"] and record["messages"] == expected["messages"]
                )
                passed &= record["matches_t32_benign_metadata"]
            splits[name] = record
        grid_counts = self.grid_input.metadata["counts_by_split"]
        benign_total = sum(splits[s]["windows"] for s in (cfg.TRAIN_BENIGN, cfg.VAL_BENIGN, cfg.TEST_BENIGN))
        gridsybil_total = sum(
            splits[s]["windows"] for s in (cfg.TRAIN_GRIDSYBIL, cfg.VAL_GRIDSYBIL, REDERIVED_TEST_GRIDSYBIL)
        )
        # the grid input = the same benign windows (re-split) + every GridSybil window
        grid_total = sum(grid_counts[s]["windows"] for s in ("train", "val", "test"))
        all_used = gridsybil_total == grid_total - benign_total
        return {
            "splits": splits,
            "gridsybil_windows_train_val_test": gridsybil_total,
            "grid_input_windows_minus_benign_windows": grid_total - benign_total,
            "every_gridsybil_window_of_the_grid_input_assigned": all_used,
            "training_pools": {r: self.cache.regime_composition(r) for r in cfg.REGIMES},
            "validation_pools": {r: self.cache.regime_composition(r, "val") for r in cfg.REGIMES},
            "passed": bool(passed and all_used),
        }

    def check_arrays(self) -> dict:
        log_dtau = self.cache.meta["features"].index(LOG_DTAU)
        splits, passed, ids = {}, True, []
        for name in (*cfg.ALL_SPLITS, REDERIVED_TEST_GRIDSYBIL):
            s = self.cache.split(name)
            x, mask = np.asarray(s.x), np.asarray(s.mask)
            record = {
                "shape_ok": x.shape == (len(s), self.config.seq_len, self.config.n_features)
                and x.dtype == np.float32
                and mask.shape == (len(s), self.config.seq_len)
                and mask.dtype == bool,
                "nonzero_padding_values": int(np.count_nonzero(x[~mask])),
                "prefix_masks": is_prefix_mask(mask),
                "n_messages_equal_mask_rows": bool(np.array_equal(mask.sum(axis=1), s.n_messages)),
                "finite": bool(np.isfinite(x).all()),
                "nonzero_missing_log_dtau_cells": int(np.count_nonzero(x[s.link_window_index == 0, 0, log_dtau])),
                "unique_window_ids": len(np.unique(s.window_id)) == len(s),
            }
            record["passed"] = bool(
                record["shape_ok"]
                and record["nonzero_padding_values"] == 0
                and record["prefix_masks"]
                and record["n_messages_equal_mask_rows"]
                and record["finite"]
                and record["nonzero_missing_log_dtau_cells"] == 0
                and record["unique_window_ids"]
            )
            passed &= record["passed"]
            splits[name] = record
            if name != REDERIVED_TEST_GRIDSYBIL:
                ids.append(np.asarray(s.window_id))
        all_ids = np.concatenate(ids)
        unique_across = len(np.unique(all_ids)) == len(all_ids)
        return {
            "splits": splits,
            "window_ids_unique_across_splits": unique_across,
            "source_nonzero_padding_values": self.cache.meta["source_nonzero_padding_values"],
            "passed": bool(passed and unique_across),
        }

    def run(self) -> dict:
        report = {
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "cache_meta_sha256": self.cache.fingerprint(),
            "a_rederived_test_gridsybil": self.check_rederivation(),
            "b_disjoint_vehicles": self.check_disjoint_vehicles(),
            "c_vehicle_lists": self.check_membership(),
            "d_counts": self.check_counts(),
            "e_arrays": self.check_arrays(),
        }
        report["passed"] = all(v["passed"] for k, v in report.items() if isinstance(v, dict))
        (Path(self.config.cache_dir) / "verify_report.json").write_text(json.dumps(report, indent=1))
        return report


def main() -> None:
    report = CacheVerifier().run()
    a, d = report["a_rederived_test_gridsybil"], report["d_counts"]
    print(f"a. re-derived GridSybil test windows: {a}")
    print(f"b. vehicles: {report['b_disjoint_vehicles']}")
    print(f"c. vehicle lists: passed={report['c_vehicle_lists']['passed']}")
    for name, s in d["splits"].items():
        print(f"d. {name}: {s['windows']:,} windows, {s['vehicle_keys']} vehicles, 1-3 rows {s['share_1_3_rows']:.4f}")
    for regime, pool in d["training_pools"].items():
        print(f"d. training pool {regime}: {pool}")
    print(f"e. arrays: passed={report['e_arrays']['passed']}")
    print(f"ALL CHECKS PASSED: {report['passed']}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
