"""Task 1.5 leakage audit — verifies the committed splits in data/prepared_data/.

Read-only against data/. Checks:
  1. idx_train / idx_val / idx_test are pairwise disjoint and index-complete.
  2. Zero sender_uid overlap between train/val, train/test, val/test
     (sender_uid = family_group_subfolder_senderPseudo — one physical vehicle
     run in one scenario file).
  3. Zero sender_uid overlap between idx_group_0709 and idx_group_1416.
  4. Raw senderPseudo overlap between 0709 and 1416 (informational — pseudonym
     integers can be reused by SUMO/VeReMi across separate scenario runs; this
     does not by itself imply sender_uid leakage, since group is baked into
     sender_uid, but it is worth knowing about).
  5. Recomputed train-only normalization mean/std vs. the committed
     norm_mean.npy / norm_std.npy.

Exits 1 if any PASS/FAIL check fails.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PREPARED = Path(__file__).resolve().parent.parent / "data" / "prepared_data"

failures = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def main():
    idx_train = np.load(PREPARED / "idx_train.npy")
    idx_val = np.load(PREPARED / "idx_val.npy")
    idx_test = np.load(PREPARED / "idx_test.npy")
    idx_0709 = np.load(PREPARED / "idx_group_0709.npy")
    idx_1416 = np.load(PREPARED / "idx_group_1416.npy")
    meta = pd.read_parquet(PREPARED / "window_metadata.parquet")
    norm_mean = np.load(PREPARED / "norm_mean.npy")
    norm_std = np.load(PREPARED / "norm_std.npy")
    n_windows = len(meta)

    print("=" * 78)
    print("1. TRAIN/VAL/TEST — index-level partition")
    print("=" * 78)
    s_train, s_val, s_test = set(idx_train), set(idx_val), set(idx_test)
    check("train/val index-disjoint", s_train.isdisjoint(s_val))
    check("train/test index-disjoint", s_train.isdisjoint(s_test))
    check("val/test index-disjoint", s_val.isdisjoint(s_test))
    union = s_train | s_val | s_test
    check(
        "train+val+test covers all windows",
        len(union) == n_windows,
        f"union={len(union):,} vs n_windows={n_windows:,}",
    )
    print(f"    sizes: train={len(idx_train):,} val={len(idx_val):,} test={len(idx_test):,}")

    print()
    print("=" * 78)
    print("2. TRAIN/VAL/TEST — sender_uid leakage (the core Task 1.5 check)")
    print("=" * 78)
    su_train = set(meta.iloc[idx_train]["sender_uid"])
    su_val = set(meta.iloc[idx_val]["sender_uid"])
    su_test = set(meta.iloc[idx_test]["sender_uid"])
    ov_tv = su_train & su_val
    ov_tt = su_train & su_test
    ov_vt = su_val & su_test
    check("train/val sender_uid overlap", len(ov_tv) == 0, f"{len(ov_tv)} shared senders")
    check("train/test sender_uid overlap", len(ov_tt) == 0, f"{len(ov_tt)} shared senders")
    check("val/test sender_uid overlap", len(ov_vt) == 0, f"{len(ov_vt)} shared senders")
    print(
        f"    unique senders: train={len(su_train):,} val={len(su_val):,} test={len(su_test):,}"
    )

    print()
    print("=" * 78)
    print("3. GROUP 0709 / 1416 — scenario-holdout partition")
    print("=" * 78)
    s0709, s1416 = set(idx_0709), set(idx_1416)
    check("group_0709/1416 index-disjoint", s0709.isdisjoint(s1416))
    check(
        "group_0709+group_1416 covers all windows",
        len(s0709 | s1416) == n_windows,
        f"union={len(s0709 | s1416):,} vs n_windows={n_windows:,}",
    )
    su_0709 = set(meta.iloc[idx_0709]["sender_uid"])
    su_1416 = set(meta.iloc[idx_1416]["sender_uid"])
    ov_group_su = su_0709 & su_1416
    check(
        "group_0709/1416 sender_uid overlap",
        len(ov_group_su) == 0,
        f"{len(ov_group_su)} shared sender_uid (expected 0 — group is baked into sender_uid)",
    )

    print()
    print("=" * 78)
    print("4. GROUP 0709 / 1416 — raw senderPseudo overlap (informational)")
    print("=" * 78)
    pseudo_0709 = set(meta.iloc[idx_0709]["senderPseudo"])
    pseudo_1416 = set(meta.iloc[idx_1416]["senderPseudo"])
    ov_pseudo = pseudo_0709 & pseudo_1416
    pct = 100 * len(ov_pseudo) / max(len(pseudo_0709 | pseudo_1416), 1)
    print(
        f"[INFO] raw senderPseudo overlap: {len(ov_pseudo):,} of "
        f"{len(pseudo_0709 | pseudo_1416):,} distinct ids ({pct:.1f}%) appear in both groups — "
        "not a leakage failure by construction (sender_uid disambiguates by group+subfolder), "
        "but confirms senderPseudo integers are recycled by SUMO across separate scenario runs, "
        "so senderPseudo alone must never be used as the grouping key for any future split."
    )

    print()
    print("=" * 78)
    print("5. NORMALIZATION — recomputed from train split only")
    print("=" * 78)
    X = np.load(PREPARED / "X_windows.npy")
    X_train_flat = X[idx_train].reshape(-1, X.shape[-1])
    recomputed_mean = X_train_flat.mean(axis=0)
    recomputed_std = X_train_flat.std(axis=0)
    recomputed_std_safe = np.where(recomputed_std < 1e-8, 1.0, recomputed_std)

    mean_close = np.allclose(recomputed_mean, norm_mean, atol=1e-6)
    std_close = np.allclose(recomputed_std_safe, norm_std, atol=1e-6)
    check("norm_mean.npy matches train-only recomputation", mean_close)
    check("norm_std.npy matches train-only recomputation", std_close)
    if not (mean_close and std_close):
        max_mean_diff = np.abs(recomputed_mean - norm_mean).max()
        max_std_diff = np.abs(recomputed_std_safe - norm_std).max()
        print(f"    max |mean diff| = {max_mean_diff:.6g}, max |std diff| = {max_std_diff:.6g}")

    print()
    print("=" * 78)
    if failures:
        print(f"RESULT: {len(failures)} CHECK(S) FAILED: {', '.join(failures)}")
        print("=" * 78)
        sys.exit(1)
    else:
        print("RESULT: ALL CHECKS PASSED")
        print("=" * 78)


if __name__ == "__main__":
    main()
