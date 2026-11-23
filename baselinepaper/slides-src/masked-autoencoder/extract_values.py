"""Extract every number of the masked-autoencoder deck into one JSON with sources.

Run from the repo root:  .venv-train/bin/python <this file> <out.json>
Reads only run files (config / env / summary / history / test_metrics / scores) of task/masked_autoencoder/runs, the
TimesNet T32 evaluation summaries + score files, and baselinepaper/01-our-studies/results.json. Every AUROC recomputed
from the score files is asserted equal to test_metrics.json (|diff| <= 1e-6). T64 labels: through the package's own
source object (task.masked_autoencoder.data.make_source), exactly as notebook cell 33 does; the read is recorded.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, ".")
from task.masked_autoencoder.config import MAEConfig  # noqa: E402
from task.masked_autoencoder.data import make_source  # noqa: E402
from task.masked_autoencoder.evaluate import load_scores  # noqa: E402

OUT = Path(sys.argv[1])
R = Path("task/masked_autoencoder/runs")
TN = Path("src/runs/evaluation/all/T32_benign")
W, SEEDS, SC = (64, 48, 32), (0, 1, 2, 3), ("masked_recon", "full_recon", "mahalanobis")
ATT = ("GridSybil", "DataReplay", "DoSRandom", "DoSDisruptive")
FPR_GRID = np.r_[np.linspace(0, 0.1, 41), np.linspace(0.1, 1, 91)[1:]]
CHECKED = []


ENCS = ("mlp", "resnet")


def rd(t, s, enc="mlp"):
    return R / enc / f"T{t}-seed{s}"


def lm_auroc(y, v, n):
    """Notebook cell 43 `length_matched_auroc`, same logic: attack-weighted AUROC inside each exact window length."""
    aucs, w = [], []
    for k in np.unique(n[y == 1]):
        sel = n == k
        if 0 < y[sel].sum() < sel.sum():
            aucs.append(roc_auc_score(y[sel], v[sel]))
            w.append(y[sel].sum())
    return float(np.average(aucs, weights=w)), float(np.sum(w) / (y == 1).sum())


def roc_pts(y, v):
    fpr, tpr, _ = roc_curve(y, v)
    return [round(float(x), 4) for x in np.interp(FPR_GRID, fpr, tpr)]


def ms(vals):
    a = np.array(vals, float)
    return {
        "values": [float(x) for x in vals],
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


src64 = make_source(MAEConfig.from_dict(json.loads((rd(64, 0) / "config.json").read_text())))
codes64 = src64.labels("test", "masked-AE deck: ROC / length-matched AUROC from saved T64 scores")
ids64 = src64.load("test").ids
assert set(np.unique(codes64)) <= {0, 16}


def roc_data(t, s, a, enc):
    f = rd(t, s, enc) / "scores"
    if t == 64:
        te = load_scores(f, "test")
        assert np.array_equal(te.ids, ids64)
        return (codes64 != 0).astype(int), te.scores, te.n_messages
    b, x = load_scores(f, "test"), load_scores(f, f"test_{a.lower()}")
    y = np.r_[np.zeros(len(b), int), np.ones(len(x), int)]
    return y, {k: np.r_[b.scores[k], x.scores[k]] for k in SC}, np.r_[b.n_messages, x.n_messages]


def extract(enc):
    per = {}
    for t in W:
        for s in SEEDS:
            d = rd(t, s, enc)
            tm = json.loads((d / "test_metrics.json").read_text())
            sm = json.loads((d / "summary.json").read_text())
            cfg = json.loads((d / "config.json").read_text())
            env = json.loads((d / "env.json").read_text())
            hist = [json.loads(l) for l in (d / "history.jsonl").read_text().splitlines() if l.strip()]
            assert cfg["window"] == t and cfg["seed"] == s and cfg["max_windows"] == 0 and cfg["encoder"] == enc
            e = {
                "params": sm["params"],
                "latent_dim": sm["latent_dim"],
                "best_epoch": sm["best_epoch"],
                "epochs_run": sm["epochs_run"],
                "stop_reason": sm["stop_reason"],
                "best_val_loss": sm["best_val_loss"],
                "lr_halvings_after": [c["after_epoch"] for c in sm["lr_changes"]],
                "train_seconds": sm["train_seconds"],
                "wall_seconds": sm["wall_seconds"],
                "device": sm["device"],
                "env": {k: env[k] for k in ("python", "torch", "platform", "device", "git_commit", "git_dirty")},
                "config": {
                    k: cfg[k]
                    for k in (
                        "encoder",
                        "blocks_dims",
                        "blocks_dropout",
                        "masked_prob",
                        "mlp_hidden_dims",
                        "mlp_dropout",
                        "feed_masked_input",
                        "linear_mlp_output",
                        "n_eval_draws",
                        "eval_mask_seed",
                        "lr",
                        "weight_decay",
                        "batch_size",
                        "max_epochs",
                        "patience",
                        "plateau_factor",
                        "plateau_patience",
                        "min_delta",
                        "grad_clip",
                    )
                },
                "counts": {
                    "train_windows": sm["train_windows"],
                    "val_windows": sm["val_windows"],
                    "train_vehicles": sm["train_vehicles"],
                    "val_vehicles": sm["val_vehicles"],
                    "test": tm["windows"],
                    "test_vehicles": tm["label_checks"]["test_vehicles"],
                    "test_vehicles_in_train_or_val": tm["label_checks"]["test_vehicles_in_train_or_val"],
                },
                "label_reads": sm["label_reads"],
                "z_dead_units_train": sm["z_dead_units_train"],
                "mahalanobis_cond": sm["mahalanobis"]["cond"],
                "history": [
                    {k: h[k] for k in ("epoch", "train_loss", "val_masked_mse", "val_full_mse", "lr")} for h in hist
                ],
                "attacks": {},
            }
            for a, m in tm["attacks"].items():
                y, sc, n = roc_data(t, s, a, enc)
                L = m["length_only"]["auroc"]
                assert abs(roc_auc_score(y, -n.astype(float)) - L) <= 1e-6
                CHECKED.append((enc, t, s, a, "length_only"))
                row = {"length_only": L, "share_rows_1_3": m["share_rows_1_3"]}
                for k in SC:
                    v = m["window"][k]
                    rec = roc_auc_score(y, sc[k])
                    assert abs(rec - v["auroc"]) <= 1e-6, (t, s, a, k, rec, v["auroc"])
                    CHECKED.append((enc, t, s, a, k))
                    lmv, cov = lm_auroc(y, sc[k], n)
                    row[k] = {
                        "auroc": v["auroc"],
                        "recomputed": float(rec),
                        "length_matched": lmv,
                        "covered": cov,
                        "recall_p95": v["recall_p95"],
                        "benign_fpr_p95": v["benign_fpr_p95"],
                        "auroc_rows_1_3": v["rows_1_3"]["auroc"],
                        "n_attack_rows_1_3": v["rows_1_3"]["n_attack"],
                        "auroc_rows_ge_4": v["rows_ge_4"]["auroc"],
                        "n_attack_rows_ge_4": v["rows_ge_4"]["n_attack"],
                        "n_benign_rows_ge_4": v["rows_ge_4"]["n_benign"],
                    }
                if a == "GridSybil" or (t == 32 and s == 0):
                    row["roc"] = {k: roc_pts(y, sc[k]) for k in SC}
                    row["roc"]["length_only"] = roc_pts(y, -n.astype(float))
                if a == "GridSybil":
                    ben = y == 0
                    short = (n[ben] <= 3).astype(int)
                    row["benign_short_vs_long"] = {k: float(roc_auc_score(short, sc[k][ben])) for k in SC}
                    if t == 32 and s == 0:
                        nb = n[ben]
                        row["benign_by_rows"] = {
                            int(r): {
                                "n": int((nb == r).sum()),
                                **{k: float(np.median(sc[k][ben][nb == r])) for k in SC},
                            }
                            for r in np.unique(nb)
                        }
                if t == 32:
                    row["one_row_share"] = {
                        "attack": float(np.mean(n[y == 1] == 1)),
                        "benign": float(np.mean(n[y == 0] == 1)),
                    }
                e["attacks"][a] = row
            h_best = [h for h in hist if h["epoch"] == sm["best_epoch"]][0]
            e["train_over_val_at_best"] = h_best["train_loss"] / sm["best_val_loss"]
            e["val_full_over_masked_at_best"] = h_best["val_full_mse"] / sm["best_val_loss"]
            v1 = hist[0]["val_masked_mse"]
            v10 = [h for h in hist if h["epoch"] == 10][0]["val_masked_mse"]
            e["share_of_val_decrease_by_epoch10"] = (v1 - v10) / (v1 - sm["best_val_loss"])
            per[f"T{t}-seed{s}"] = e

    summ = {}
    for t in W:
        atts = ATT if t == 32 else ("GridSybil",)
        for a in atts:
            g = [per[f"T{t}-seed{s}"]["attacks"][a] for s in SEEDS]
            x = {k: ms([r[k]["auroc"] for r in g]) for k in SC}
            x.update({f"{k}_length_matched": ms([r[k]["length_matched"] for r in g]) for k in SC})
            x.update({f"{k}_rows_ge_4": ms([r[k]["auroc_rows_ge_4"] for r in g]) for k in SC})
            x.update({f"{k}_rows_1_3": ms([r[k]["auroc_rows_1_3"] for r in g]) for k in SC})
            x["n_attack_rows_ge_4"] = g[0]["masked_recon"]["n_attack_rows_ge_4"]
            x["length_only"] = g[0]["length_only"]
            assert all(r["length_only"] == x["length_only"] for r in g)
            x["share_rows_1_3"] = g[0]["share_rows_1_3"]
            x["recall_p95_masked"] = ms([r["masked_recon"]["recall_p95"] for r in g])
            x["fpr_p95_masked"] = ms([r["masked_recon"]["benign_fpr_p95"] for r in g])
            if a == "GridSybil":
                x["benign_short_vs_long"] = {k: ms([r["benign_short_vs_long"][k] for r in g]) for k in SC}
            if t == 32:
                x["one_row_share_seed0"] = g[0]["one_row_share"]
            summ[f"T{t} {a}"] = x
        runs = [per[f"T{t}-seed{s}"] for s in SEEDS]
        assert all(r["counts"] == runs[0]["counts"] for r in runs)
        summ[f"T{t} training"] = {
            "params": runs[0]["params"],
            "counts": runs[0]["counts"],
            "best_epochs": [r["best_epoch"] for r in runs],
            "epochs_run": [r["epochs_run"] for r in runs],
            "lr_halvings": [len(r["lr_halvings_after"]) for r in runs],
            "best_val_masked": ms([r["best_val_loss"] for r in runs]),
            "train_over_val_at_best": ms([r["train_over_val_at_best"] for r in runs]),
            "val_full_over_masked_at_best": ms([r["val_full_over_masked_at_best"] for r in runs]),
            "share_of_val_decrease_by_epoch10": ms([r["share_of_val_decrease_by_epoch10"] for r in runs]),
            "train_minutes": [round(r["train_seconds"] / 60, 1) for r in runs],
            "wall_minutes": [round(r["wall_seconds"] / 60, 1) for r in runs],
            "devices": sorted({r["device"] for r in runs}),
            "z_dead_units": [r["z_dead_units_train"] for r in runs],
            "mahalanobis_cond": [r["mahalanobis_cond"] for r in runs],
        }
    return per, summ


per, summ = extract("mlp")
per_res, summ_res = extract("resnet")

conv = json.loads(Path("baselinepaper/01-our-studies/results.json").read_text())
e0, b48, b32 = (
    conv["task/encoder"]["seed_summary"]["seeds_0_3"],
    conv["task/autoencoder_benign_t48"]["seed_summary"],
    conv["task/autoencoder_benign_t32"]["seed_summary"],
)
convae = {
    "T64 GridSybil": {"recon": ms(e0["auroc_recon"]["values"]), "mahalanobis": ms(e0["auroc_maha"]["values"])},
    "T48 GridSybil": {"recon": ms(b48["auroc_recon"]["values"]), "mahalanobis": ms(b48["auroc_maha"]["values"])},
    "T32 GridSybil": {
        "recon": ms(b32["GridSybil_recon_auroc"]["values"]),
        "mahalanobis": ms(b32["GridSybil_maha_auroc"]["values"]),
    },
    "T32 DataReplay": {"recon": ms(b32["DataReplay_recon_auroc"]["values"])},
    "T32 DoSRandom": {"recon": ms(b32["DoSRandom_recon_auroc"]["values"])},
    "T32 DoSDisruptive": {"recon": ms(b32["DoSDisruptive_recon_auroc"]["values"])},
    "params": {"T64": 80189, "T48": 71741, "T32": 63293},
}

tn = {}
for name in ("model-t32b-seed0", "model-t32b-seed0-random-init"):
    m = json.loads((TN / name / "summary.json").read_text())["metrics"]
    ben = np.load(TN / name / "scores" / "test.npz")
    out = {}
    for a in ATT:
        sp = f"test_{a.lower()}"
        att = np.load(TN / name / "scores" / f"{sp}.npz")
        y = np.r_[np.zeros(len(ben["ids"]), int), np.ones(len(att["ids"]), int)]
        n = np.r_[ben["n_real_rows"], att["n_real_rows"]]
        row = {"length_only": m[sp]["window"]["length_only"]["all"]["auroc"]}
        for k in ("knn_k5", "mahalanobis", "recon"):
            v = np.r_[ben[k], att[k]]
            ok = np.isfinite(v)
            ref = m[sp]["window"][k]["all"]["auroc"]
            if ok.all():
                rec = roc_auc_score(y, v)
                assert abs(rec - ref) <= 1e-6, (name, a, k, rec, ref)
                CHECKED.append(("TimesNet", name, a, k))
                lmv, cov = lm_auroc(y, v, n)
            else:
                lmv, cov = None, None
            row[k] = {"auroc": ref, "length_matched": lmv, "covered": cov}
        out[a] = row
    tn[name] = out

OUT.write_text(
    json.dumps(
        {
            "_sources": {
                "masked_ae_runs": "task/masked_autoencoder/runs/{mlp,resnet}/T{64,48,32}-seed{0..3}/ (config.json, env.json, "
                "summary.json, history.jsonl, test_metrics.json, scores/*.npz)",
                "auroc_check": f"{len(CHECKED)} AUROCs recomputed from score files, |diff| <= 1e-6 vs test_metrics.json / "
                "TimesNet summary.json",
                "length_matched": "masked_autoencoder.ipynb cell 43 length_matched_auroc (same logic, lm_auroc here)",
                "roc": "sklearn roc_curve on the score files, tpr interpolated on a fixed FPR grid (fpr_grid)",
                "t64_labels": src64.label_reads,
                "convae": "baselinepaper/01-our-studies/results.json seed_summary (task/encoder seeds_0_3; "
                "task/autoencoder_benign_t48; task/autoencoder_benign_t32; T32 seed 0 = rounded notebook values); "
                "params from inventory.md (80,189 / 71,741 / 63,293)",
                "timesnet": "src/runs/evaluation/all/T32_benign/model-t32b-seed0{,-random-init}/summary.json metrics + "
                "scores/*.npz (length-matched recomputed with lm_auroc; one seed only)",
                "widedeep": "https://pytorch-widedeep.readthedocs.io/en/latest/examples/15_Self_Supervised_Pretraning_pt1.html "
                "+ pytorch_widedeep 1.7.0 source (read only): masked_prob default 0.2 "
                "(self_supervised_training/encoder_decoder_trainer.py L43/L85), batch_size default 32 (L112), "
                "AdamW default (_base_encoder_decoder_trainer.py L59-63), _forward feeds encoder(X) "
                "(models/tabular/self_supervised/encoder_decoder_model.py L60-70), EncoderDecoderLoss (losses.py)",
            },
            "fpr_grid": [round(float(x), 4) for x in FPR_GRID],
            "summary": summ,
            "convae": convae,
            "timesnet_t32_seed0": tn,
            "per_run": per,
            "summary_resnet": summ_res,
            "per_run_resnet": per_res,
        },
        indent=1,
    )
)
print("checked", len(CHECKED))
