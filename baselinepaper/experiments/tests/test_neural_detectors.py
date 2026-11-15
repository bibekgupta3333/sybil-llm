"""Tests of the neural detectors (`convae`, `lstmae`, `supae`) on tiny synthetic windows (CPU, a few seconds)."""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest
import torch

from baselinepaper.experiments import config as cfg
from baselinepaper.experiments.data import FeatureStatistics, LabelledWindowSet, WindowSet
from baselinepaper.experiments.detectors import available, get_detector
from baselinepaper.experiments.detectors.conv_autoencoder import ConvAutoencoderDetector
from baselinepaper.experiments.detectors.lstm_autoencoder import LSTMAutoencoder, LSTMAutoencoderDetector
from baselinepaper.experiments.detectors.neural_common import NeuralDetector, count_params
from baselinepaper.experiments.detectors.supervised_autoencoder import SupervisedAutoencoderDetector
from task.autoencoder_benign_t32 import config as conv_config
from task.autoencoder_benign_t32.model import ConvAutoencoder
from task.autoencoder_benign_t32.trainer import Trainer as ConvTaskTrainer
from task.autoencoder_benign_t32.trainer import seed_everything
from task.autoencoder_st_t32 import config as sup_config
from task.autoencoder_st_t32.model import SupervisedAutoencoder
from task.autoencoder_st_t32.trainer import Trainer as SupTaskTrainer

T, F = cfg.SEQ_LEN, cfg.N_FEATURES
UNSUPERVISED = (ConvAutoencoderDetector, LSTMAutoencoderDetector)
TINY = {"batch_size": 64, "max_epochs": 3, "eval_batch": 100}


def make_windows(n: int, seed: int, shift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Random-walk windows with random prefix lengths; padding rows are zero."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, T + 1, size=n)
    mask = np.arange(T)[None, :] < lengths[:, None]
    x = (0.3 * rng.normal(size=(n, T, F)).cumsum(axis=1) + shift).astype(np.float32)
    x[~mask] = 0.0
    return x, mask


def unlabelled(n: int, seed: int) -> WindowSet:
    return WindowSet(*make_windows(n, seed))


def labelled(n: int, seed: int) -> LabelledWindowSet:
    """Benign = random walks around 0, attack = the same shifted by +2 (separable)."""
    xb, mb = make_windows(n // 2, seed)
    xa, ma = make_windows(n - n // 2, seed + 1000, shift=2.0)
    labels = np.r_[np.zeros(len(xb)), np.ones(len(xa))].astype(np.int8)
    order = np.random.default_rng(seed).permutation(n)
    return LabelledWindowSet(np.concatenate([xb, xa])[order], np.concatenate([mb, ma])[order], labels[order])


def tiny(detector_cls, regime: str, statistics: FeatureStatistics | None = None, **overrides) -> NeuralDetector:
    """A detector with small training settings, on CPU, silent."""
    detector = detector_cls(regime=regime, device="cpu", statistics=statistics)
    detector.settings = dataclasses.replace(detector.settings, **{**TINY, **overrides})
    detector.log = None
    return detector


def fitted(detector_cls, seed: int = 0, regime: str | None = None, statistics=None) -> NeuralDetector:
    supervised = detector_cls is SupervisedAutoencoderDetector
    detector = tiny(detector_cls, regime or (cfg.SUPERVISED if supervised else cfg.BENIGN_ONLY), statistics)
    make = labelled if supervised else unlabelled
    detector.fit(make(320, 1), make(128, 2), seed)
    return detector


def test_registry_regimes_and_parameter_counts():
    detectors = available()
    assert detectors["convae"].regimes == cfg.UNLABELLED_REGIMES and detectors["convae"].score_names == (
        "recon",
        "maha",
    )
    assert detectors["lstmae"].regimes == cfg.UNLABELLED_REGIMES and detectors["lstmae"].score_names == ("recon",)
    assert detectors["supae"].regimes == (cfg.SUPERVISED,) and detectors["supae"].score_names == ("prob", "recon")
    assert all(detectors[name].is_stochastic for name in ("convae", "lstmae", "supae"))
    conv = count_params(tiny(ConvAutoencoderDetector, cfg.BENIGN_ONLY).build_model())
    lstm = count_params(tiny(LSTMAutoencoderDetector, cfg.BENIGN_ONLY).build_model())
    sup = count_params(tiny(SupervisedAutoencoderDetector, cfg.SUPERVISED).build_model())
    assert conv == conv_config.TOTAL_PARAMS == 63_293
    assert sup == sup_config.TOTAL_PARAMS == 63_870
    assert lstm == 64_429 and abs(lstm - conv) / conv < 0.15
    with pytest.raises(ValueError):
        ConvAutoencoderDetector(regime=cfg.SUPERVISED)
    with pytest.raises(ValueError):
        SupervisedAutoencoderDetector(regime=cfg.BENIGN_ONLY)


def test_frozen_settings_come_from_the_task_packages():
    conv, sup = conv_config.AutoencoderConfig(), sup_config.SupervisedAEConfig()
    shared = ("lr", "weight_decay", "batch_size", "patience", "min_delta", "plateau_factor", "plateau_patience")
    for name in (*shared, "grad_clip", "eval_batch", "max_epochs"):
        assert getattr(ConvAutoencoderDetector.settings, name) == getattr(conv, name)
        assert getattr(SupervisedAutoencoderDetector.settings, name) == getattr(sup, name)
    for name in (*shared, "grad_clip", "eval_batch"):  # lstmae = convae except the architecture and the epoch cap
        assert getattr(LSTMAutoencoderDetector.settings, name) == getattr(ConvAutoencoderDetector.settings, name)
    assert LSTMAutoencoderDetector.settings.max_epochs <= ConvAutoencoderDetector.settings.max_epochs
    hyper = get_detector("convae")(regime=cfg.BENIGN_ONLY).hyperparameters()
    assert hyper["architecture"]["mahalanobis"]["max_cond"] == conv.mahalanobis_max_cond


@pytest.mark.parametrize("detector_cls", (*UNSUPERVISED, SupervisedAutoencoderDetector))
def test_scores_have_one_finite_value_per_window(detector_cls):
    detector = fitted(detector_cls)
    x, mask = make_windows(257, 3)
    scores = detector.score(x, mask)
    assert set(scores) == set(detector.score_names)
    for values in scores.values():
        assert values.shape == (257,) and values.dtype == np.float64 and np.isfinite(values).all()
    assert 1 <= len(detector.history()) <= detector.settings.max_epochs
    info = detector.fit_info()
    assert info["device"] == "cpu" and info["train_windows"] == 320 and info["val_windows"] == 128
    assert info["parameters"] == count_params(detector.trainer.model)
    assert 1 <= info["best_epoch"] <= info["epochs_run"]
    if detector_cls is SupervisedAutoencoderDetector:
        assert ((scores["prob"] >= 0) & (scores["prob"] <= 1)).all()
        assert info["pos_weight"] == pytest.approx(1.0) and info["monitor"] == "val_auroc"


@pytest.mark.parametrize("detector_cls", UNSUPERVISED)
def test_windows_far_from_training_data_score_higher(detector_cls):
    detector = fitted(detector_cls)
    near_x, near_mask = make_windows(200, 4)
    far_x, far_mask = make_windows(200, 4, shift=8.0)
    near, far = detector.score(near_x, near_mask), detector.score(far_x, far_mask)
    for name in detector.score_names:
        assert np.median(far[name]) > np.quantile(near[name], 0.95), name


def test_supervised_probability_is_higher_for_the_attack_class():
    detector = fitted(SupervisedAutoencoderDetector)
    benign_x, benign_mask = make_windows(200, 5)
    attack_x, attack_mask = make_windows(200, 6, shift=2.0)
    assert detector.score(attack_x, attack_mask)["prob"].mean() > detector.score(benign_x, benign_mask)["prob"].mean()


@pytest.mark.parametrize("detector_cls", (*UNSUPERVISED, SupervisedAutoencoderDetector))
@pytest.mark.parametrize("restandardised", (False, True))
def test_padding_rows_do_not_change_scores(detector_cls, restandardised):
    statistics, regime = None, None
    if restandardised and detector_cls is not SupervisedAutoencoderDetector:
        statistics = FeatureStatistics(np.full(F, 0.2), np.full(F, 1.7), np.full(F, 1000))
        regime = cfg.MIX_GRIDFOLDERS
    elif restandardised:
        pytest.skip("the supervised regime keeps the cached normalisation")
    detector = fitted(detector_cls, regime=regime, statistics=statistics)
    x, mask = make_windows(150, 7)
    changed = x.copy()
    changed[~mask] = np.random.default_rng(0).normal(scale=50.0, size=(int((~mask).sum()), F)).astype(np.float32)
    assert (~mask).any() and not np.array_equal(changed, x)
    before, after = detector.score(x, mask), detector.score(changed, mask)
    for name in detector.score_names:
        np.testing.assert_allclose(after[name], before[name], rtol=1e-6, atol=1e-9, err_msg=name)


def test_lstm_latent_is_the_state_at_the_last_real_row():
    """z of a padded window equals z of the same window cut to its real rows (packed-sequence behaviour)."""
    torch.manual_seed(0)
    model = LSTMAutoencoder().eval()
    x, mask = make_windows(20, 8)
    with torch.no_grad():
        recon, z = model(torch.from_numpy(x), torch.from_numpy(mask))
        for i in range(len(x)):
            n = int(mask[i].sum())
            recon_cut, z_cut = model(torch.from_numpy(x[i : i + 1, :n]), torch.from_numpy(mask[i : i + 1, :n]))
            torch.testing.assert_close(z[i : i + 1], z_cut, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(recon[i : i + 1, :n], recon_cut, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("detector_cls", (*UNSUPERVISED, SupervisedAutoencoderDetector))
def test_same_seed_same_scores_on_cpu(detector_cls):
    x, mask = make_windows(100, 9)
    first, again, other = fitted(detector_cls, seed=3), fitted(detector_cls, seed=3), fitted(detector_cls, seed=4)
    assert first.history() != other.history()
    for name in first.score_names:
        np.testing.assert_array_equal(first.score(x, mask)[name], again.score(x, mask)[name])
        assert not np.array_equal(first.score(x, mask)[name], other.score(x, mask)[name])
    strip = lambda rows: [{k: v for k, v in row.items() if k != "seconds"} for row in rows]  # noqa: E731
    assert strip(first.history()) == strip(again.history())


@pytest.mark.parametrize("detector_cls", UNSUPERVISED)
def test_standardise_is_applied_in_fit_and_score(detector_cls):
    statistics = FeatureStatistics(np.linspace(-1, 1, F), np.linspace(0.5, 3, F), np.full(F, 1000))
    detector = tiny(detector_cls, cfg.MIX_ALL, statistics)
    calls = []
    original = detector.standardise
    detector.standardise = lambda x, mask: (calls.append(len(x)), original(x, mask))[1]
    train, val = unlabelled(320, 1), unlabelled(128, 2)
    detector.fit(train, val, 0)
    assert calls == [320, 128]
    x, mask = make_windows(250, 3)
    scores = detector.score(x, mask)
    assert calls[2:] == [100, 100, 50]
    # Scoring raw x with the regime statistics equals scoring pre-standardised x with no statistics.
    detector.standardise = lambda x, mask: x
    expected = detector.score(statistics.apply(x, mask), mask)
    for name in detector.score_names:
        np.testing.assert_allclose(scores[name], expected[name], rtol=1e-6)


def test_supervised_score_takes_no_labels_and_fit_requires_them():
    assert list(inspect.signature(SupervisedAutoencoderDetector.score).parameters) == ["self", "x", "mask"]
    detector = fitted(SupervisedAutoencoderDetector)
    seen = []
    forward = detector.trainer.model.forward
    detector.trainer.model.forward = lambda *args, **kwargs: (seen.append((len(args), sorted(kwargs))), forward(*args))[
        1
    ]
    x, mask = make_windows(60, 10)
    scores = detector.score(x, mask)
    assert seen == [(2, [])] and scores["prob"].shape == (60,)  # the model gets x and mask, nothing else
    with pytest.raises(AttributeError):
        tiny(SupervisedAutoencoderDetector, cfg.SUPERVISED).fit(unlabelled(64, 1), unlabelled(64, 2), 0)
    one_class = LabelledWindowSet(*make_windows(64, 1), np.zeros(64, dtype=np.int8))
    with pytest.raises(ValueError):
        tiny(SupervisedAutoencoderDetector, cfg.SUPERVISED).fit(one_class, one_class, 0)


def test_pos_weight_comes_from_the_training_labels():
    detector = tiny(SupervisedAutoencoderDetector, cfg.SUPERVISED)
    x, mask = make_windows(200, 11)
    labels = np.r_[np.zeros(150), np.ones(50)].astype(np.int8)
    x[labels == 1] += 2.0 * mask[labels == 1][..., None]
    detector.fit(LabelledWindowSet(x, mask, labels), labelled(128, 2), 0)
    assert detector.fit_info()["pos_weight"] == pytest.approx(3.0)
    assert detector.fit_info()["train_class_counts"] == {"benign": 150, "gridsybil": 50}


def test_training_inputs_are_not_modified_and_read_only_arrays_work():
    train, val = unlabelled(320, 1), unlabelled(128, 2)
    copy = train.x.copy()
    train.x.setflags(write=False)
    detector = tiny(ConvAutoencoderDetector, cfg.BENIGN_ONLY)
    detector.fit(train, val, 0)
    np.testing.assert_array_equal(train.x, copy)
    read_only, mask = make_windows(50, 3)
    read_only.setflags(write=False)
    assert detector.score(read_only, mask)["recon"].shape == (50,)


def test_convae_loop_equals_the_task_trainer(tmp_path):
    """Same seed, data and settings: this package's loop and task/autoencoder_benign_t32's give the same model."""
    train, val = unlabelled(320, 1), unlabelled(128, 2)
    detector = tiny(ConvAutoencoderDetector, cfg.BENIGN_ONLY, max_epochs=4)
    detector.fit(train, val, 5)
    settings = dataclasses.replace(
        conv_config.AutoencoderConfig(),
        seed=5,
        device="cpu",
        max_epochs=4,
        **{k: v for k, v in TINY.items() if k != "max_epochs"},
    )
    seed_everything(5)
    task = ConvTaskTrainer(ConvAutoencoder(settings), settings, torch.device("cpu"))
    result = task.fit((train.x, train.mask), (val.x, val.mask), tmp_path)
    ours = detector.history()
    assert [row["val_loss"] for row in ours] == pytest.approx(
        [float(v) for v in _history(tmp_path, "val_loss")], rel=1e-9
    )
    assert [row["train_loss"] for row in ours] == pytest.approx(_history(tmp_path, "train_loss"), rel=1e-9)
    assert detector.fit_info()["best_epoch"] == result["best_epoch"]
    x, mask = make_windows(100, 3)
    reference = task.score(x, mask)
    np.testing.assert_allclose(detector.score(x, mask)["recon"], reference["error"], rtol=1e-6)
    np.testing.assert_allclose(detector.forward(x, mask)["z"], reference["z"], rtol=1e-6, atol=1e-7)


def test_supae_loop_equals_the_task_trainer(tmp_path):
    """Same seed, data and settings: this package's loop and task/autoencoder_st_t32's give the same model."""
    train, val = labelled(320, 1), labelled(128, 2)
    detector = tiny(SupervisedAutoencoderDetector, cfg.SUPERVISED, max_epochs=4)
    detector.fit(train, val, 5)
    settings = dataclasses.replace(
        sup_config.SupervisedAEConfig(),
        seed=5,
        device="cpu",
        max_epochs=4,
        **{k: v for k, v in TINY.items() if k != "max_epochs"},
    )
    seed_everything(5)
    pos_weight = float((train.labels == 0).sum() / (train.labels == 1).sum())
    task = SupTaskTrainer(SupervisedAutoencoder(settings), settings, torch.device("cpu"), pos_weight)
    as_float = lambda s: (s.x, s.mask, s.labels.astype(np.float32))  # noqa: E731
    result = task.fit(as_float(train), as_float(val), tmp_path)
    ours = detector.history()
    assert [row["val_auroc"] for row in ours] == pytest.approx(_history(tmp_path, "val_auroc"), rel=1e-9)
    assert [row["val_loss"] for row in ours] == pytest.approx(_history(tmp_path, "val_loss"), rel=1e-6)
    assert detector.fit_info()["best_epoch"] == result["best_epoch"]
    x, mask = make_windows(100, 3)
    reference = task.predict(x, mask)
    scores = detector.score(x, mask)
    np.testing.assert_allclose(scores["prob"], reference["prob"], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(scores["recon"], reference["error"], rtol=1e-6)


def _history(run_dir, key: str) -> list[float]:
    import json

    return [json.loads(line)[key] for line in (run_dir / "history.jsonl").read_text().splitlines()]


def test_save_writes_weights_and_mahalanobis(tmp_path):
    detector = fitted(ConvAutoencoderDetector)
    detector.save(tmp_path)
    state = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    assert set(state["model"]) == set(detector.trainer.model.state_dict())
    saved = np.load(tmp_path / "mahalanobis.npz")
    np.testing.assert_array_equal(saved["mean"], detector.mahalanobis.mean)
    assert detector.fit_info()["mahalanobis"]["fitted_windows"] == 320
