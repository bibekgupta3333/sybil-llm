"""Build baselinepaper/masked-autoencoder-slides.html from mae_deck_values.json + deck_base.json.

Usage (any Python 3, no third-party packages):  python3 build_deck.py <out.html>
  mae_deck_values.json  written by extract_values.py (run from the repo root with .venv-train/bin/python)
  deck_base.json        CSS, navigation script and SVG defs copied from baselinepaper/baseline-seeds-slides.html
Every number on the slides comes from mae_deck_values.json, except the fixed method settings quoted from the package
source (config.py / model.py / README.md) and widedeep 1.7.0 source line numbers, which are written as text.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
V = json.loads((HERE / "mae_deck_values.json").read_text())
BASE = json.loads((HERE / "deck_base.json").read_text())
OUT = Path(sys.argv[1])
S, PER, CONV, TN, GRID = V["summary"], V["per_run"], V["convae"], V["timesnet_t32_seed0"], V["fpr_grid"]
W, SEEDS = (64, 48, 32), (0, 1, 2, 3)
ATT = ("GridSybil", "DataReplay", "DoSRandom", "DoSDisruptive")
STAGES = ["Method", "Setup", "Results", "Checks", "Wrap-up"]
WIDEDEEP_URL = "https://pytorch-widedeep.readthedocs.io/en/latest/examples/15_Self_Supervised_Pretraning_pt1.html"


def f3(x):
    return f"{x:.3f}"


def ms(d, dp=3):
    return f"{d['mean']:.{dp}f} ± {d['sd']:.{dp}f}"


def n0(x):
    return f"{x:,}"


def pct(x, dp=1):
    return f"{100 * x:.{dp}f}%"


def strip(stage):
    parts = [f'<span class="on">{s}</span>' if s == stage else f"<span>{s}</span>" for s in STAGES]
    return '<div class="strip" aria-label="sections">' + "<i>&rsaquo;</i>".join(parts) + "</div>"


SLIDES = []


def slide(stage, body, notes, cls=""):
    c = f"slide {cls}".strip()
    SLIDES.append(f'<section class="{c}" data-stage="{stage}">\n{strip(stage)}\n{body}\n'
                  f'<aside class="notes">{notes}</aside>\n</section>')


def gs(t):
    return S[f"T{t} GridSybil"]


TR = {t: S[f"T{t} training"] for t in W}
SR, PERR = V["summary_resnet"], V["per_run_resnet"]


def gr(t):
    return SR[f"T{t} GridSybil"]


TRR = {t: SR[f"T{t} training"] for t in W}

# ---- consistency checks of the numbers that the text states in words
for t in W:
    p = TR[t]["params"]
    calc = (14 * t * 64 + 64) + (64 * 16 + 16) + (16 * 64 + 64) + (64 * 13 * t + 13 * t)
    assert p == calc, (t, p, calc)
    assert TR[t]["counts"]["test_vehicles_in_train_or_val"] == 0
    assert all(r == "early_stopping" for r in [PER[f"T{t}-seed{s}"]["stop_reason"] for s in SEEDS])
    assert TR[t]["devices"] == ["mps"]
    assert all(PER[f"T{t}-seed{s}"]["config"]["masked_prob"] == 0.2 for s in SEEDS)
    assert all(PER[f"T{t}-seed{s}"]["config"]["feed_masked_input"] for s in SEEDS)
    assert all(PER[f"T{t}-seed{s}"]["config"]["mlp_hidden_dims"] == [64, 16] for s in SEEDS)
    assert all(PER[f"T{t}-seed{s}"]["config"]["batch_size"] == 256 for s in SEEDS)
    assert gs(t)["masked_recon"]["mean"] < CONV[f"T{t} GridSybil"]["recon"]["mean"]
    assert gs(t)["mahalanobis"]["max"] < 0.5
for a in ATT:
    assert S[f"T32 {a}"]["mahalanobis"]["max"] < 0.5
all_min = min(TR[t]["train_minutes"][i] for t in W for i in range(4))
all_max = max(TR[t]["train_minutes"][i] for t in W for i in range(4))
ratio_lo = min(TR[t]["train_over_val_at_best"]["min"] for t in W)
ratio_hi = max(TR[t]["train_over_val_at_best"]["max"] for t in W)
dec_lo = min(TR[t]["share_of_val_decrease_by_epoch10"]["min"] for t in W)
dec_hi = max(TR[t]["share_of_val_decrease_by_epoch10"]["max"] for t in W)
env0 = PER["T32-seed0"]["env"]
commits = {PER[f"T{t}-seed{s}"]["env"]["git_commit"][:7] for t in W for s in SEEDS}
dirty = {PER[f"T{t}-seed{s}"]["env"]["git_dirty"] for t in W for s in SEEDS}
gap = {t: CONV[f"T{t} GridSybil"]["recon"]["mean"] - gs(t)["masked_recon"]["mean"] for t in W}
tn_p, tn_r = TN["model-t32b-seed0"], TN["model-t32b-seed0-random-init"]
dr = S["T32 DataReplay"]
assert dr["masked_recon_length_matched"]["mean"] < 0.5 < dr["masked_recon"]["mean"]
# TabResnet (encoder "resnet"): same data, loss, schedule; only the encoder / decoder differ
for t in W:
    assert TRR[t]["counts"] == TR[t]["counts"] and TRR[t]["devices"] == ["mps"]
    for s in SEEDS:
        cf = PERR[f"T{t}-seed{s}"]["config"]
        assert cf["encoder"] == "resnet" and cf["blocks_dims"] == [64, 32, 16] and cf["masked_prob"] == 0.2
        assert cf["feed_masked_input"] and cf["batch_size"] == 256
        assert PERR[f"T{t}-seed{s}"]["stop_reason"] == "early_stopping"
        assert PER[f"T{t}-seed{s}"]["config"]["encoder"] == "mlp"
    assert gs(t)["masked_recon"]["mean"] < gr(t)["masked_recon"]["mean"] < CONV[f"T{t} GridSybil"]["recon"]["mean"]
    assert gr(t)["masked_recon"]["max"] < CONV[f"T{t} GridSybil"]["recon"]["min"]
    assert gr(t)["mahalanobis"]["max"] < 0.5
drr = SR["T32 DataReplay"]
res_gain = {t: gr(t)["masked_recon"]["mean"] - gs(t)["masked_recon"]["mean"] for t in W}
gap_r = {t: CONV[f"T{t} GridSybil"]["recon"]["mean"] - gr(t)["masked_recon"]["mean"] for t in W}
commits_r = {PERR[f"T{t}-seed{s}"]["env"]["git_commit"][:7] for t in W for s in SEEDS}
dirty_r = {PERR[f"T{t}-seed{s}"]["env"]["git_dirty"] for t in W for s in SEEDS}
min_mlp = min(TR[t]["train_minutes"][i] for t in W for i in range(4))
max_mlp = max(TR[t]["train_minutes"][i] for t in W for i in range(4))
min_res = min(TRR[t]["train_minutes"][i] for t in W for i in range(4))
max_res = max(TRR[t]["train_minutes"][i] for t in W for i in range(4))
maha_hi_res = max(SR[f"T32 {a}"]["mahalanobis"]["mean"] for a in ATT)

# ---------------------------------------------------------------- SVG helpers
MK = {"id": "ah0"}


def arrow_defs(name):
    """Marker defs with an id unique to one SVG (a marker inside a hidden slide does not render elsewhere)."""
    MK["id"] = f"ah-{name}"
    return (f'<defs><marker id="{MK["id"]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="ahp"/></marker></defs>')


def box(o, x, y, w, h, l1, l2="", cls="bx", fs1=17, fs2=14.5):
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="{cls}"/>')
    cx = x + w / 2
    if l2:
        o.append(f'<text x="{cx}" y="{y + h / 2 - 4}" font-size="{fs1}" font-weight="700" text-anchor="middle">{l1}</text>')
        o.append(f'<text x="{cx}" y="{y + h / 2 + 18}" font-size="{fs2}" text-anchor="middle" class="mut">{l2}</text>')
    else:
        o.append(f'<text x="{cx}" y="{y + h / 2 + 6}" font-size="{fs1}" font-weight="700" text-anchor="middle">{l1}</text>')


def arr(o, x1, y1, x2, y2, cls="arr"):
    o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{cls}" marker-end="url(#{MK["id"]})"/>')


def txt(o, x, y, s, fs=15, anchor="start", cls="", weight=""):
    c = f' class="{cls}"' if cls else ""
    wt = f' font-weight="{weight}"' if weight else ""
    o.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{fs}" text-anchor="{anchor}"{c}{wt}>{s}</text>')


def roc_panel(o, x0, y0, size, curves, title, fs=14):
    """curves: list of (tpr list on GRID, css class, extra style)."""
    o.append(f'<rect x="{x0}" y="{y0}" width="{size}" height="{size}" class="frame"/>')
    for v in (0.5,):
        o.append(f'<line x1="{x0 + v * size:.1f}" y1="{y0}" x2="{x0 + v * size:.1f}" y2="{y0 + size}" class="gl"/>')
        o.append(f'<line x1="{x0}" y1="{y0 + size - v * size:.1f}" x2="{x0 + size}" y2="{y0 + size - v * size:.1f}" class="gl"/>')
    o.append(f'<line x1="{x0}" y1="{y0 + size}" x2="{x0 + size}" y2="{y0}" class="diag"/>')
    for tpr, cls, style in curves:
        pts = " ".join(f"{x0 + f * size:.1f},{y0 + size - t * size:.1f}" for f, t in zip(GRID, tpr))
        o.append(f'<polyline points="{pts}" class="crv {cls}" style="{style}"/>')
    txt(o, x0 + size / 2, y0 - 10, title, fs, "middle", weight="700")
    for v, lab in ((0, "0"), (0.5, "0.5"), (1, "1")):
        txt(o, x0 + v * size, y0 + size + 18, lab, 13, "middle", "mut")
        txt(o, x0 - 6, y0 + size - v * size + 4, lab, 13, "end", "mut")


CSS_EXTRA = """
:root { --full: #7a5af5; --tn: #8a4bbf; --res: #c0398a; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --full: #b39dff; --tn: #c99ae6; --res: #f28cc4; } }
:root[data-theme="dark"] { --full: #b39dff; --tn: #c99ae6; --res: #f28cc4; }
svg .c-res { fill: var(--res); } svg .l-res { stroke: var(--res); } .sw.res { background: var(--res); }
svg .ptr { fill: var(--res); stroke: var(--bg); stroke-width: 1.2; }
svg .l-full { stroke: var(--full); } svg .c-full { fill: var(--full); } svg .l-maha { stroke: var(--bar-s); }
svg .c-maha { fill: var(--bar-s); } svg .l-len { stroke: var(--text); stroke-dasharray: 5 4; }
svg .c-tn { fill: var(--tn); } svg .l-tn { stroke: var(--tn); } svg .l-oc2 { stroke: var(--oc); }
svg .hid { fill: var(--val); } svg .real { fill: var(--surface); stroke: var(--border); stroke-width: 1; }
svg .pad { fill: url(#hatchG); stroke: var(--border); stroke-width: 1; }
svg .g0 { fill: var(--s0); } svg .g1 { fill: var(--s1); } svg .g2 { fill: var(--s2); } svg .g3 { fill: var(--s3); }
svg .g4 { fill: var(--mae); }
svg .ptm { fill: var(--mae); stroke: var(--bg); stroke-width: 1.2; } svg .ptf { fill: var(--full); stroke: var(--bg); stroke-width: 1.2; }
svg .ring { fill: var(--bg); stroke-width: 2.5; }
.sw.full { background: var(--full); } .sw.mah { background: var(--bar-s); } .sw.tn { background: var(--tn); }
table.dv { font-size: 16px; } table.dv th { font-size: 14.5px; } table.dv td, table.dv th { padding: 6px 8px; line-height: 1.32; }
table.dv td:first-child { font-weight: 650; }
table.cnt { font-size: 18px; } table.cnt th { font-size: 15px; } table.cnt td, table.cnt th { padding: 6px 9px; }
table.cnt td:first-child { font-weight: 650; white-space: nowrap; }
table.res { font-size: 19px; } table.res th { font-size: 15px; } table.res td, table.res th { padding: 7px 9px; }
table.res td:first-child { white-space: nowrap; } table.res tr.refrow td { background: var(--surface); color: var(--muted); }
table.res tr.hlrow td { font-weight: 650; }
ul.cap2 { font-size: 18px; line-height: 1.38; padding-left: 22px; margin: 0; } ul.cap2 li { margin-bottom: 9px; }
.fm { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 16px; background: var(--surface);
  border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; line-height: 1.5; }
.mini h3 { font-size: 19px; color: var(--accent); margin: 0 0 4px; } .mini p { font-size: 17px; color: var(--muted); margin: 0; line-height: 1.38; }
.mini .card { padding: 12px 16px; }
"""

# ================================================================ 1 title
SLIDES.append(f"""<section class="slide title-slide" data-stage="">
{strip("")}
<div class="kicker">Baseline · one-class masked autoencoder</div>
<h1>A Masked Autoencoder Baseline for Sybil Detection</h1>
<p class="sub">Widedeep-style masked-cell reconstruction, trained on benign windows only, two encoders (TabMlp, TabResnet), at T = 64, 48 and 32 (4 seeds each)</p>
<p class="author">Bibek Gupta<br><span>Department of Computer Science, Florida Polytechnic University</span></p>
<p class="foot">18 November 2026 · side experiment, not the thesis TimesNet model · <code>task/masked_autoencoder/</code></p>
<aside class="notes">This is one more baseline. It is a masked autoencoder, built after the pytorch-widedeep library. It learns only from benign windows. I trained it with two encoders, TabMlp and TabResnet, at three window sizes with four seeds each: 24 runs. The short answer: TabResnet is better than TabMlp, but both are weaker than our one-class ConvAE, and part of the score is window length.</aside>
</section>""")

# ================================================================ 2 why
slide("Method", f"""<h2>Why try a masked autoencoder?</h2>
<p class="lead">Hide some input cells. Ask the model to fill them in. No attack labels are needed.</p>
<div class="cards3 qs" style="margin-top: 10px">
  <div class="card"><h3>Masked-cell reconstruction</h3><p>Each step hides 20% of the cells. The model must guess them from the other cells. So it has to learn how the 13 features move together.</p></div>
  <div class="card"><h3>Self-supervised objective</h3><p>The target is the input itself. The loss counts only the hidden cells. No label enters the loss.</p></div>
  <div class="card"><h3>One-class anomaly score</h3><p>Train on benign windows only. A window that is hard to fill in is unusual. High error = alarm.</p></div>
</div>
<div class="sub2">
  <div><h3>Question</h3><p>Does masked reconstruction find GridSybil better than plain reconstruction (our ConvAE)?</p></div>
  <div><h3>Same data</h3><p>Exactly the windows of the one-class ConvAE cells at T = 64, 48, 32.</p></div>
  <div><h3>Source of the method</h3><p>pytorch-widedeep self-supervised pre-training (encoder-decoder, TabNet style).</p></div>
</div>
<p class="foot">Idea from TabNet's self-supervised pre-training as implemented in pytorch-widedeep: <span style="font-size:14px">{WIDEDEEP_URL}</span></p>""",
      "A plain autoencoder can learn to copy its input. A masked one cannot copy hidden cells. It must use the relations between features. The hope is that these relations are what a fake vehicle breaks. We train on benign windows only, so the score is how badly the model fills in a window.")

# ================================================================ 3 widedeep method
o = [f'<svg width="100%" viewBox="0 0 1160 300" role="img" aria-label="widedeep encoder-decoder pre-training: input, '
     f'RandomObfuscator, TabMlp encoder, mirrored decoder, EncoderDecoderLoss on hidden cells">', arrow_defs("wd")]
box(o, 0, 70, 170, 80, "Table row X", "continuous columns")
box(o, 215, 70, 210, 80, "RandomObfuscator", "M ~ Bernoulli(0.2) per cell", "bx-w")
box(o, 470, 70, 190, 80, "TabMlp encoder", "MLP, ReLU, dropout", "bx-k")
box(o, 705, 70, 200, 80, "TabMlpDecoder", "widths mirrored", "bx-k")
box(o, 950, 70, 210, 80, "EncoderDecoderLoss", "only cells with M = 1", "bx-w")
arr(o, 170, 110, 213, 110)
arr(o, 425, 110, 468, 110)
arr(o, 660, 110, 703, 110)
arr(o, 905, 110, 948, 110)
txt(o, 447, 62, "X·(1−M)", 14, "middle", "mut")
o.append(f'<path d="M 85 150 L 85 205 L 565 205 L 565 152" class="arr-d" marker-end="url(#{MK["id"]})"/>')
txt(o, 325, 228, "widedeep 1.7.0, TabMlp / TabResnet path: the encoder is called on the clean X", 15, "middle", "mut")
txt(o, 325, 248, "(the masked input is computed but not used; only the TabNet path uses it)", 15, "middle", "mut")
txt(o, 580, 30, "EncoderDecoderTrainer(encoder=TabMlp(...), masked_prob=0.2)  ·  AdamW by default  ·  batch 32 by default", 16, "middle", weight="700")
o.append("</svg>")
slide("Method", f"""<h2>The widedeep method: hide 20% of cells, rebuild them</h2>
{''.join(o)}
<div class="cols" style="gap: 24px; flex: none">
<div style="flex: 0 0 640px"><div class="fm">loss = mean over rows i of<br>&nbsp;&nbsp;Σ<sub>j</sub> M<sub>ij</sub> (x̂<sub>ij</sub> − x<sub>ij</sub>)² / var<sub>j</sub> &nbsp;/&nbsp; Σ<sub>j</sub> M<sub>ij</sub><br><span style="color:var(--muted)">var<sub>j</sub> = batch variance of column j</span></div></div>
<div><ul class="cap2"><li>Hidden cells are set to 0 (mask is drawn on the CPU per cell).</li><li>Eval mode: mask of ones, nothing hidden.</li><li>Decoder = encoder widths reversed.</li></ul></div>
</div>
<p class="foot">Sources: widedeep docs example 15 (Self-Supervised Pre-Training pt 1); widedeep 1.7.0 source, read only: <code>_random_obfuscator.py</code>, <code>encoder_decoder_model.py</code> L60–70, <code>losses.py</code> EncoderDecoderLoss, <code>encoder_decoder_trainer.py</code> (masked_prob 0.2, batch 32).</p>""",
      "This is the library method. A mask hides each cell with probability 0.2. An MLP encoder and a mirrored decoder rebuild the row. The loss looks only at hidden cells, and each column is divided by its batch variance. One surprise in the source: for TabMlp the encoder receives the clean input, not the masked one. I come back to this on the deviations slide.")

# ================================================================ 4 adaptation
o = [f'<svg width="100%" viewBox="0 0 1160 330" role="img" aria-label="Adaptation to windows: T x 13 window with row '
     f'mask, padding never hidden, flattened to 14 T columns, TabMlp, z = 16, decoder to 13 T">', arrow_defs("ad")]
gx, gy, cw, ch, rows, cols = 10, 40, 16, 16, 12, 13
hidden_cells = {(0, 2), (0, 9), (1, 5), (2, 0), (2, 11), (3, 7), (4, 3), (4, 12), (5, 1), (5, 8), (6, 6), (7, 4), (7, 10), (8, 2)}
for r in range(rows):
    for c in range(cols):
        cls = "pad" if r >= 9 else ("hid" if (r, c) in hidden_cells else "real")
        o.append(f'<rect x="{gx + c * cw}" y="{gy + r * ch}" width="{cw - 2}" height="{ch - 2}" class="{cls}"/>')
    mcls = "real" if r < 9 else "pad"
    o.append(f'<rect x="{gx + cols * cw + 8}" y="{gy + r * ch}" width="{cw - 2}" height="{ch - 2}" class="{mcls}"/>')
txt(o, gx, gy - 12, "window: T × 13", 14, weight="700")
txt(o, gx + cols * cw + 15, gy - 12, "mask", 13, "middle", "mut")
lg = gy + rows * ch + 22
o.append(f'<rect x="{gx}" y="{lg - 11}" width="13" height="13" class="real"/>')
txt(o, gx + 18, lg, "real cell", 13, cls="mut")
o.append(f'<rect x="{gx + 92}" y="{lg - 11}" width="13" height="13" class="hid"/>')
txt(o, gx + 110, lg, "hidden (20%)", 13, cls="mut")
o.append(f'<rect x="{gx}" y="{lg + 9}" width="13" height="13" class="pad"/>')
txt(o, gx + 18, lg + 20, "padding: never hidden, never in loss or score", 13, cls="mut")
x1 = 260
box(o, x1, 80, 190, 84, "Flatten", "13 T + T mask = 14 T")
box(o, x1 + 230, 80, 190, 84, "Encoder", "TabMlp or TabResnet", "bx-k")
box(o, x1 + 460, 80, 150, 84, "z", "16 numbers", "bx")
box(o, x1 + 650, 80, 240, 84, "Decoder", "mirrored, then Linear → 13 T", "bx-k")
arr(o, 232, 122, x1 - 2, 122)
arr(o, x1 + 190, 122, x1 + 228, 122)
arr(o, x1 + 420, 122, x1 + 458, 122)
arr(o, x1 + 610, 122, x1 + 648, 122)
box(o, x1 + 230, 210, 660, 70, "Loss: widedeep EncoderDecoderLoss, hidden real cells only",
    "variance per feature (13 values), pooled over the batch's real cells", "bx-w", 16, 14.5)
arr(o, x1 + 770, 164, x1 + 770, 208)
txt(o, x1 + 325, 50, "TabMlp: 14 T → 64 → 16 (Linear, ReLU, Dropout 0.1)", 13, "middle", "mut")
txt(o, x1 + 325, 68, "TabResnet: 14 T → 64 → 32 → 16 (dense residual blocks, Dropout 0.1)", 13, "middle", "mut")
o.append("</svg>")
slide("Method", f"""<h2>How we adapted it to message windows</h2>
<p class="lead">A window is a small table: T messages × 13 features, plus a row mask for padding.</p>
{''.join(o)}
<div class="sub2" style="margin-top: 6px">
  <div><h3>Input</h3><p>{14 * 64} / {14 * 48} / {14 * 32} input columns at T = 64 / 48 / 32.</p></div>
  <div><h3>Reconstruction</h3><p>13 T outputs; padding cells are zero and never scored.</p></div>
  <div><h3>Training mask</h3><p>A new random 20% of real cells every step (seeded). Same for both encoders.</p></div>
</div>
<p class="foot">Source: <code>task/masked_autoencoder/model.py</code> (MaskedTabularAutoencoder, RandomObfuscator, EncoderDecoderLoss) and <code>config.py</code>.</p>""",
      "Widedeep works on tables. Our window is a table too: T rows and 13 features. We flatten it and add the row mask as extra columns, like the mask channel of the ConvAE. Only real cells can be hidden. Padding is never hidden, never in the loss and never in the score.")

# ================================================================ 5 deviations
slide("Method", """<h2>Six deviations from widedeep 1.7.0, and why</h2>
<table class="dv">
<tr><th style="width:250px">What</th><th style="width:330px">widedeep 1.7.0</th><th style="width:250px">Ours</th><th>Why</th></tr>
<tr><td>1. Encoder input</td><td>TabMlp / TabResnet path encodes the clean X; masked X is computed, not used</td><td>encoder sees the masked input</td><td>Otherwise it is a plain AE with a random-cell loss. Literal mode kept behind a flag, not run.</td></tr>
<tr><td>2. Decoder output</td><td>ReLU + dropout also after the last layer</td><td>last layer plain Linear</td><td>z-scored features are negative; ReLU cannot output them.</td></tr>
<tr><td>3. Padding</td><td>no padding concept</td><td>row mask as input; padding never hidden or scored</td><td>Windows have 1 to T real rows.</td></tr>
<tr><td>4. Loss variance</td><td>one variance per column</td><td>one per feature (13), over real cells</td><td>A late time step can have no real cell in a batch. Same as widedeep when T = 1 (tested).</td></tr>
<tr><td>5. Schedule</td><td>batch 32, fixed epochs, no LR schedule</td><td>batch 256, early stopping, ReduceLROnPlateau</td><td>Same protocol as the ConvAE cells we compare with.</td></tr>
<tr><td>6. Widths</td><td>TabMlp [200, 100], TabResnet [200, 100, 100]</td><td>TabMlp [64, 16], TabResnet [64, 32, 16]; z = 16</td><td>Similar size to the ConvAEs (z = 16).</td></tr>
</table>
<p class="note" style="font-size: 17px">Validation loss = masked-reconstruction MSE with 5 fixed masks (widedeep's eval loss uses a mask of ones on the clean input).</p>
<p class="foot">Sources: <code>task/masked_autoencoder/README.md</code> "Deviations from widedeep"; widedeep source <code>encoder_decoder_model.py</code> L64–65, <code>mlp/_layers.py</code> L20–21, <code>losses.py</code> L954–970, <code>tab_mlp.py</code> defaults.</p>""",
      "These are all the places where we differ from the library. The first one matters most. In widedeep 1.7.0 the TabMlp encoder gets the clean input, so the model never has to fill in a hidden cell. We feed the masked input, which is the real masked-autoencoder idea. The literal version is in the code behind a flag but I did not run it.")

# ================================================================ 6 data
c = {t: TR[t]["counts"] for t in W}


def att_cell(t):
    tc = c[t]["test"]
    if t == 32:
        return (f"GridSybil {n0(tc['GridSybil'])} · DataReplay {n0(tc['DataReplay'])}<br>"
                f"DoSRandom {n0(tc['DoSRandom'])} · DoSDisruptive {n0(tc['DoSDisruptive'])}")
    return f"GridSybil {n0(tc['GridSybil'])}"


rows = ""
names = {64: ("benign_gridsybil/T64", "task/encoder (AE mode)", "10% of train vehicles (seed 0)"),
         48: ("benign_gridsybil/T48_benign", "task/autoencoder_benign_t48", "on-disk val"),
         32: ("all/T32_benign (8 folders)", "task/autoencoder_benign_t32", "on-disk val")}
for t in W:
    d, sib, val = names[t]
    rows += (f"<tr><td>T = {t}</td><td><code>{d}</code><br><span style='color:var(--muted);font-size:15px'>as {sib}</span></td>"
             f"<td class='n'>{n0(c[t]['train_windows'])}<br><span style='color:var(--muted)'>{n0(c[t]['train_vehicles'])} veh.</span></td>"
             f"<td class='n'>{n0(c[t]['val_windows'])}<br><span style='color:var(--muted)'>{n0(c[t]['val_vehicles'])} veh.</span></td>"
             f"<td class='n'>{n0(c[t]['test']['benign_test'])}</td><td>{att_cell(t)}</td>"
             f"<td class='n'>{c[t]['test_vehicles']}</td></tr>")
slide("Setup", f"""<h2>Data: the same windows as the one-class ConvAE cells</h2>
<table class="cnt">
<tr><th>Window</th><th>Input folder</th><th class="n">Train (benign)</th><th class="n">Val (benign)</th><th class="n">Test benign</th><th>Test attack</th><th class="n">Test veh.</th></tr>
{rows}
</table>
<ul class="cap2" style="columns: 2; column-gap: 40px; margin-top: 16px">
<li>Split by vehicle: 0 test vehicles in train or val (all T).</li>
<li>Thresholds = benign <b>val</b> p95 / p99.</li>
<li>All scores are saved <b>before</b> any label is read.</li>
<li>Labels were used only to choose the benign training data. So this is <b>one-class</b>, not unsupervised.</li>
<li>T = 32 test differs: other vehicles, benign from all 8 folders, GridSybil share {pct(c[32]['test']['GridSybil'] / (c[32]['test']['GridSybil'] + c[32]['test']['benign_test']))} (T = 48: {pct(c[48]['test']['GridSybil'] / (c[48]['test']['GridSybil'] + c[48]['test']['benign_test']))}).</li>
</ul>
<p class="foot">Counts from each run's <code>summary.json</code> and <code>test_metrics.json</code> (equal for all four seeds). At T = 64 the test split is mixed; its label file is read once after scoring.</p>""",
      "No new data. Each window size uses exactly the windows and splits of the one-class ConvAE at that size, so the numbers sit side by side. Train and val are benign only. Note: the labels picked the benign windows, so I call it one-class, not unsupervised. T = 32 has a different test set from T = 48 and T = 64.")

# ================================================================ 7 model size + training
mx = 140000
o = [f'<svg width="100%" viewBox="0 0 560 330" role="img" aria-label="Parameters per window size: masked AE TabMlp, TabResnet and one-class ConvAE">']
x0, y0, h = 70, 20, 250
for v in (0, 50000, 100000):
    y = y0 + h - v / mx * h
    o.append(f'<line x1="{x0}" y1="{y:.1f}" x2="550" y2="{y:.1f}" class="gl"/>')
    txt(o, x0 - 8, y + 5, f"{v // 1000}k", 14, "end", "mut")
for i, t in enumerate(W):
    gx = x0 + 30 + i * 160
    for j, (v, cls) in enumerate(((TR[t]["params"], "c-mae"), (TRR[t]["params"], "c-res"), (CONV["params"][f"T{t}"], "c-oc"))):
        bh = v / mx * h
        o.append(f'<rect x="{gx + j * 44}" y="{y0 + h - bh:.1f}" width="40" height="{bh:.1f}" class="{cls}"/>')
        txt(o, gx + j * 44 + 20, y0 + h - bh - 8, f"{v / 1000:.0f}k", 13.5, "middle", "v")
    txt(o, gx + 64, y0 + h + 24, f"T = {t}", 16, "middle")
o.append("</svg>")
slide("Setup", f"""<h2>Model size and training setup</h2>
<div class="cols" style="gap: 30px">
<div style="flex: 0 0 560px">
<div class="legend"><span><i class="sw mae"></i>masked AE TabMlp</span><span><i class="sw res"></i>masked AE TabResnet</span><span><i class="sw oc"></i>one-class ConvAE</span></div>
{''.join(o)}
<p class="cap">TabMlp {n0(TR[64]['params'])} / {n0(TR[48]['params'])} / {n0(TR[32]['params'])}, TabResnet {n0(TRR[64]['params'])} / {n0(TRR[48]['params'])} / {n0(TRR[32]['params'])} parameters (T64 / T48 / T32), mostly in the first and last Linear (they grow with T). z = 16 in all models.</p>
</div>
<div>
<table class="cnt">
<tr><th>Setting</th><th>Value</th></tr>
<tr><td>Optimizer</td><td>AdamW, lr 1e-3, weight decay 0.01</td></tr>
<tr><td>Batch</td><td>256</td></tr>
<tr><td>Masked share</td><td>0.2 of real cells per step</td></tr>
<tr><td>LR schedule</td><td>ReduceLROnPlateau, factor 0.5</td></tr>
<tr><td>Early stopping</td><td>val masked MSE; patience 5 (T64), 10 (T48, T32)</td></tr>
<tr><td>Max epochs</td><td>300 (never reached)</td></tr>
<tr><td>Hardware</td><td>Mac, MPS, torch {env0['torch']}</td></tr>
<tr><td>Run time</td><td>TabMlp {min_mlp}–{max_mlp}, TabResnet {min_res}–{max_res} min per run</td></tr>
<tr><td>Best epochs</td><td>TabMlp {min(TR[64]['best_epochs'])}–{max(TR[64]['best_epochs'])} / {min(TR[48]['best_epochs'])}–{max(TR[48]['best_epochs'])} / {min(TR[32]['best_epochs'])}–{max(TR[32]['best_epochs'])}<br>TabResnet {min(TRR[64]['best_epochs'])}–{max(TRR[64]['best_epochs'])} / {min(TRR[48]['best_epochs'])}–{max(TRR[48]['best_epochs'])} / {min(TRR[32]['best_epochs'])}–{max(TRR[32]['best_epochs'])} (T64 / T48 / T32)</td></tr>
</table>
</div>
</div>
<p class="foot">Sources: <code>config.json</code>, <code>env.json</code>, <code>summary.json</code> of the 24 runs; ConvAE parameters from <code>baselinepaper/01-our-studies/inventory.md</code>. Patience and plateau settings copied from each ConvAE sibling.</p>""",
      f"The masked AE comes in two encoders: a plain MLP (TabMlp) and a residual MLP (TabResnet), both small. At T = 64 and 48 they have more parameters than the ConvAE, at T = 32 TabMlp has fewer. Almost all weights sit in the first and the last layer. TabMlp trains in under six minutes per run, TabResnet in up to {max_res:.0f} minutes. All 24 runs stopped by early stopping.")

# ================================================================ 8 loss curves
pw, ph, gxs, gys = 172, 112, 70, 26
o = [f'<svg width="100%" viewBox="0 0 800 470" role="img" aria-label="Loss curves of all 12 runs: training loss, '
     f'val masked MSE, val full MSE, best epoch">']
for i, t in enumerate(W):
    vals = [v for s in SEEDS for hh in PER[f"T{t}-seed{s}"]["history"]
            for v in (hh["train_loss"], hh["val_masked_mse"], hh["val_full_mse"])]
    lo, hi = min(vals), max(vals)
    lo, hi = lo - 0.03 * (hi - lo), hi + 0.03 * (hi - lo)
    for j, s in enumerate(SEEDS):
        r = PER[f"T{t}-seed{s}"]
        hist = r["history"]
        x0 = gxs + j * (pw + 12)
        y0 = gys + i * (ph + 40)
        ne = r["epochs_run"]
        o.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" class="frame"/>')
        X = lambda e: x0 + (e - 1) / max(ne - 1, 1) * pw
        Y = lambda v: y0 + ph - (v - lo) / (hi - lo) * ph
        bx = X(r["best_epoch"])
        o.append(f'<line x1="{bx:.1f}" y1="{y0}" x2="{bx:.1f}" y2="{y0 + ph}" class="dot2"/>')
        for key, cls, dash in (("train_loss", "ln0", ""), ("val_masked_mse", "ln1", ""), ("val_full_mse", "l-full", "4 3")):
            pts = " ".join(f"{X(hh['epoch']):.1f},{Y(hh[key]):.1f}" for hh in hist)
            o.append(f'<polyline points="{pts}" class="crv {cls}" style="stroke-width:1.8{";stroke-dasharray:" + dash if dash else ""}"/>')
        txt(o, x0 + pw / 2, y0 - 7, f"T{t} · seed {s} · best {r['best_epoch']} / {ne}", 12.5, "middle")
        if j == 0:
            for v in (lo + 0.05 * (hi - lo), (lo + hi) / 2, hi - 0.05 * (hi - lo)):
                txt(o, x0 - 6, Y(v) + 4, f"{v:.2f}", 12, "end", "mut")
        if i == 2:
            txt(o, x0 + pw / 2, y0 + ph + 16, "epoch", 12, "middle", "mut")
o.append("</svg>")
slide("Setup", f"""<h2>Loss curves of the 12 TabMlp runs</h2>
<div class="cols" style="gap: 18px">
<div style="flex: 0 0 800px">
<div class="legend"><span><svg width="26" height="10"><line x1="0" y1="5" x2="26" y2="5" class="crv ln0"/></svg>train loss (widedeep loss)</span><span><svg width="26" height="10"><line x1="0" y1="5" x2="26" y2="5" class="crv ln1"/></svg>val masked MSE</span><span><svg width="26" height="10"><line x1="0" y1="5" x2="26" y2="5" class="crv l-full" style="stroke-dasharray:4 3"/></svg>val full MSE</span><span>dotted = best epoch</span></div>
{''.join(o)}
</div>
<div style="padding-top: 30px">
<span class="lab">Why train is above val</span>
<ul class="cap2">
<li>They are <b>different quantities</b>, so the gap is not over- or under-fitting.</li>
<li>Train: hidden cells only, divided by the feature variance, dropout on.</li>
<li>Val: plain MSE on z-scored features, 5 fixed masks, dropout off.</li>
<li>Train / val at best epoch: {ratio_lo:.2f}–{ratio_hi:.2f}.</li>
<li>{pct(dec_lo, 0)}–{pct(dec_hi, 0)} of the val drop is reached by epoch 10.</li>
</ul>
</div>
</div>
<p class="foot">From each run's <code>history.jsonl</code> and <code>summary.json</code>; y axis shared per row. A PNG per run (with the LR) is in <code>runs/{{mlp,resnet}}/&lt;run&gt;/loss_curve.png</code> (TabResnet curves there).</p>""",
      "Each small panel is one run. Blue is the training loss, orange the validation loss. Blue sits above orange in every run, but this is by construction: training uses widedeep's variance-normalised loss with dropout on, validation uses plain mean squared error. Most of the learning happens in the first ten epochs. Seeds look almost the same.")

# ================================================================ 9 scoring
o = [f'<svg width="100%" viewBox="0 0 470 300" role="img" aria-label="Five fixed mask groups partition the cells; each real cell is hidden exactly once">']
import random
rng = random.Random(7)
cells = [(r, cc) for r in range(10) for cc in range(13)]
order = list(range(len(cells)))
rng.shuffle(order)
grp = {cells[k]: i % 5 for i, k in enumerate(order)}
for (r, cc), g in grp.items():
    cls = "pad" if r >= 8 else f"g{g}"
    o.append(f'<rect x="{20 + cc * 22}" y="{30 + r * 22}" width="20" height="20" class="{cls}"/>')
txt(o, 20 + 13 * 11, 20, "illustration: cells coloured by mask group", 14, "middle", "mut")
for g in range(5):
    o.append(f'<rect x="330" y="{40 + g * 30}" width="18" height="18" class="g{g}"/>')
    txt(o, 356, 54 + g * 30, f"group {g + 1}", 14, cls="mut")
o.append('<rect x="330" y="195" width="18" height="18" class="pad"/>')
txt(o, 356, 209, "padding", 14, cls="mut")
txt(o, 330, 245, "draw k hides group k", 14, cls="mut")
o.append("</svg>")
slide("Setup", f"""<h2>Three window scores, all on real cells only</h2>
<div class="cols" style="gap: 28px">
<div style="flex: 0 0 470px">{''.join(o)}
<p class="cap">The real partition uses a fixed seed (20261117), the same for every window, run and seed.</p></div>
<div class="stack">
<div class="card"><h3>masked_recon (main score)</h3><p>5 fixed masks split the cells into 5 groups of 20%. Each real cell is hidden once. Score = mean squared error of cells rebuilt while hidden.</p></div>
<div class="card"><h3>full_recon</h3><p>Nothing hidden (widedeep eval mode). Mean squared error over real cells.</p></div>
<div class="card"><h3>mahalanobis</h3><p>Distance of z (clean input) to a Gaussian fitted on benign train z.</p></div>
</div>
</div>
<p class="foot">Thresholds: benign val p95 / p99 per score. Scores of val and every test split saved to <code>scores/&lt;split&gt;.npz</code> before any label read. Source: <code>scoring.py</code>.</p>""",
      "The main score uses five fixed masks. Together they cover every cell once, so each real cell is rebuilt while hidden, like in training. The score does not depend on chance. I also report plain reconstruction and a Mahalanobis distance in the latent space.")

# ================================================================ 10 GridSybil results
o = [f'<svg width="100%" viewBox="0 0 690 420" role="img" aria-label="GridSybil AUROC per window size: masked recon, '
     f'full recon, Mahalanobis, ConvAE recon, with seed dots and the length-only level">']
x0, y0, h, ymin, ymax = 60, 20, 340, 0.3, 0.8
Y = lambda v: y0 + h - (v - ymin) / (ymax - ymin) * h
for v in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
    o.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="685" y2="{Y(v):.1f}" class="{"chance" if v == 0.5 else "gl"}"/>')
    txt(o, x0 - 8, Y(v) + 5, f"{v:.1f}", 14, "end", "mut")
bars = (("masked_recon", "c-mae"), ("res_masked", "c-res"), ("full_recon", "c-full"), ("mahalanobis", "c-maha"),
        ("convae", "c-oc"))
bw = 32
for i, t in enumerate(W):
    gx = x0 + 18 + i * 205
    for j, (k, cls) in enumerate(bars):
        d = (CONV[f"T{t} GridSybil"]["recon"] if k == "convae" else gr(t)["masked_recon"] if k == "res_masked"
             else gs(t)[k])
        bx = gx + j * (bw + 4)
        o.append(f'<rect x="{bx}" y="{Y(d["mean"]):.1f}" width="{bw}" height="{Y(ymin) - Y(d["mean"]):.1f}" class="{cls}"/>')
        for q, v in enumerate(d["values"]):
            o.append(f'<circle cx="{bx + 5 + q * 7.3:.1f}" cy="{Y(v):.1f}" r="3" class="sdx"/>')
        txt(o, bx + bw / 2, Y(d["mean"]) - 9, f"{d['mean']:.2f}", 12.5, "middle", "v")
    L = gs(t)["length_only"]
    o.append(f'<line x1="{gx - 6}" y1="{Y(L):.1f}" x2="{gx + 5 * (bw + 4)}" y2="{Y(L):.1f}" class="len"/>')
    txt(o, gx + 2.5 * (bw + 4) - 2, Y(ymin) + 24, f"T = {t}", 16, "middle")
o.append("</svg>")
rows = ""
for lab, k in (("masked_recon", "masked_recon"), ("full_recon", "full_recon"), ("mahalanobis", "mahalanobis")):
    for enc, G in (("Mlp", gs), ("Resnet", gr)):
        rows += (f"<tr{' class=hlrow' if k == 'masked_recon' else ''}><td>Tab{enc} {lab}</td>"
                 + "".join(f"<td class='n'>{ms(G(t)[k])}</td>" for t in W) + "</tr>")
rows += "<tr><td>ConvAE recon</td>" + "".join(f"<td class='n'>{ms(CONV[f'T{t} GridSybil']['recon'])}</td>" for t in W) + "</tr>"
rows += "<tr class='refrow'><td>length only</td>" + "".join(f"<td class='n'>{f3(gs(t)['length_only'])}</td>" for t in W) + "</tr>"
rows += "<tr><td>recall @ val p95</td>" + "".join(f"<td class='n'>{f3(gs(t)['recall_p95_masked']['mean'])} / {f3(gr(t)['recall_p95_masked']['mean'])}</td>" for t in W) + "</tr>"
rows += "<tr><td>benign FPR @ p95</td>" + "".join(f"<td class='n'>{f3(gs(t)['fpr_p95_masked']['mean'])} / {f3(gr(t)['fpr_p95_masked']['mean'])}</td>" for t in W) + "</tr>"
slide("Results", f"""<h2>GridSybil: TabResnet beats TabMlp, both below the ConvAE</h2>
<div class="cols" style="gap: 22px">
<div style="flex: 0 0 600px">
<div class="legend"><span><i class="sw mae"></i>TabMlp masked</span><span><i class="sw res"></i>TabResnet masked</span><span><i class="sw full"></i>TabMlp full</span><span><i class="sw mah"></i>TabMlp Mahalanobis</span><span><i class="sw oc"></i>ConvAE recon</span><span>dots = seeds · dashed = length only</span></div>
{''.join(o)}
</div>
<div>
<table class="res" style="font-size:14.5px">
<tr><th>Window AUROC</th><th class="n">T = 64</th><th class="n">T = 48</th><th class="n">T = 32</th></tr>
{rows}
</table>
<p class="cap" style="margin-top:8px">Mean ± sd over seeds 0–3. Recall / FPR: masked_recon at the benign val p95 threshold, TabMlp / TabResnet.</p>
</div>
</div>
<p class="foot">Masked AE: <code>test_metrics.json</code> of each run (recomputed from score files, |Δ| ≤ 1e-6). ConvAE: <code>results.json</code> seed_summary (T32 seed 0 = rounded notebook value).</p>""",
      f"This is the main result. The masked score improves from T64 to T32. TabResnet is better than TabMlp at every window size, by {res_gain[64]:.3f}, {res_gain[48]:.3f} and {res_gain[32]:.3f}. But both stay below the ConvAE: TabMlp by {gap[64]:.2f}, {gap[48]:.2f}, {gap[32]:.2f}, TabResnet by {gap_r[64]:.2f}, {gap_r[48]:.2f}, {gap_r[32]:.2f}. At T64 TabMlp is at the length-only level. Mahalanobis is below chance for both encoders. The seed dots sit on top of each other, so the result is stable.")

# ================================================================ 11 ROC
o = [f'<svg width="100%" viewBox="0 0 1160 410" role="img" aria-label="ROC curves GridSybil vs benign per window size, 4 seeds">']
for i, t in enumerate(W):
    curves = []
    for s in SEEDS:
        rc = PER[f"T{t}-seed{s}"]["attacks"]["GridSybil"]["roc"]
        curves += [(rc["mahalanobis"], "l-maha", "stroke-width:1.4"), (rc["full_recon"], "l-full", "stroke-width:1.4"),
                   (rc["masked_recon"], "l-mae", "stroke-width:1.6"),
                   (PERR[f"T{t}-seed{s}"]["attacks"]["GridSybil"]["roc"]["masked_recon"], "l-res", "stroke-width:1.6")]
    curves.append((PER[f"T{t}-seed0"]["attacks"]["GridSybil"]["roc"]["length_only"], "l-len", "stroke-width:1.8"))
    roc_panel(o, 60 + i * 380, 30, 320, curves, f"T = {t} (4 seeds)")
    txt(o, 60 + i * 380 + 160, 392, "benign test FPR", 13, "middle", "mut")
txt(o, 18, 190, "GridSybil recall", 13, "middle", "mut")
o[-1] = o[-1].replace("<text ", '<text transform="rotate(-90 18 190)" ', 1)
o.append("</svg>")
slide("Results", f"""<h2>ROC curves: close to the diagonal, seeds overlap</h2>
<div class="legend"><span><i class="sw mae"></i>TabMlp masked_recon</span><span><i class="sw res"></i>TabResnet masked_recon</span><span><i class="sw full"></i>TabMlp full_recon</span><span><i class="sw mah"></i>TabMlp mahalanobis</span><span>dashed = length only (−rows)</span><span>dotted diagonal = chance</span></div>
{''.join(o)}
<p class="foot">Recomputed from <code>scores/*.npz</code> of every run (T64 labels read once through the package's source object); TPR interpolated on a fixed FPR grid. Mean masked AUROC TabMlp {f3(gs(64)['masked_recon']['mean'])} / {f3(gs(48)['masked_recon']['mean'])} / {f3(gs(32)['masked_recon']['mean'])}, TabResnet {f3(gr(64)['masked_recon']['mean'])} / {f3(gr(48)['masked_recon']['mean'])} / {f3(gr(32)['masked_recon']['mean'])}.</p>""",
      "These are the ROC curves from the saved scores. Four seeds per score, and they lie on top of each other. The pink TabResnet curves sit a little above the teal TabMlp ones. The reconstruction curves are only a little above the diagonal. Mahalanobis is below it: attack windows get a smaller distance than benign windows.")

# ================================================================ 12 T32 four attacks
rows = ""
for a in ATT:
    x = S[f"T32 {a}"]
    cv = CONV[f"T32 {a}"]["recon"]
    xr = SR[f"T32 {a}"]
    rows += (f"<tr><td>{a}</td><td class='n'>{ms(x['masked_recon'])}</td><td class='n'>{ms(xr['masked_recon'])}</td>"
             f"<td class='n'>{ms(x['full_recon'])}</td>"
             f"<td class='n'>{ms(x['mahalanobis'])}</td><td class='n'>{f3(x['length_only'])}</td><td class='n'>{ms(cv)}</td>"
             f"<td class='n'>{pct(x['share_rows_1_3']['attack'])}</td></tr>")
rows += (f"<tr class='refrow'><td>benign test</td><td></td><td></td><td></td><td></td><td></td><td></td>"
         f"<td class='n'>{pct(S['T32 GridSybil']['share_rows_1_3']['benign'])}</td></tr>")
o = [f'<svg width="100%" viewBox="0 0 1160 240" role="img" aria-label="ROC curves T32 seed 0 for the four attacks">']
for i, a in enumerate(ATT):
    rc = PER["T32-seed0"]["attacks"][a]["roc"]
    curves = [(rc["mahalanobis"], "l-maha", "stroke-width:1.6"), (rc["full_recon"], "l-full", "stroke-width:1.6"),
              (rc["masked_recon"], "l-mae", "stroke-width:2"),
              (PERR["T32-seed0"]["attacks"][a]["roc"]["masked_recon"], "l-res", "stroke-width:2"),
              (rc["length_only"], "l-len", "stroke-width:1.8")]
    roc_panel(o, 80 + i * 280, 26, 185, curves, f"{a} (seed 0)", 14)
o.append("</svg>")
slide("Results", f"""<h2>T = 32, four attacks: high numbers, but look at length only</h2>
<table class="res" style="font-size:16.5px">
<tr><th>Attack vs benign</th><th class="n">TabMlp masked</th><th class="n">TabResnet masked</th><th class="n">TabMlp full</th><th class="n">TabMlp Mahalanobis</th><th class="n">length only</th><th class="n">ConvAE recon</th><th class="n">1–3-row windows</th></tr>
{rows}
</table>
<div class="legend" style="margin-top:8px"><span><i class="sw mae"></i>TabMlp masked</span><span><i class="sw res"></i>TabResnet masked</span><span><i class="sw full"></i>TabMlp full</span><span><i class="sw mah"></i>TabMlp Mahalanobis</span><span>dashed = length only</span></div>
{''.join(o)}
<p class="foot">Test window AUROC, mean ± sd over seeds 0–3; ROC = seed 0. TabResnet Mahalanobis: GridSybil {ms(SR['T32 GridSybil']['mahalanobis'])}, DoSRandom {ms(SR['T32 DoSRandom']['mahalanobis'])} (unstable). DataReplay and DoS windows are mostly 1–3 rows (pseudonyms change every 1–2 messages, F11).</p>""",
      "At T = 32 we also have the three other attacks. Both encoders give almost the same numbers here. DoSRandom looks very high, and DataReplay and DoSDisruptive look better than the ConvAE. But length alone already gives 0.95 to 0.97, because these windows are almost all very short. The next slides show that the masked score is partly a length score.")

# ================================================================ 13 length confound
br = PER["T32-seed0"]["attacks"]["GridSybil"]["benign_by_rows"]
rws = sorted(int(k) for k in br)
o = [f'<svg width="100%" viewBox="0 0 620 380" role="img" aria-label="Median benign test score by number of real rows, T32 seed 0">']
x0, y0, pw2, h = 70, 20, 520, 300
mx = max(max(br[str(r)]["masked_recon"], br[str(r)]["full_recon"]) for r in rws) * 1.1
X = lambda r: x0 + (r - 1) / (max(rws) - 1) * pw2
Y = lambda v: y0 + h - v / mx * h
for v in (0, 0.2, 0.4, 0.6):
    if v <= mx:
        o.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x0 + pw2}" y2="{Y(v):.1f}" class="gl"/>')
        txt(o, x0 - 8, Y(v) + 5, f"{v:.1f}", 14, "end", "mut")
for k, cls, pc in (("full_recon", "l-full", "ptf"), ("masked_recon", "l-mae", "ptm")):
    pts = " ".join(f"{X(r):.1f},{Y(br[str(r)][k]):.1f}" for r in rws)
    o.append(f'<polyline points="{pts}" class="crv {cls}"/>')
    for r in rws:
        o.append(f'<circle cx="{X(r):.1f}" cy="{Y(br[str(r)][k]):.1f}" r="3.3" class="{pc}"/>')
for r in (1, 4, 8, 16, 24, 32):
    txt(o, X(r), y0 + h + 20, str(r), 14, "middle", "mut")
txt(o, x0 + pw2 / 2, y0 + h + 42, "real rows in the window (benign test, T = 32, TabMlp seed 0)", 14, "middle", "mut")
txt(o, X(1) + 10, Y(br["1"]["masked_recon"]) + 4, f"1 row: {br['1']['masked_recon']:.3f}", 14, cls="v")
o.append("</svg>")
bsl = {t: gs(t)["benign_short_vs_long"] for t in W}
bslr = {t: gr(t)["benign_short_vs_long"] for t in W}
slide("Checks", f"""<h2>The masked score partly measures window length</h2>
<div class="cols" style="gap: 26px">
<div style="flex: 0 0 600px">
<div class="legend"><span><i class="sw mae"></i>median masked_recon</span><span><i class="sw full"></i>median full_recon</span></div>
{''.join(o)}
</div>
<div class="stack" style="gap: 12px">
<div class="card"><span class="lab">Benign only: 1–3 rows vs longer, AUROC</span><p style="font-size:19px"><b style="color:var(--text)">TabMlp masked {f3(bsl[64]['masked_recon']['mean'])} / {f3(bsl[48]['masked_recon']['mean'])} / {f3(bsl[32]['masked_recon']['mean'])}</b><br>TabMlp full {f3(bsl[64]['full_recon']['mean'])} / {f3(bsl[48]['full_recon']['mean'])} / {f3(bsl[32]['full_recon']['mean'])}<br><b style="color:var(--text)">TabResnet masked {f3(bslr[64]['masked_recon']['mean'])} / {f3(bslr[48]['masked_recon']['mean'])} / {f3(bslr[32]['masked_recon']['mean'])}</b> (T64 / T48 / T32)</p></div>
<div class="card"><span class="lab warm">1-row windows, T = 32</span><p style="font-size:20px"><b style="color:var(--text)">DataReplay {pct(dr['one_row_share_seed0']['attack'])}</b> vs benign {pct(dr['one_row_share_seed0']['benign'])}</p></div>
<div class="card"><span class="lab">Why</span><p style="font-size:19px">A 1-row window has no time neighbours. A hidden cell must be guessed from the other 12 features of the same message only.</p></div>
</div>
</div>
<p class="foot">No attack window is used in the left plot or in the first box. Short-vs-long AUROC: mean over seeds 0–3 from the score files.</p>""",
      f"Here I use only benign test windows. One-row windows get a much higher masked error than long windows. So the masked score alone can tell short from long benign windows with AUROC about 0.7. Full reconstruction does this much less. This matters because 70% of DataReplay windows have one row, against 5% of benign windows.")

# ================================================================ 14 pooled vs length-matched
cases = [("T64", "GridSybil"), ("T48", "GridSybil"), ("T32", "GridSybil"), ("T32", "DataReplay"), ("T32", "DoSRandom"),
         ("T32", "DoSDisruptive")]
o = [f'<svg width="100%" viewBox="0 0 700 410" role="img" aria-label="Pooled vs length-matched masked_recon AUROC per case, with length-only">']
x0, xr, y0, rh = 200, 680, 30, 56
lo, hi = 0.4, 1.0
X = lambda v: x0 + (v - lo) / (hi - lo) * (xr - x0)
for v in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    o.append(f'<line x1="{X(v):.1f}" y1="{y0 - 10}" x2="{X(v):.1f}" y2="{y0 + 6 * rh}" class="{"chance" if v == 0.5 else "gl"}"/>')
    txt(o, X(v), y0 + 6 * rh + 22, f"{v:.1f}", 14, "middle", "mut")
for i, (t, a) in enumerate(cases):
    x = S[f"{t} {a}"]
    y = y0 + i * rh + rh / 2
    L = x["length_only"]
    txt(o, x0 - 14, y + 5, f"{t} {a}", 16, "end")
    for xx, yy, col, cls in ((x, y - 10, "var(--mae)", "c-mae"), (SR[f"{t} {a}"], y + 10, "var(--res)", "c-res")):
        p, m = xx["masked_recon"]["mean"], xx["masked_recon_length_matched"]["mean"]
        o.append(f'<line x1="{X(p):.1f}" y1="{yy:.1f}" x2="{X(m):.1f}" y2="{yy:.1f}" class="{cls.replace("c-", "l-")}" style="stroke-width:3"/>')
        o.append(f'<circle cx="{X(p):.1f}" cy="{yy:.1f}" r="6" class="ring" style="stroke:{col}"/>')
        o.append(f'<circle cx="{X(m):.1f}" cy="{yy:.1f}" r="6" class="{cls}"/>')
        lx, anc = (X(min(m, p)) - 11, "end") if m < p else (X(max(m, p)) + 11, "start")
        txt(o, lx, yy + 5, f3(m), 13, anc, "v")
    o.append(f'<rect x="{X(L) - 6:.1f}" y="{y - 6:.1f}" width="12" height="12" transform="rotate(45 {X(L):.1f} {y:.1f})" class="lenmk"/>')
o.append("</svg>")
rows = ""
for t, a in cases:
    x = S[f"{t} {a}"]
    xr = SR[f"{t} {a}"]
    rows += (f"<tr><td>{t} {a}</td><td class='n'>{f3(x['masked_recon']['mean'])} → {f3(x['masked_recon_length_matched']['mean'])}</td>"
             f"<td class='n'>{f3(xr['masked_recon']['mean'])} → {f3(xr['masked_recon_length_matched']['mean'])}</td>"
             f"<td class='n'>{f3(x['full_recon']['mean'])} → {f3(x['full_recon_length_matched']['mean'])}</td></tr>")
slide("Checks", f"""<h2>Length-matched AUROC: DataReplay falls to about chance</h2>
<div class="cols" style="gap: 20px">
<div style="flex: 0 0 600px">
<div class="legend"><span><svg width="16" height="16"><circle cx="8" cy="8" r="6" class="ring" style="stroke:var(--text)"/></svg> pooled</span><span><svg width="16" height="16"><circle cx="8" cy="8" r="6" class="c-mae"/></svg> TabMlp</span><span><svg width="16" height="16"><circle cx="8" cy="8" r="6" class="c-res"/></svg> TabResnet</span><span>filled = length-matched masked_recon</span><span><svg width="16" height="16"><rect x="3" y="3" width="10" height="10" transform="rotate(45 8 8)" class="lenmk"/></svg> length only</span></div>
{''.join(o)}
</div>
<div>
<table class="res" style="font-size:14.5px">
<tr><th>Case</th><th class="n">TabMlp masked</th><th class="n">TabResnet masked</th><th class="n">TabMlp full</th></tr>
{rows}
</table>
<p class="cap" style="margin-top:10px">Pooled → length-matched, mean over seeds 0–3.</p>
</div>
</div>
<p class="foot">Length-matched = AUROC inside each exact window length (1, 2, …, T real rows), weighted by the attack windows at that length; covers 100% of attack windows in every case. Code: notebook cell 43, rerun on the score files.</p>""",
      f"Length-matched means: compare attack and benign windows only when they have the same number of rows, then average. DataReplay falls from {f3(dr['masked_recon']['mean'])} to {f3(dr['masked_recon_length_matched']['mean'])} with TabMlp, so it is not detected at all, and from {f3(drr['masked_recon']['mean'])} to {f3(drr['masked_recon_length_matched']['mean'])} with TabResnet, barely above chance. TabResnet keeps more of its GridSybil score: {f3(gr(32)['masked_recon_length_matched']['mean'])} length-matched at T = 32. For TabMlp: DoSDisruptive falls to {f3(S['T32 DoSDisruptive']['masked_recon_length_matched']['mean'])}. DoSRandom stays high, {f3(S['T32 DoSRandom']['masked_recon_length_matched']['mean'])}. GridSybil drops a little at every T.")

# ================================================================ 15 comparison T32
items = [("ConvAE recon (4 seeds)", CONV["T32 GridSybil"]["recon"]["mean"], None, "c-oc"),
         ("Masked AE TabResnet masked (4 seeds)", gr(32)["masked_recon"]["mean"], gr(32)["masked_recon_length_matched"]["mean"], "c-res"),
         ("Masked AE TabMlp masked (4 seeds)", gs(32)["masked_recon"]["mean"], gs(32)["masked_recon_length_matched"]["mean"], "c-mae"),
         ("Masked AE TabMlp full (4 seeds)", gs(32)["full_recon"]["mean"], gs(32)["full_recon_length_matched"]["mean"], "c-full"),
         ("TimesNet kNN k5 (seed 0)", tn_p["GridSybil"]["knn_k5"]["auroc"], tn_p["GridSybil"]["knn_k5"]["length_matched"], "c-tn"),
         ("TimesNet Mahalanobis (seed 0)", tn_p["GridSybil"]["mahalanobis"]["auroc"], tn_p["GridSybil"]["mahalanobis"]["length_matched"], "c-tn"),
         ("TimesNet random-init kNN (seed 0)", tn_r["GridSybil"]["knn_k5"]["auroc"], tn_r["GridSybil"]["knn_k5"]["length_matched"], "c-maha")]
o = [f'<svg width="100%" viewBox="0 0 860 440" role="img" aria-label="T32 GridSybil AUROC: ConvAE, masked AE TabResnet and TabMlp, TimesNet, pooled and length-matched">']
x0, xr, y0, rh = 330, 820, 30, 54
lo, hi = 0.4, 1.0
X = lambda v: x0 + (v - lo) / (hi - lo) * (xr - x0)
for v in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    o.append(f'<line x1="{X(v):.1f}" y1="{y0 - 6}" x2="{X(v):.1f}" y2="{y0 + 7 * rh}" class="{"chance" if v == 0.5 else "gl"}"/>')
    txt(o, X(v), y0 + 7 * rh + 22, f"{v:.1f}", 14, "middle", "mut")
for i, (lab, p, m, cls) in enumerate(items):
    y = y0 + i * rh
    txt(o, x0 - 12, y + rh / 2 + 5, lab, 15.5, "end")
    o.append(f'<rect x="{x0}" y="{y + 8}" width="{X(p) - x0:.1f}" height="17" class="{cls}"/>')
    txt(o, X(p) + 6, y + 21, f"{f3(p)} pooled", 13.5, cls="v")
    if m is not None:
        o.append(f'<rect x="{x0}" y="{y + 28}" width="{X(m) - x0:.1f}" height="17" class="{cls}" style="opacity:0.5"/>')
        txt(o, X(m) + 6, y + 41, f"{f3(m)} matched", 13.5, cls="mut")
    else:
        txt(o, X(p) + 6, y + 41, "length-matched: not computed", 13, cls="mut")
L = gs(32)["length_only"]
o.append(f'<line x1="{X(L):.1f}" y1="{y0 - 6}" x2="{X(L):.1f}" y2="{y0 + 7 * rh}" class="len"/>')
txt(o, X(L) + 6, y0 + 4, f"length only {f3(L)}", 13, cls="mut")
o.append("</svg>")
slide("Checks", f"""<h2>Same T = 32 windows: TimesNet ≫ ConvAE > masked AE (TabResnet > TabMlp)</h2>
<div class="cols" style="gap: 18px">
<div style="flex: 0 0 860px">{''.join(o)}</div>
<div style="padding-top: 20px">
<ul class="cap2">
<li>Same benign train / val / test and GridSybil test windows for all three.</li>
<li>TimesNet: <b>one seed only</b>.</li>
<li>TimesNet gain over random-init survives length matching ({f3(tn_p['GridSybil']['knn_k5']['length_matched'])} vs {f3(tn_r['GridSybil']['knn_k5']['length_matched'])}).</li>
<li>Masked AE: lowest of the three; TabResnet is its better encoder ({f3(gr(32)['masked_recon']['mean'])} vs {f3(gs(32)['masked_recon']['mean'])}) and still {gap_r[32]:.3f} below the ConvAE.</li>
</ul>
</div>
</div>
<p class="foot">TimesNet: <code>src/runs/evaluation/all/T32_benign/model-t32b-seed0{{,-random-init}}/summary.json</code>, length-matched recomputed from its score files (same method). ConvAE: <code>results.json</code>.</p>""",
      "Now the three one-class models on the same T = 32 windows. TimesNet with kNN scoring is far ahead, and even an untrained TimesNet beats both autoencoders. The pretrained TimesNet stays above random-init after length matching. But TimesNet has only one seed so far. The masked autoencoder is the weakest, even with the better TabResnet encoder.")

# ================================================================ 16 findings
slide("Wrap-up", f"""<h2>Findings</h2>
<ol class="take">
<li>Masked reconstruction does <b>not</b> beat plain reconstruction. GridSybil AUROC TabMlp {f3(gs(64)['masked_recon']['mean'])} / {f3(gs(48)['masked_recon']['mean'])} / {f3(gs(32)['masked_recon']['mean'])}, TabResnet {f3(gr(64)['masked_recon']['mean'])} / {f3(gr(48)['masked_recon']['mean'])} / {f3(gr(32)['masked_recon']['mean'])} vs ConvAE {f3(CONV['T64 GridSybil']['recon']['mean'])} / {f3(CONV['T48 GridSybil']['recon']['mean'])} / {f3(CONV['T32 GridSybil']['recon']['mean'])} (T64 / T48 / T32).</li>
<li>The <b>TabResnet</b> encoder helps at every T (+{res_gain[64]:.3f} / +{res_gain[48]:.3f} / +{res_gain[32]:.3f}; length-matched {f3(gr(64)['masked_recon_length_matched']['mean'])} / {f3(gr(48)['masked_recon_length_matched']['mean'])} / {f3(gr(32)['masked_recon_length_matched']['mean'])}), but does not close the gap.</li>
<li>The masked score is partly a <b>length score</b>: benign short vs long AUROC ≈ {f3(min(bsl[t]['masked_recon']['mean'] for t in W))}–{f3(max(bsl[t]['masked_recon']['mean'] for t in W))} (TabMlp), {f3(min(bslr[t]['masked_recon']['mean'] for t in W))}–{f3(max(bslr[t]['masked_recon']['mean'] for t in W))} (TabResnet).</li>
<li>DataReplay {f3(dr['masked_recon']['mean'])} / {f3(drr['masked_recon']['mean'])} pooled is length, not detection: <b>{f3(dr['masked_recon_length_matched']['mean'])} / {f3(drr['masked_recon_length_matched']['mean'])}</b> length-matched (TabMlp / TabResnet), and length alone gives {f3(dr['length_only'])}.</li>
<li>Mahalanobis on z is below chance on GridSybil for both encoders (TabResnet DoSRandom {ms(SR['T32 DoSRandom']['mahalanobis'], 2)}, unstable).</li>
<li>Results are stable: sd ≤ {max(max(G(t)['masked_recon']['sd'] for t in W for G in (gs, gr)), max(X_[f'T32 {a}']['masked_recon']['sd'] for a in ATT for X_ in (S, SR))):.3f} for masked_recon over 4 seeds.</li>
</ol>
<p class="foot">One-class side experiment; all numbers from the 24 run folders (12 TabMlp + 12 TabResnet), reported as produced (RULE 1).</p>""",
      "Six points. The masked autoencoder is weaker than our ConvAE. A stronger encoder, TabResnet, helps a little but not enough. Its main score mixes in window length. The good-looking DataReplay number disappears when we match lengths. Mahalanobis does not work here. And the seeds agree, so this is not noise.")

# ================================================================ 17 caveats
slide("Wrap-up", f"""<h2>Caveats</h2>
<div class="cards3 ev" style="grid-template-columns: repeat(2, 1fr); gap: 14px">
<div class="card"><h3>Two tabular encoders only</h3><p>TabMlp and TabResnet ({n0(TRR[64]['params'])} / {n0(TRR[48]['params'])} / {n0(TRR[32]['params'])} params). Both flatten the window; no convolutional or attention encoder with the masked loss.</p></div>
<div class="card"><h3>Mac only</h3><p>All 24 runs on MPS, git {', '.join(sorted(commits))} (TabMlp) / {', '.join(sorted(commits_r))} (TabResnet), dirty tree (git_dirty = {list(dirty | dirty_r)[0]}); no EC2 repeat.</p></div>
<div class="card"><h3>Not literal widedeep</h3><p>Our encoder sees the masked input (deviation 1). widedeep 1.7.0's own TabMlp / TabResnet path was not run.</p></div>
<div class="card"><h3>Mahalanobis below chance</h3><p>{f3(min(G(t)['mahalanobis']['min'] for t in W for G in (gs, gr)))}–{f3(max(G(t)['mahalanobis']['max'] for t in W for G in (gs, gr)))} on GridSybil (both encoders); the least stable score over seeds.</p></div>
<div class="card"><h3>DoS ≥ 4 rows = 1.000</h3><p>Rests on only {S['T32 DoSRandom']['n_attack_rows_ge_4']} / {S['T32 DoSDisruptive']['n_attack_rows_ge_4']} attack windows (DoSRandom / DoSDisruptive).</p></div>
<div class="card"><h3>Test sets differ by T</h3><p>T = 32: {c[32]['test_vehicles']} test vehicles, benign from 8 folders. T = 48 / 64: {c[48]['test_vehicles']} vehicles, GridSybil is the majority. Compare columns with care.</p></div>
</div>
<p class="foot">Also: one-class means labels chose the benign training windows; ConvAE T32 seed 0 is a rounded value; TimesNet comparison is one seed.</p>""",
      "Some limits. Only two tabular encoder types. Only the Mac. We changed one key detail of widedeep on purpose. Mahalanobis fails. The perfect DoS numbers on long windows rest on about five hundred windows. And the T = 32 test set is not the same as at T = 48 and 64.")

# ================================================================ 18 next steps
slide("Wrap-up", """<h2>Next steps (suggestions)</h2>
<ol class="plan">
<li><b>Quote TabResnet</b> as the masked-AE row in the paper tables (the stronger encoder), TabMlp as the smaller variant.</li>
<li><b>Literal widedeep run</b> (<code>feed_masked_input = False</code>): measure what deviation 1 does.</li>
<li><b>Length-matched AUROC everywhere</b>: report it next to the pooled AUROC in every baseline and in the paper tables.</li>
<li>Keep the masked AE as a <b>negative baseline</b>: masking alone does not help on these windows.</li>
</ol>
<p class="foot">Suggestions only; nothing here is run yet. Question for you: is the masked AE worth more runs, or should it stay a short negative result?</p>""",
      "These are suggestions, not done work. TabResnet is now done, so it should be the masked-AE number we quote. A literal widedeep run would close the last open method question. The most useful change for the paper is to report length-matched AUROC everywhere. I would like your view on whether this baseline deserves more time.")

# ---------------------------------------------------------------- assemble
js = BASE["js"].replace("baseline-seeds-deck-theme", "masked-ae-deck-theme")
html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Masked Autoencoder Baseline</title>
<style>{BASE['css']}
{CSS_EXTRA}</style>
</head>
<body>
{BASE['defs']}
<div id="ui">
  <button id="prev" aria-label="Previous slide">&larr;</button>
  <button id="next" aria-label="Next slide">&rarr;</button>
  <button id="notes" aria-label="Toggle speaker notes (N)" aria-pressed="false">Notes</button>
  <button id="theme" aria-label="Toggle light or dark theme">Dark</button>
</div>
<div id="viewport"><div id="stage">
{chr(10).join(SLIDES)}

<div id="counter"></div>
<div id="progress"></div>
</div></div>
<script>{js}</script>
</body>
</html>
"""
OUT.write_text(html)
print(OUT, len(SLIDES), "slides", len(html), "bytes")
