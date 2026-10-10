"""Build ``model-all-report.html``: one self-contained findings page on the ``model-all`` pretraining run.

The page is for the professor: what was built, how pretraining went, what the frozen encoder does on the test
windows of group 1416, what is wrong and what to fix next. It is generated from files other tools wrote and adds no
numbers of its own (RULE 1); every displayed number is rounded for reading and carries its exact value in a tooltip.

Inputs (all read-only; every one except the training summary is optional and rendered as "missing" when absent):

* ``src/runs/evaluation/all/T24/model-all/training_summary.json`` (``scripts/summarize_pretraining.py``);
* ``simulation/public/data/detection/manifest.json`` (``npm run sim:detect``: label-free kNN anomaly score);
* ``docs/reports/model-all-findings.json`` (the written diagnosis: summary, issues, fixes, direction, limitations);
* ``src/runs/evaluation/all/T24/model-all/{report_data,config}.json`` (``npm run eval:all``: frozen probes). They are
  shown only when ``config.json`` records the full evaluation (≥ 3 seeds, ≥ 20,000 fit windows per class, ≥ 1,000
  bootstrap resamples); a reduced (smoke) run is reported as "not the full evaluation" without any of its numbers;
* ``src/data/encoder_input/all/T24/metadata.json`` (window counts per class, for the data section).

The output is a single HTML file with inline CSS and inline SVG: no scripts, no external fonts, images or data.

Usage:
    python scripts/build_model_report.py
    python scripts/build_model_report.py --out /tmp/report.html --eval-dir <dir>
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import html
import json
import math
import pathlib
import sys
from collections.abc import Callable, Sequence
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_DIR = "src/runs/evaluation/all/T24/model-all"

SECTION_IDS: tuple[str, ...] = (
    "summary",
    "pipeline",
    "data",
    "training",
    "test",
    "diagnosis",
    "fixes",
    "limitations",
    "reproducibility",
)
CLASSES: tuple[str, ...] = ("Benign", "GridSybil", "DataReplaySybil", "DoSRandomSybil", "DoSDisruptiveSybil")
FAMILIES: tuple[str, ...] = CLASSES[1:]
SHORT: dict[str, str] = {
    "Benign": "Benign",
    "GridSybil": "GridSybil",
    "DataReplaySybil": "DataReplay",
    "DoSRandomSybil": "DoSRandom",
    "DoSDisruptiveSybil": "DoSDisruptive",
}
REPRESENTATIONS: tuple[str, ...] = ("model-all", "random-init", "length", "stats")
REP_LABEL: dict[str, str] = {
    "model-all": "model-all (pretrained z)",
    "random-init": "random-init TimesNet z",
    "length": "length only (n rows)",
    "stats": "simple stats (53 cols)",
}
FULL_EVAL_MIN = {"seeds": 3, "per_class": 20_000, "bootstrap": 1_000}


def esc(value: Any) -> str:
    """HTML-escape any value as text."""
    return html.escape(str(value), quote=True)


class Num:
    """Formats numbers for display; the exact value goes into a ``title`` tooltip."""

    @staticmethod
    def exact(value: Any) -> str:
        if isinstance(value, float):
            return repr(value)
        return str(value)

    @classmethod
    def f(cls, value: Any, digits: int = 3) -> str:
        """Fixed decimals (or a thousands-separated integer); ``None`` becomes a dash."""
        if value is None:
            return '<span class="na" title="null (not defined)">—</span>'
        if isinstance(value, bool):
            return esc(value)
        if isinstance(value, int):
            return f'<span class="n" title="{esc(value)}">{value:,}</span>'
        if not math.isfinite(value):
            return f'<span class="n">{esc(value)}</span>'
        return f'<span class="n" title="{esc(cls.exact(value))}">{value:,.{digits}f}</span>'

    @classmethod
    def pct(cls, value: float | None, digits: int = 1) -> str:
        """A share in [0, 1] shown as a percentage."""
        if value is None:
            return cls.f(None)
        return f'<span class="n" title="{esc(cls.exact(value))}">{100 * value:.{digits}f}%</span>'

    @classmethod
    def sci(cls, value: float | None) -> str:
        if value is None:
            return cls.f(None)
        return f'<span class="n" title="{esc(cls.exact(value))}">{value:.2e}</span>'

    @staticmethod
    def plain(value: float | None, digits: int = 3) -> str:
        """Unformatted text (for SVG labels; the SVG ``<title>`` carries the exact value)."""
        if value is None:
            return "n/a"
        return f"{value:.{digits}f}"

    @classmethod
    def mean_std_ci(cls, cell: dict[str, Any] | None, digits: int = 3) -> str:
        """``mean ± std [lo, hi]`` from a report_data cell."""
        if not cell or cell.get("mean") is None:
            return cls.f(None)
        out = cls.f(cell["mean"], digits)
        if cell.get("std") is not None:
            out += f' <span class="sub">± {cls.f(cell["std"], digits)}</span>'
        ci = cell.get("ci95") or {}
        if ci.get("lo") is not None:
            out += f' <span class="sub">[{cls.f(ci["lo"], digits)}, {cls.f(ci["hi"], digits)}]</span>'
        return out


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Round tick values covering [lo, hi]."""
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / max(1, n)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step) * step
    ticks, t = [], start
    while t <= hi + step * 1e-9:
        ticks.append(round(t, 12))
        t += step
    return ticks


def tick_label(value: float) -> str:
    a = abs(value)
    if a >= 1_000_000:
        return f"{value / 1e6:g}M"
    if a >= 1_000:
        return f"{value / 1e3:g}k"
    if a == 0:
        return "0"
    if a < 0.01:
        return f"{value:.2g}"
    return f"{value:g}"


@dataclasses.dataclass
class Series:
    """One line of a line chart."""

    label: str
    x: Sequence[float]
    y: Sequence[float | None]
    tone: int = 1
    dashed: bool = False
    points: bool = False
    tags: Sequence[str] | None = None


class LineChart:
    """Inline SVG line chart with nice ticks, optional reference lines and per-point tooltips."""

    W, H = 560, 250
    ML, MR, MT, MB = 58, 50, 14, 40

    def __init__(
        self,
        title: str,
        series: Sequence[Series],
        x_label: str,
        y_label: str = "",
        y_min: float | None = None,
        y_max: float | None = None,
        refs: Sequence[tuple[float, str]] = (),
        note: str = "",
    ) -> None:
        self.title, self.series, self.x_label, self.y_label = title, series, x_label, y_label
        self.y_min, self.y_max, self.refs, self.note = y_min, y_max, refs, note

    def render(self) -> str:
        xs = [x for s in self.series for x in s.x]
        ys = [y for s in self.series for y in s.y if y is not None] + [r[0] for r in self.refs]
        if not xs or not ys:
            return f'<figure class="chart"><figcaption>{esc(self.title)}</figcaption><p class="na">no data</p></figure>'
        x0, x1 = min(xs), max(xs)
        y0 = self.y_min if self.y_min is not None else min(ys)
        y1 = self.y_max if self.y_max is not None else max(ys)
        if self.y_min is None:
            y0 -= 0.05 * (y1 - y0 or 1)
        if self.y_max is None:
            y1 += 0.05 * (y1 - y0 or 1)
        yt = [t for t in nice_ticks(y0, y1) if y0 - 1e-12 <= t <= y1 + 1e-12]
        xt = [t for t in nice_ticks(x0, x1, 5) if x0 - 1e-12 <= t <= x1 + 1e-12]
        pw, ph = self.W - self.ML - self.MR, self.H - self.MT - self.MB

        def px(x: float) -> float:
            return self.ML + (x - x0) / ((x1 - x0) or 1) * pw

        def py(y: float) -> float:
            return self.MT + (1 - (y - y0) / ((y1 - y0) or 1)) * ph

        parts = [
            f'<svg viewBox="0 0 {self.W} {self.H}" role="img" aria-label="{esc(self.title)}" class="svgchart">',
            f"<title>{esc(self.title)}</title>",
        ]
        for t in yt:
            parts.append(
                f'<line class="grid" x1="{self.ML}" x2="{self.W - self.MR}" y1="{py(t):.1f}" y2="{py(t):.1f}"/>'
                f'<text class="tick" x="{self.ML - 6}" y="{py(t) + 4:.1f}" text-anchor="end">{tick_label(t)}</text>'
            )
        for t in xt:
            parts.append(
                f'<text class="tick" x="{px(t):.1f}" y="{self.H - self.MB + 16}" text-anchor="middle">'
                f"{tick_label(t)}</text>"
            )
        parts.append(
            f'<line class="axis" x1="{self.ML}" x2="{self.W - self.MR}" y1="{self.MT + ph}" y2="{self.MT + ph}"/>'
            f'<text class="axlabel" x="{self.ML + pw / 2:.0f}" y="{self.H - 6}" text-anchor="middle">'
            f"{esc(self.x_label)}</text>"
        )
        if self.y_label:
            parts.append(
                f'<text class="axlabel" transform="translate(13 {self.MT + ph / 2:.0f}) rotate(-90)" '
                f'text-anchor="middle">{esc(self.y_label)}</text>'
            )
        for value, label in self.refs:
            parts.append(
                f'<line class="ref" x1="{self.ML}" x2="{self.W - self.MR}" y1="{py(value):.1f}" y2="{py(value):.1f}">'
                f"<title>{esc(label)}: {esc(value)}</title></line>"
                f'<text class="reflabel" x="{self.W - self.MR + 4}" y="{py(value) + 4:.1f}" text-anchor="start">'
                f"{esc(label)}</text>"
            )
        for s in self.series:
            pts = [(px(x), py(y)) for x, y in zip(s.x, s.y) if y is not None]
            if not pts:
                continue
            d = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
            dash = ' stroke-dasharray="6 4"' if s.dashed else ""
            parts.append(
                f'<polyline class="line t{s.tone}" points="{d}"{dash}><title>{esc(s.label)}</title></polyline>'
            )
            if s.points:
                for i, (x, y) in enumerate(zip(s.x, s.y)):
                    if y is None:
                        continue
                    tag = s.tags[i] if s.tags else f"x = {x}"
                    parts.append(
                        f'<circle class="pt t{s.tone}" cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.2">'
                        f"<title>{esc(s.label)} · {esc(tag)}: {esc(Num.exact(y))}</title></circle>"
                    )
        parts.append("</svg>")
        legend = "".join(
            f'<span class="key"><i class="sw t{s.tone}{" dash" if s.dashed else ""}"></i>{esc(s.label)}</span>'
            for s in self.series
        )
        note = f'<p class="cnote">{self.note}</p>' if self.note else ""
        return (
            f'<figure class="chart"><figcaption>{esc(self.title)}</figcaption>{"".join(parts)}'
            f'<div class="legend">{legend}</div>{note}</figure>'
        )


@dataclasses.dataclass
class Bars:
    """One bar series of a grouped bar chart (``lo`` / ``hi`` = optional interval whiskers)."""

    label: str
    values: Sequence[float | None]
    tone: int = 1
    lo: Sequence[float | None] | None = None
    hi: Sequence[float | None] | None = None


class BarChart:
    """Inline SVG grouped (vertical) bar chart with value labels and optional whiskers."""

    W, ML, MR, MT, MB = 560, 52, 50, 16, 46

    def __init__(
        self,
        title: str,
        groups: Sequence[str],
        bars: Sequence[Bars],
        y_min: float = 0.0,
        y_max: float | None = None,
        refs: Sequence[tuple[float, str]] = (),
        y_label: str = "",
        value_fmt: Callable[[float], str] = lambda v: f"{v:.2f}",
        height: int = 250,
        show_values: bool = True,
        note: str = "",
    ) -> None:
        self.title, self.groups, self.bars, self.y_min, self.y_max = title, groups, bars, y_min, y_max
        self.refs, self.y_label, self.value_fmt, self.H = refs, y_label, value_fmt, height
        self.show_values, self.note = show_values, note

    def render(self) -> str:
        vals = [v for b in self.bars for v in b.values if v is not None]
        vals += [v for b in self.bars for v in (b.hi or []) if v is not None]
        y1 = self.y_max if self.y_max is not None else (max(vals) * 1.08 if vals else 1.0)
        y0 = self.y_min
        pw, ph = self.W - self.ML - self.MR, self.H - self.MT - self.MB

        def py(y: float) -> float:
            return self.MT + (1 - (y - y0) / ((y1 - y0) or 1)) * ph

        parts = [
            f'<svg viewBox="0 0 {self.W} {self.H}" role="img" aria-label="{esc(self.title)}" class="svgchart">',
            f"<title>{esc(self.title)}</title>",
        ]
        for t in nice_ticks(y0, y1):
            if y0 - 1e-12 <= t <= y1 + 1e-12:
                parts.append(
                    f'<line class="grid" x1="{self.ML}" x2="{self.W - self.MR}" y1="{py(t):.1f}" y2="{py(t):.1f}"/>'
                    f'<text class="tick" x="{self.ML - 6}" y="{py(t) + 4:.1f}" text-anchor="end">'
                    f"{tick_label(t)}</text>"
                )
        if self.y_label:
            parts.append(
                f'<text class="axlabel" transform="translate(12 {self.MT + ph / 2:.0f}) rotate(-90)" '
                f'text-anchor="middle">{esc(self.y_label)}</text>'
            )
        ng, nb = len(self.groups), max(1, len(self.bars))
        gw = pw / max(1, ng)
        bw = min(34.0, gw * 0.8 / nb)
        for gi, g in enumerate(self.groups):
            gx = self.ML + gi * gw + (gw - bw * nb) / 2
            for bi, b in enumerate(self.bars):
                v = b.values[gi]
                x = gx + bi * bw
                if v is None:
                    parts.append(
                        f'<text class="tick" x="{x + bw / 2:.1f}" y="{py(y0) - 4:.1f}" text-anchor="middle">n/a</text>'
                    )
                    continue
                top, base = py(max(min(v, y1), y0)), py(y0)
                parts.append(
                    f'<rect class="bar t{b.tone}" x="{x + 1:.1f}" y="{top:.1f}" width="{max(bw - 2, 1):.1f}" '
                    f'height="{max(base - top, 0.5):.1f}"><title>{esc(g)} · {esc(b.label)}: {esc(Num.exact(v))}'
                    f"</title></rect>"
                )
                lo = b.lo[gi] if b.lo else None
                hi = b.hi[gi] if b.hi else None
                if lo is not None and hi is not None:
                    cx = x + bw / 2
                    parts.append(
                        f'<path class="whisk" d="M{cx:.1f},{py(lo):.1f}V{py(hi):.1f}M{cx - 3:.1f},{py(lo):.1f}'
                        f'h6M{cx - 3:.1f},{py(hi):.1f}h6"><title>95% CI [{esc(lo)}, {esc(hi)}]</title></path>'
                    )
                if self.show_values and bw >= 22:
                    ty = min(py(hi) if hi is not None else top, top) - 4
                    parts.append(
                        f'<text class="vlabel" x="{x + bw / 2:.1f}" y="{ty:.1f}" text-anchor="middle">'
                        f"{esc(self.value_fmt(v))}</text>"
                    )
            parts.append(
                f'<text class="tick" x="{self.ML + gi * gw + gw / 2:.1f}" y="{self.H - self.MB + 16}" '
                f'text-anchor="middle">{esc(g)}</text>'
            )
        for value, label in self.refs:
            parts.append(
                f'<line class="ref" x1="{self.ML}" x2="{self.W - self.MR}" y1="{py(value):.1f}" y2="{py(value):.1f}">'
                f"<title>{esc(label)}</title></line>"
                f'<text class="reflabel" x="{self.W - self.MR + 4}" y="{py(value) + 4:.1f}" text-anchor="start">'
                f"{esc(label)}</text>"
            )
        parts.append(
            f'<line class="axis" x1="{self.ML}" x2="{self.W - self.MR}" y1="{py(y0):.1f}" y2="{py(y0):.1f}"/></svg>'
        )
        legend = (
            "".join(f'<span class="key"><i class="sw t{b.tone}"></i>{esc(b.label)}</span>' for b in self.bars)
            if len(self.bars) > 1
            else ""
        )
        note = f'<p class="cnote">{self.note}</p>' if self.note else ""
        return (
            f'<figure class="chart"><figcaption>{esc(self.title)}</figcaption>{"".join(parts)}'
            f'<div class="legend">{legend}</div>{note}</figure>'
        )


class StackedBar:
    """Horizontal 100% bars (one row per item), each split into labelled shares."""

    W, LABEL_W, ROW_H = 640, 170, 34

    def __init__(self, title: str, rows: Sequence[tuple[str, Sequence[tuple[str, float | None, int]]]]) -> None:
        self.title, self.rows = title, rows

    def render(self) -> str:
        h = 12 + self.ROW_H * len(self.rows)
        bw = self.W - self.LABEL_W - 10
        parts = [
            f'<svg viewBox="0 0 {self.W} {h}" role="img" aria-label="{esc(self.title)}" class="svgchart">',
            f"<title>{esc(self.title)}</title>",
        ]
        keys: dict[str, int] = {}
        for ri, (name, shares) in enumerate(self.rows):
            y = 6 + ri * self.ROW_H
            parts.append(f'<text class="tick" x="{self.LABEL_W - 8}" y="{y + 18}" text-anchor="end">{esc(name)}</text>')
            x = float(self.LABEL_W)
            for label, share, tone in shares:
                keys.setdefault(label, tone)
                if not share:
                    continue
                w = share * bw
                parts.append(
                    f'<rect class="bar t{tone}" x="{x:.1f}" y="{y}" width="{w:.1f}" height="26">'
                    f"<title>{esc(name)} · {esc(label)}: {esc(Num.exact(share))}</title></rect>"
                )
                if w > 44:
                    parts.append(
                        f'<text class="inbar" x="{x + w / 2:.1f}" y="{y + 18}" text-anchor="middle">'
                        f"{100 * share:.1f}%</text>"
                    )
                x += w
        parts.append("</svg>")
        legend = "".join(f'<span class="key"><i class="sw t{t}"></i>{esc(k)}</span>' for k, t in keys.items())
        return (
            f'<figure class="chart wide"><figcaption>{esc(self.title)}</figcaption>{"".join(parts)}'
            f'<div class="legend">{legend}</div></figure>'
        )


class Heatmap:
    """Row-normalised confusion matrix as an SVG grid."""

    def __init__(self, title: str, labels: Sequence[str], matrix: Sequence[Sequence[float]]) -> None:
        self.title, self.labels, self.matrix = title, labels, matrix

    def render(self) -> str:
        n, cell, lw, top = len(self.labels), 52, 104, 84
        w, h = lw + n * cell + 64, top + n * cell + 6
        parts = [
            f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(self.title)}" class="svgchart heat">',
            f"<title>{esc(self.title)}</title>",
            f'<text class="axlabel" x="{lw + n * cell / 2}" y="12" text-anchor="middle">predicted</text>',
        ]
        for j, lab in enumerate(self.labels):
            x = lw + j * cell + cell / 2
            parts.append(
                f'<text class="tick" transform="translate({x:.0f} {top - 6}) rotate(-35)" text-anchor="start">'
                f"{esc(lab)}</text>"
            )
        for i, lab in enumerate(self.labels):
            y = top + i * cell
            parts.append(f'<text class="tick" x="{lw - 6}" y="{y + cell / 2 + 4}" text-anchor="end">{esc(lab)}</text>')
            for j in range(n):
                v = float(self.matrix[i][j])
                parts.append(
                    f'<rect class="hcell" x="{lw + j * cell}" y="{y}" width="{cell - 2}" height="{cell - 2}" '
                    f'fill-opacity="{0.06 + 0.9 * v:.3f}"><title>true {esc(lab)} → predicted '
                    f"{esc(self.labels[j])}: {esc(Num.exact(v))}</title></rect>"
                    f'<text class="{"hin" if v > 0.5 else "hout"}" x="{lw + j * cell + cell / 2 - 1}" '
                    f'y="{y + cell / 2 + 4}" text-anchor="middle">{v:.2f}</text>'
                )
        parts.append("</svg>")
        return f'<figure class="chart small"><figcaption>{esc(self.title)}</figcaption>{"".join(parts)}</figure>'


class PipelineDiagram:
    """SVG of the pipeline: raw logs → links → windows → TimesNet → SSL heads; frozen z → probes / kNN score."""

    @staticmethod
    def box(x: float, y: float, w: float, h: float, lines: Sequence[str], kind: str = "") -> str:
        out = [f'<rect class="pbox {kind}" x="{x}" y="{y}" width="{w}" height="{h}" rx="8"/>']
        y0 = y + h / 2 - (len(lines) - 1) * 8 + 4
        for i, line in enumerate(lines):
            cls = "ptitle" if i == 0 else "psub"
            out.append(f'<text class="{cls}" x="{x + w / 2}" y="{y0 + i * 16}" text-anchor="middle">{esc(line)}</text>')
        return "".join(out)

    @staticmethod
    def arrow(d: str, label: str = "") -> str:
        return f'<path class="parrow" d="{d}" marker-end="url(#ah)"><title>{esc(label)}</title></path>'

    @classmethod
    def render(cls, seq_len: int, d_model: int, n_blocks: int) -> str:
        b = cls.box
        a = cls.arrow
        parts = [
            '<svg viewBox="0 0 1000 410" role="img" aria-label="Pipeline diagram" class="svgchart pipe">',
            "<title>Pipeline: raw VeReMi logs to frozen-encoder evaluation</title>",
            '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            'orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" class="ahead"/></marker></defs>',
            '<rect class="pgroup" x="392" y="150" width="600" height="118" rx="10"/>',
            '<text class="pglabel" x="402" y="166">self-supervised pretraining (no attack labels)</text>',
            '<rect class="pgroup eval" x="392" y="284" width="600" height="118" rx="10"/>',
            '<text class="pglabel" x="402" y="300">evaluation (labels only for probe fitting and metrics)</text>',
            b(8, 30, 170, 76, ["Raw VeReMi logs", "8 folders, traceJSON", "receiver view (D1′)"]),
            b(212, 30, 170, 76, ["Receiver links", "13 features / message", "no time of day (D3)"]),
            b(416, 30, 170, 76, [f"T = {seq_len} windows", "real rows + mask", "1–3 rows: F11"]),
            b(
                620,
                30,
                200,
                76,
                ["TimesNet encoder", f"{n_blocks} TimesBlocks, d = {d_model}", "H per step, z pooled"],
                "enc",
            ),
            a("M178,68H208"),
            a("M382,68H412"),
            a("M586,68H616"),
            b(408, 180, 160, 72, ["Masked recon", "MLP on H, MSE"], "head"),
            b(588, 180, 160, 72, ["P1–P3 heads", "injected violations"], "head"),
            b(768, 180, 160, 72, ["NT-Xent", "augmented views of z"], "head"),
            a("M650,106L492,176", "H"),
            a("M700,106L670,176", "z"),
            a("M780,106L846,176", "z"),
            b(768, 314, 160, 72, ["Frozen z", "best.pt, not tuned"], "frz"),
            b(588, 314, 160, 72, ["Probes", "LR + kNN, 2 / 5 class"], "out"),
            b(408, 314, 160, 72, ["kNN anomaly", "label-free, θ = p95"], "out"),
            a("M820,68H972V292H848V310", "encoder frozen after pretraining"),
            a("M768,350H752"),
            a("M848,386V396H488V390"),
        ]
        parts.append("</svg>")
        return f'<figure class="chart wide pipefig">{"".join(parts)}</figure>'


class Inputs:
    """Loads the input JSON files; a missing optional file becomes ``None``."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.summary_path = absolute(args.summary)
        self.manifest_path = absolute(args.manifest)
        self.findings_path = absolute(args.findings)
        self.eval_dir = absolute(args.eval_dir)
        self.metadata_path = absolute(args.metadata)
        self.summary = self.load(self.summary_path, required=True)
        self.manifest = self.load(self.manifest_path)
        self.findings = self.load(self.findings_path)
        self.metadata = self.load(self.metadata_path)
        self.eval_config = self.load(self.eval_dir / "config.json")
        self.eval_env = self.load(self.eval_dir / "env.json")
        report = self.load(self.eval_dir / "report_data.json")
        self.eval_status, self.eval_reason = self.classify_eval(report, self.eval_config)
        self.report = report if self.eval_status == "full" else None

    @staticmethod
    def load(path: pathlib.Path, required: bool = False) -> Any:
        if not path.is_file():
            if required:
                raise FileNotFoundError(f"required input missing: {path}")
            return None
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def classify_eval(report: Any, config: Any) -> tuple[str, str]:
        """``full`` only when config.json records at least the full-evaluation settings; never shows a smoke run."""
        if report is None:
            return "absent", "report_data.json not found"
        ev = (config or {}).get("evaluation") or {}
        if not ev:
            return "unverified", "config.json missing or without evaluation settings"
        problems = []
        if len(ev.get("seeds") or []) < FULL_EVAL_MIN["seeds"]:
            problems.append(f"seeds = {ev.get('seeds')}")
        if (ev.get("per_class") or 0) < FULL_EVAL_MIN["per_class"]:
            problems.append(f"per_class = {ev.get('per_class')}")
        if (ev.get("bootstrap") or 0) < FULL_EVAL_MIN["bootstrap"]:
            problems.append(f"bootstrap = {ev.get('bootstrap')}")
        if problems:
            return "reduced", "reduced settings (" + ", ".join(problems) + ")"
        return "full", ""


def absolute(path: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


class ReportPage:
    """Renders every section of the page from :class:`Inputs`."""

    def __init__(self, inputs: Inputs, built: str) -> None:
        self.i = inputs
        self.s = inputs.summary
        self.f = inputs.findings
        self.m = inputs.manifest
        self.built = built

    # ---------- helpers ----------
    @staticmethod
    def section(sid: str, title: str, body: str, lead: str = "") -> str:
        lead_html = f'<p class="lead">{lead}</p>' if lead else ""
        return f'<section id="{sid}"><h2>{esc(title)}</h2>{lead_html}{body}</section>'

    @staticmethod
    def note(text: str, kind: str = "warn") -> str:
        return f'<div class="callout {kind}">{text}</div>'

    @staticmethod
    def chip(text: str, kind: str) -> str:
        return f'<span class="chip {esc(kind)}">{esc(text)}</span>'

    @staticmethod
    def table(head: Sequence[str], rows: Sequence[Sequence[str]], cls: str = "") -> str:
        th = "".join(f"<th>{h}</th>" for h in head)
        body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
        return f'<div class="tablewrap"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'

    def findings_missing(self) -> str:
        return self.note(
            f"<strong>Findings missing.</strong> <code>{esc(rel(self.i.findings_path))}</code> was not found or "
            "is not valid; this part of the report is empty. Write the findings file, then run "
            "<code>npm run report:model-all</code> again."
        )

    # ---------- header ----------
    def header(self) -> str:
        s, f = self.s, self.f or {}
        rs, settings = s.get("run_summary", {}), s.get("settings", {})
        sha = (self.m or {}).get("model", {}).get("sha256") or (
            s.get("checksums", {}).get("files", {}).get("best.pt", {}).get("sha256")
        )
        evals = s.get("eval_per_epoch", [])
        status_eval = {
            "full": "frozen-probe evaluation done",
            "absent": "frozen-probe evaluation pending",
        }.get(self.i.eval_status, "frozen-probe evaluation pending (only a reduced run found)")
        conv = next((v for v in s.get("verdicts", []) if v.get("check") == "converged?"), None)
        conv_text = "not converged" if conv and conv.get("status") == "look" else "converged"
        status = (
            f"{rs.get('epoch', '?')} epochs, {Num.f(rs.get('step'))} batches, {Num.f(rs.get('hours'), 1)} h; "
            f"best val_joint {Num.f(rs.get('best_val_joint'), 3)} ({esc(evals[-1]['tag'] if evals else '?')}); "
            f"{conv_text}; {status_eval}."
        )
        title = f.get("title") or "model-all: findings report"
        date = f.get("date") or self.built[:10]
        author = f.get("author")
        meta = [
            f"<span><b>Model</b> {esc(rel(absolute(s.get('run_dir', ''))))} · TimesNet d = {esc(settings.get('d_model'))}, "
            f"T = {esc(settings.get('seq_len'))}, seed {esc(settings.get('seed'))}</span>",
            f'<span><b>Checkpoint</b> best.pt · sha256 <code title="{esc(sha)}">{esc((sha or "?")[:12])}</code></span>',
            f"<span><b>Date</b> {esc(date)}{' · ' + esc(author) if author else ''}</span>",
        ]
        toc = "".join(
            f'<a href="#{sid}">{esc(sid.capitalize() if sid != "fixes" else "Fixes & direction")}</a>'
            for sid in SECTION_IDS
        )
        return (
            f'<header id="top"><p class="kicker">RoadFM-Lite · pretraining report</p><h1>{esc(title)}</h1>'
            f'<div class="meta">{"".join(meta)}</div><p class="status">{status}</p>'
            f'<nav aria-label="Sections">{toc}</nav></header>'
        )

    # ---------- summary ----------
    def summary(self) -> str:
        if not self.f:
            return self.section("summary", "Summary", self.findings_missing())
        cards = []
        for item in self.f.get("summary", []):
            st = item.get("status", "pending")
            cards.append(
                f'<article class="card {esc(st)}"><div class="cardhead">{self.chip(st, st)}</div>'
                f"<h3>{esc(item.get('q', ''))}</h3><p>{esc(item.get('a', ''))}</p></article>"
            )
        pending = self.f.get("pending") or []
        pend = (
            "<h3>Still pending</h3><ul>" + "".join(f"<li>{esc(p)}</li>" for p in pending) + "</ul>" if pending else ""
        )
        return self.section("summary", "Summary", f'<div class="cards">{"".join(cards)}</div>{pend}')

    # ---------- pipeline ----------
    def pipeline(self) -> str:
        st = self.s.get("settings", {})
        diagram = PipelineDiagram.render(st.get("seq_len", 24), st.get("d_model", 128), st.get("n_blocks", 4))
        p = self.s.get("params", {})
        text = (
            f"<p>Encoder {Num.f(p.get('params_encoder'))} parameters, {Num.f(p.get('params_total'))} with the "
            f"pretraining heads. Joint loss = {esc(st.get('lambda_recon'))} × reconstruction + "
            f"{esc(st.get('lambda_nce'))} × NT-Xent + {esc(st.get('lambda_p1'))} × (P1 + P2 + P3); windows with "
            f"1 row carry no loss, 2–{esc((st.get('min_rows_aux') or 4) - 1)} rows only reconstruction, "
            f"≥ {esc(st.get('min_rows_aux'))} rows all five. The heads are discarded after pretraining; evaluation "
            "uses the frozen pooled embedding z.</p>"
        )
        return self.section("pipeline", "Pipeline", diagram + text)

    # ---------- data ----------
    def data(self) -> str:
        d = self.s.get("data", {})
        scan = d.get("length_scan", {})
        hist = scan.get("hist_n", {})
        parts = []
        rows = d.get("rows", {})
        parts.append(
            f"<p>Pretraining read the train split of <code>{esc(self.s.get('settings', {}).get('data_dir'))}</code> "
            f"only: {Num.f(rows.get('train'))} windows for training and {Num.f(rows.get('pretrain_check'))} in the "
            f"label-free check set (10% of train vehicles); label files opened: "
            f"{Num.f(d.get('label_files_opened'))}. {esc(d.get('dataset_note', ''))}.</p>"
        )
        meta = self.i.metadata
        if meta and meta.get("counts"):
            per: dict[str, list[int]] = {c: [0, 0] for c in CLASSES}
            for split in meta["counts"].values():
                for fam in split.values():
                    for cls, c in fam.items():
                        if cls in per:
                            per[cls][0] += int(c.get("windows", 0))
                            per[cls][1] += int(c.get("windows_1_3", 0))
            trs = [
                [
                    esc(SHORT[c]),
                    Num.f(per[c][0]),
                    Num.f(per[c][1]),
                    Num.pct(per[c][1] / per[c][0] if per[c][0] else None),
                ]
                for c in CLASSES
            ]
            tot = meta.get("totals", {})
            trs.append(
                [
                    "<b>all</b>",
                    Num.f(tot.get("windows")),
                    Num.f(tot.get("windows_1_3")),
                    Num.pct(tot["windows_1_3"] / tot["windows"] if tot.get("windows") else None),
                ]
            )
            parts.append(
                "<h3>Windows per class (all/T24, train + test, both groups)</h3>"
                + self.table(["Class", "Windows", "1–3 rows", "Share 1–3 rows"], trs, "num")
                + '<p class="small">From <code>src/data/encoder_input/all/T24/metadata.json</code>. Benign windows '
                "repeat across the four families (they re-simulate the same traffic, F1).</p>"
            )
        else:
            parts.append(self.note("Encoder-input metadata.json not found: per-class window counts not shown.", "info"))
        if hist:
            ns = sorted(hist, key=int)
            total = sum(hist.values())
            note = (
                f"{Num.f(scan.get('windows'))} windows in {Num.f(scan.get('shards'))} shards. n = 1: "
                f"{Num.pct(scan.get('share_n1_no_loss'))} (no loss); n = 2–3: "
                f"{Num.pct(scan.get('share_n2_3_recon_only'))} (reconstruction only); n ≥ 4: "
                f"{Num.pct(scan.get('share_n_ge_4_all_losses'))} (all losses)."
            )
            chart = BarChart(
                "Train windows by number of real rows n (label-free scan)",
                ns,
                [Bars("windows", [hist[k] / total for k in ns], 1)],
                y_label="share of windows",
                value_fmt=lambda v: f"{100 * v:.0f}%" if v >= 0.05 else "",
                note=note,
            )
            parts.append(f'<div class="grid1">{chart.render()}</div>')
        if self.m:
            full = self.m.get("metrics", {}).get("full_test_1416", {})
            trs = [
                [esc(SHORT[c]), Num.f(full.get(c, {}).get("n")), Num.f(full.get(c, {}).get("mean_rows"), 2)]
                for c in CLASSES
                if c in full
            ]
            parts.append(
                "<h3>Test windows of group 1416 (the scope of every test number below)</h3>"
                + self.table(["Class", "Windows", "Mean real rows"], trs, "num")
            )
        parts.append(
            self.note(
                "<strong>Scope.</strong> F11: DataReplay and DoS rotate pseudonyms every 1–2 messages, so their "
                "windows are almost all 1–3 rows and their length alone separates them from benign. F14: "
                "GridSybil_0709 is a separate re-simulation whose vehicles near-copy across the vehicle split, so "
                "0709 test windows are excluded until F14 is decided; every test number on this page is group 1416.",
                "info",
            )
        )
        return self.section("data", "Data", "".join(parts))

    # ---------- training ----------
    def training(self) -> str:
        s = self.s
        curves = s.get("curves", {}).get("train", {})
        ev = s.get("eval_per_epoch", [])
        out = []
        loss_charts = []
        names = {"recon": "reconstruction", "nce": "NT-Xent", "p1": "P1", "p2": "P2", "p3": "P3"}
        akb, rob = curves.get("all_losses_batches", {}), curves.get("recon_only_batches", {})
        if akb.get("joint"):
            ser = [Series("all-loss batches (n ≥ 4)", akb["joint"]["step"], akb["joint"]["value"], 1)]
            if rob.get("joint"):
                ser.append(
                    Series("reconstruction-only batches (n 2–3)", rob["joint"]["step"], rob["joint"]["value"], 2)
                )
            loss_charts.append(LineChart("Train joint loss (normalised)", ser, "batch").render())
        if akb.get("recon"):
            ser = [Series("all-loss batches", akb["recon"]["step"], akb["recon"]["value"], 1)]
            if rob.get("recon"):
                ser.append(Series("reconstruction-only batches", rob["recon"]["step"], rob["recon"]["value"], 2))
            loss_charts.append(LineChart("Train reconstruction loss", ser, "batch").render())
        aux = [Series(names[k], akb[k]["step"], akb[k]["value"], t) for k, t in (("nce", 3),) if akb.get(k)]
        if aux:
            loss_charts.append(LineChart("Train NT-Xent loss", aux, "batch").render())
        phys = [
            Series(names[k], akb[k]["step"], akb[k]["value"], t)
            for k, t in (("p1", 4), ("p2", 5), ("p3", 6))
            if akb.get(k)
        ]
        if phys:
            loss_charts.append(LineChart("Train physics-head BCE (P1–P3)", phys, "batch").render())
        if loss_charts:
            out.append(
                "<h3>Train losses per component</h3>"
                f'<p class="small">{esc(s.get("curves", {}).get("method", ""))}; the data are length-bucketed, so a '
                "batch's losses follow its length (curves split by batch kind).</p>"
                f'<div class="grid2">{"".join(loss_charts)}</div>'
            )
        if ev:
            xs = [e["epoch"] for e in ev]
            tags = [e["tag"] for e in ev]

            def ser(key: str, label: str, tone: int, dashed: bool = False) -> Series:
                return Series(label, xs, [e.get(key) for e in ev], tone, dashed, True, tags)

            charts = [
                LineChart("val_joint per epoch (check set)", [ser("val_joint", "val_joint", 1)], "epoch (0 = init)"),
                LineChart(
                    "Masked reconstruction vs linear interpolation",
                    [ser("val_recon", "model", 1), ser("val_recon_interp_baseline", "linear interpolation", 2, True)],
                    "epoch (0 = init)",
                    y_min=0,
                ),
                LineChart(
                    "P1–P3 AUROC on injected violations",
                    [ser("auroc_p1", "P1", 4), ser("auroc_p2", "P2", 5), ser("auroc_p3", "P3", 6)],
                    "epoch (0 = init)",
                    y_min=0.4,
                    y_max=1.0,
                    refs=[(0.5, "chance")],
                    note="Detection of violations we inject, not of real attacks.",
                ),
                LineChart(
                    "Effective rank of z",
                    [ser("z_effective_rank", "effective rank", 3)],
                    "epoch (0 = init)",
                    y_min=0,
                    y_max=float(s.get("settings", {}).get("d_model", 128)),
                    note=f"Upper bound = d = {esc(s.get('settings', {}).get('d_model', 128))}.",
                ),
                LineChart(
                    "Window length recoverable from z (ridge R²)",
                    [ser("length_probe_r2", "length probe R²", 2)],
                    "epoch (0 = init)",
                    y_min=0,
                    y_max=1,
                    note="High = z encodes how many real rows the window has (F11 shortcut).",
                ),
            ]
            out.append(
                "<h3>Evaluation per epoch (label-free check set)</h3>"
                f'<div class="grid2">{"".join(c.render() for c in charts)}</div>'
            )
        out.append(self.lr_chart())
        out.append(self.skipped_chart())
        out.append(self.settings_table())
        out.append(self.verdicts())
        lead = esc(s.get("scope_note", ""))
        return self.section("training", "Training", "".join(out), lead)

    def lr_chart(self) -> str:
        lr = self.s.get("lr_schedule", {})
        curve = self.s.get("curves", {}).get("train", {}).get("lr")
        if not curve or not lr:
            return ""
        peak, warm, total = lr.get("peak_lr", 0.0), lr.get("warmup_steps", 0), lr.get("planned_total_steps", 1)
        xs = [0] + list(range(warm, total + 1, max(1, total // 200))) + [total]

        def planned(step: int) -> float:
            if step < warm:
                return peak * (step + 1) / warm
            prog = (step - warm) / max(1, total - warm)
            return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

        chart = LineChart(
            "Learning rate: logged vs the configured schedule",
            [
                Series("logged LR", curve["step"], curve["value"], 5),
                Series("configured: warm-up + cosine to 0 over all batches", xs, [planned(x) for x in xs], 2, True),
            ],
            "batch",
            y_min=0,
            note=(
                f"Last logged LR {Num.sci(lr.get('last_logged_lr'))} = {Num.pct(lr.get('final_lr_over_peak'))} of peak "
                f"at batch {Num.f(lr.get('last_logged_step'))}; the configured schedule is "
                f"{Num.sci(lr.get('planned_lr_at_that_step'))} there. {esc(lr.get('note', ''))}."
            ),
        )
        return f'<h3>Learning-rate schedule</h3><div class="grid1">{chart.render()}</div>'

    def skipped_chart(self) -> str:
        bk = self.s.get("batch_kinds", {}).get("share_of_all_batches_est", {})
        scan = self.s.get("data", {}).get("length_scan", {})
        if not bk:
            return ""
        rows = [
            (
                "batches (estimated)",
                [
                    ("no loss (skipped)", bk.get("skipped_no_loss"), 7),
                    ("reconstruction only", bk.get("recon_only"), 2),
                    ("all five losses", bk.get("all_losses"), 1),
                ],
            )
        ]
        if scan:
            rows.append(
                (
                    "train windows",
                    [
                        ("no loss (skipped)", scan.get("share_n1_no_loss"), 7),
                        ("reconstruction only", scan.get("share_n2_3_recon_only"), 2),
                        ("all five losses", scan.get("share_n_ge_4_all_losses"), 1),
                    ],
                )
            )
        st = self.s.get("steps", {})
        note = (
            f'<p class="small">Of {Num.f(st.get("total_batches"))} batches, skipped between '
            f"{Num.f((st.get('skipped_bracket') or [None])[0])} and {Num.f((st.get('skipped_bracket') or [None, None])[1])};"
            f" real updates {Num.f((st.get('real_updates_bracket') or [None])[0])}–"
            f"{Num.f((st.get('real_updates_bracket') or [None, None])[1])}. "
            f"{esc(self.s.get('batch_kinds', {}).get('note', ''))}.</p>"
        )
        return (
            "<h3>Which batches trained anything</h3>"
            + StackedBar("Share of batches / windows by loss coverage", rows).render()
            + note
        )

    def settings_table(self) -> str:
        st, hw, tm = self.s.get("settings", {}), self.s.get("hardware", {}), self.s.get("timing", {})
        keys = [
            ("dataset", "dataset"),
            ("seq_len", "T"),
            ("d_model", "d"),
            ("d_ff", "d_ff"),
            ("n_blocks", "TimesBlocks"),
            ("top_k", "top-k periods"),
            ("batch_size", "batch"),
            ("length_bucketed", "length-bucketed (D14)"),
            ("lr", "peak LR"),
            ("weight_decay", "weight decay"),
            ("warmup_steps", "warm-up batches"),
            ("max_epochs", "epochs"),
            ("lambda_recon", "λ recon"),
            ("lambda_nce", "λ NT-Xent"),
            ("lambda_p1", "λ P1 = λ P2 = λ P3"),
            ("temperature", "τ"),
            ("inject_prob", "injection p"),
            ("min_rows_aux", "min rows for aux losses"),
            ("seed", "seed"),
            ("optimiser", "optimiser"),
            ("schedule", "schedule (as configured)"),
        ]
        rows = [
            [esc(label), Num.f(st.get(k)) if isinstance(st.get(k), (int, float)) else esc(st.get(k))]
            for k, label in keys
        ]
        rows += [
            ["device", esc(f"{hw.get('device')} · torch {hw.get('torch')} · {hw.get('instance_note', '')}")],
            ["wall time", f"{Num.f(tm.get('wall_hours'), 2)} h ({Num.f(tm.get('mean_hours_per_epoch'), 3)} h / epoch)"],
            [
                "median step time",
                esc(", ".join(f"{k} {v} s" for k, v in (tm.get("step_seconds_median") or {}).items())),
            ],
            ["peak host RAM", f"{Num.f(hw.get('mem_gb_peak'), 2)} GB · {esc(hw.get('mem_gb_note', ''))}"],
        ]
        half = (len(rows) + 1) // 2
        return (
            '<h3>Settings</h3><div class="grid2">'
            + self.table(["Setting", "Value"], rows[:half], "kv")
            + self.table(["Setting", "Value"], rows[half:], "kv")
            + "</div>"
        )

    def verdicts(self) -> str:
        items = []
        for v in self.s.get("verdicts", []):
            st = v.get("status", "")
            items.append(
                f'<li>{self.chip("ok" if st == "ok" else "look", "yes" if st == "ok" else "partly")} '
                f"<b>{esc(v.get('check'))}</b> {esc(v.get('verdict'))}</li>"
            )
        return '<h3>Health verdicts (from the summariser)</h3><ul class="verdicts">' + "".join(items) + "</ul>"

    # ---------- test ----------
    def test(self) -> str:
        parts = ["<h3>(a) Label-free kNN anomaly score (no labels, no probe)</h3>", self.anomaly()]
        parts.append("<h3>(b) Frozen-encoder probes</h3>")
        parts.append(self.probes())
        return self.section(
            "test",
            "Test",
            "".join(parts),
            "Test split, group 1416 only (F14). Labels were opened only after every window was scored.",
        )

    def anomaly(self) -> str:
        if not self.m:
            return self.note("Detection manifest not found: run <code>npm run sim:detect</code>.")
        meth, met = self.m.get("method", {}), self.m.get("metrics", {})
        full = met.get("full_test_1416", {})
        rows = []
        for c in CLASSES:
            e = full.get(c)
            if not e:
                continue
            beat = None
            if e.get("auroc_vs_benign") is not None and e.get("length_baseline_auroc") is not None:
                beat = e["auroc_vs_benign"] > e["length_baseline_auroc"]
            rows.append(
                [
                    esc(SHORT[c]),
                    Num.f(e.get("n")),
                    Num.f(e.get("mean_rows"), 2),
                    Num.f(e.get("auroc_vs_benign")),
                    Num.f(e.get("length_baseline_auroc")),
                    "—" if beat is None else (self.chip("yes", "yes") if beat else self.chip("no", "no")),
                    Num.pct(e.get("flag_rate")),
                ]
            )
        table = self.table(
            ["Class", "Windows", "Mean rows", "AUROC vs benign", "Length-only AUROC", "Beats length?", "Flagged at θ"],
            rows,
            "num",
        )
        chart = BarChart(
            "AUROC vs benign: model score vs length-only score",
            [SHORT[f] for f in FAMILIES],
            [
                Bars("model-all kNN score", [full.get(f, {}).get("auroc_vs_benign") for f in FAMILIES], 1),
                Bars("length only (−n rows)", [full.get(f, {}).get("length_baseline_auroc") for f in FAMILIES], 7),
            ],
            y_max=1.0,
            refs=[(0.5, "chance")],
            y_label="AUROC",
        ).render()
        fpr = met.get("benign_fpr_at_theta")
        model = self.m.get("model", {})
        info = (
            f"<p>Score = {esc(meth.get('name'))}, k = {esc(meth.get('k'))}, bank {Num.f(meth.get('bank_size'))} "
            f"unlabeled train windows; θ = {Num.f(meth.get('theta'), 4)}. Benign false-positive rate at θ: "
            f"<b>{Num.pct(fpr)}</b> (nominal 5%, since θ is the p95 of check-set scores). Checkpoint "
            f"{esc(model.get('checkpoint'))}, epoch {esc(model.get('epoch'))}.</p>"
            f'<p class="small">θ rule: {esc(meth.get("theta_rule", ""))}</p>'
        )
        cav = self.m.get("caveats") or []
        cav_html = (
            '<details open><summary>Caveats from the exporter</summary><ul class="small">'
            + "".join(f"<li>{esc(c)}</li>" for c in cav)
            + "</ul></details>"
            if cav
            else ""
        )
        return table + f'<div class="grid1">{chart}</div>' + info + cav_html

    def probes(self) -> str:
        status = self.i.eval_status
        if status != "full":
            extra = ""
            if status in ("reduced", "unverified"):
                extra = (
                    f"<p>A results folder exists at <code>{esc(rel(self.i.eval_dir))}</code> but it is not the full "
                    f"evaluation ({esc(self.i.eval_reason)}); its numbers are not shown.</p>"
                )
            return self.note(
                "<strong>Pending.</strong> The frozen-probe evaluation (model-all vs a random-init TimesNet vs "
                "length-only vs simple statistics; logistic regression + kNN; binary + 5-class; 3 seeds; "
                "vehicle-bootstrap 95% CI; n ≤ 3 vs n ≥ 4) has not been run on the full test split yet."
                f"{extra}<p>Run <code>npm run eval:all</code> (on EC2), copy "
                f"<code>{esc(EVAL_DIR)}/</code> back, then <code>npm run report:model-all</code>.</p>",
                "pending",
            )
        r = self.i.report or {}
        out = []
        out.append(
            f'<p class="small">Seeds {esc(r.get("seeds"))}; created {esc(r.get("created"))}; '
            f"{esc((r.get('scope') or {}).get('note', ''))}.</p>"
        )
        main = r.get("main_table", [])
        rows = []
        for row in main:
            rows.append(
                [
                    esc(REP_LABEL.get(row["representation"], row["representation"])),
                    Num.mean_std_ci(row.get("lr_binary_auroc")),
                    Num.mean_std_ci(row.get("lr_macro_f1")),
                    Num.mean_std_ci(row.get("knn_binary_auroc")),
                    Num.mean_std_ci(row.get("knn_macro_f1")),
                    Num.mean_std_ci(row.get("anomaly_binary_auroc")),
                ]
            )
        out.append(
            "<h4>Main table (mean ± std over seeds [vehicle-bootstrap 95% CI])</h4>"
            + self.table(
                [
                    "Representation",
                    "LR binary AUROC",
                    "LR 5-class macro F1",
                    "kNN binary AUROC",
                    "kNN macro F1",
                    "Anomaly AUROC",
                ],
                rows,
                "num",
            )
        )

        def cells(key: str) -> tuple[list, list, list]:
            vals, lo, hi = [], [], []
            for rep in REPRESENTATIONS:
                row = next((x for x in main if x["representation"] == rep), {})
                c = row.get(key) or {}
                ci = c.get("ci95") or {}
                vals.append(c.get("mean"))
                lo.append(ci.get("lo"))
                hi.append(ci.get("hi"))
            return vals, lo, hi

        bars = []
        for key, label, tone in (
            ("lr_binary_auroc", "LR", 1),
            ("knn_binary_auroc", "kNN", 3),
            ("anomaly_binary_auroc", "anomaly", 7),
        ):
            v, lo, hi = cells(key)
            bars.append(Bars(label, v, tone, lo, hi))
        out.append(
            '<div class="grid1">'
            + BarChart(
                "Binary AUROC (attack vs benign) by representation",
                [REP_LABEL[r].split(" (")[0] for r in REPRESENTATIONS],
                bars,
                y_max=1.0,
                refs=[(0.5, "chance")],
                y_label="AUROC",
            ).render()
            + "</div>"
        )
        fam = r.get("per_family", [])
        for probe, plabel in (("lr", "logistic regression"), ("knn", "kNN probe"), ("anomaly", "label-free anomaly")):
            trs = []
            for e in fam:
                if e.get("probe") != probe:
                    continue
                trs.append(
                    [
                        esc(SHORT.get(e["family"], e["family"])),
                        esc(e["representation"]),
                    ]
                    + [
                        Num.mean_std_ci(
                            {
                                "mean": (e.get(s) or {}).get("auroc_mean"),
                                "std": (e.get(s) or {}).get("auroc_std"),
                                "ci95": (e.get(s) or {}).get("auroc_ci95"),
                            }
                        )
                        + f' <span class="sub">n {Num.f((e.get(s) or {}).get("n"))}</span>'
                        for s in ("all", "n<=3", "n>=4")
                    ]
                )
            if trs:
                out.append(
                    f"<h4>Per-family AUROC vs benign — {esc(plabel)}</h4>"
                    + self.table(["Family", "Representation", "all lengths", "n ≤ 3", "n ≥ 4"], trs, "num")
                )
        strat = []
        for rep, tone in (("model-all", 1), ("random-init", 2)):
            for s, dashed_tone in (("n<=3", 0), ("n>=4", 3)):
                vals = []
                for f in FAMILIES:
                    e = next(
                        (x for x in fam if x["representation"] == rep and x["probe"] == "lr" and x["family"] == f), {}
                    )
                    vals.append((e.get(s) or {}).get("auroc_mean"))
                strat.append(Bars(f"{rep} {s.replace('<=', ' ≤ ').replace('>=', ' ≥ ')}", vals, tone + dashed_tone))
        out.append(
            '<div class="grid1">'
            + BarChart(
                "LR probe AUROC per family: short (n ≤ 3) vs long (n ≥ 4) windows",
                [SHORT[f] for f in FAMILIES],
                strat,
                y_max=1.0,
                refs=[(0.5, "chance")],
                y_label="AUROC",
            ).render()
            + "</div>"
        )
        conf = r.get("confusion_row_normalised_mean", {})
        labels = [SHORT[c] for c in r.get("classes", CLASSES)]
        heats = []
        for rep in ("model-all", "random-init"):
            for probe in ("lr", "knn"):
                mat = (conf.get(rep) or {}).get(probe)
                if mat:
                    heats.append(Heatmap(f"{rep} · {probe} (row-normalised, seed mean)", labels, mat).render())
        if heats:
            out.append(f'<h4>5-class confusion matrices</h4><div class="grid2">{"".join(heats)}</div>')
        br = r.get("benign_flagged_rate", {})
        if br:
            trs = []
            for rep in REPRESENTATIONS:
                for probe in ("lr", "knn", "anomaly"):
                    e = (br.get(rep) or {}).get(probe) or {}
                    trs.append([esc(rep), esc(probe)] + [Num.pct(e.get(s)) for s in ("all", "n<=3", "n>=4")])
            out.append(
                "<h4>Benign windows flagged as attack (false-positive rate)</h4>"
                + self.table(["Representation", "Probe", "all", "n ≤ 3", "n ≥ 4"], trs, "num")
            )
        notes = r.get("notes") or []
        if notes:
            out.append('<ul class="small">' + "".join(f"<li>{esc(n)}</li>" for n in notes) + "</ul>")
        return "".join(out)

    # ---------- diagnosis / fixes / limitations ----------
    def diagnosis(self) -> str:
        if not self.f:
            return self.section("diagnosis", "Diagnosis", self.findings_missing())
        order = {"high": 0, "medium": 1, "low": 2}
        items = []
        for it in sorted(self.f.get("issues", []), key=lambda x: order.get(x.get("severity"), 3)):
            sev = it.get("severity", "low")
            ev = "".join(f"<li>{esc(e)}</li>" for e in it.get("evidence", []))
            items.append(
                f'<article class="issue {esc(sev)}"><div class="ihead"><span class="iid">{esc(it.get("id"))}</span>'
                f"{self.chip(sev, sev)}<h3>{esc(it.get('title'))}</h3></div>"
                f'<ul class="evidence">{ev}</ul><p><b>Impact.</b> {esc(it.get("impact", ""))}</p></article>'
            )
        return self.section("diagnosis", "Diagnosis", "".join(items))

    def fixes(self) -> str:
        if not self.f:
            return self.section("fixes", "What to fix & direction", self.findings_missing())
        rows = []
        for x in sorted(self.f.get("fixes", []), key=lambda x: x.get("priority", 99)):
            ok = x.get("needs_user_ok")
            rows.append(
                [
                    Num.f(x.get("priority")),
                    f"<b>{esc(x.get('id'))}</b> {esc(x.get('title'))}",
                    esc(", ".join(x.get("addresses", []))),
                    esc(x.get("what", "")),
                    esc(x.get("expected_effect", "")),
                    esc(x.get("cost", "")),
                    esc(x.get("plan_task", "")),
                    self.chip("needs OK", "partly") if ok else self.chip("no", "yes"),
                ]
            )
        table = self.table(
            ["#", "Fix", "Addresses", "What", "Expected effect", "Cost", "Plan task", "User OK"], rows, "fixes"
        )
        steps = self.f.get("direction", [])
        road = "".join(
            f'<li><svg viewBox="0 0 28 28" class="node" aria-hidden="true"><circle cx="14" cy="14" r="12"/>'
            f'<text x="14" y="19" text-anchor="middle">{i}</text></svg><p>{esc(t)}</p></li>'
            for i, t in enumerate(steps, 1)
        )
        return self.section(
            "fixes",
            "What to fix & direction",
            f'<h3>Prioritised fixes</h3>{table}<h3>Roadmap</h3><ol class="roadmap">{road}</ol>',
        )

    def limitations(self) -> str:
        if not self.f:
            return self.section("limitations", "Limitations", self.findings_missing())
        items = "".join(f"<li>{esc(x)}</li>" for x in self.f.get("limitations", []))
        return self.section("limitations", "Limitations", f"<ul>{items}</ul>")

    # ---------- reproducibility ----------
    def reproducibility(self) -> str:
        s = self.s
        cmds = [
            ("train (EC2, g4dn T4)", "npm run train:all"),
            (
                "training summary",
                "python scripts/summarize_pretraining.py --run src/runs/pretraining/all/T24/model-all",
            ),
            ("label-free anomaly export", "npm run sim:detect"),
            ("frozen-probe evaluation", "npm run eval:all"),
            ("this page", "npm run report:model-all"),
        ]
        cmd_html = "".join(f"<dt>{esc(a)}</dt><dd><code>{esc(b)}</code></dd>" for a, b in cmds)
        hw = s.get("hardware", {})
        files = s.get("checksums", {}).get("files", {})
        sums = [
            [
                f"<code>{esc(k)}</code>",
                f'<code title="{esc(v.get("sha256"))}">{esc((v.get("sha256") or "")[:16])}…</code>',
                Num.f(v.get("bytes")),
                self.chip("ok" if v.get("ok") else "mismatch", "yes" if v.get("ok") else "no"),
            ]
            for k, v in files.items()
        ]
        dates = [
            ["run started (env.json)", esc(hw.get("created_utc"))],
            ["training summary", esc(s.get("created_utc"))],
            ["detection export", esc((self.m or {}).get("created"))],
            ["frozen-probe evaluation", esc((self.i.report or {}).get("created")) if self.i.report else "pending"],
            ["findings", esc((self.f or {}).get("date")) if self.f else "missing"],
            ["page built", esc(self.built)],
        ]
        inputs = [
            [f"<code>{esc(rel(p))}</code>", "present" if ok else "absent"]
            for p, ok in (
                (self.i.summary_path, True),
                (self.i.manifest_path, self.m is not None),
                (self.i.findings_path, self.f is not None),
                (self.i.eval_dir / "report_data.json", self.i.eval_status != "absent"),
                (self.i.metadata_path, self.i.metadata is not None),
            )
        ]
        body = (
            f'<dl class="cmds">{cmd_html}</dl>'
            f"<p>Run command: <code>{esc(s.get('command', ''))}</code></p>"
            f"<p>Pretraining seed {esc(s.get('settings', {}).get('seed'))}; git {esc(hw.get('git_commit', '')[:10])}"
            f"{' (dirty)' if hw.get('git_dirty') else ''}; Python {esc(hw.get('python'))}, torch {esc(hw.get('torch'))}, "
            f"numpy {esc(hw.get('numpy'))}; {esc(hw.get('platform'))}.</p>"
            '<div class="grid2"><div><h3>Run files (SHA256SUMS checked)</h3>'
            + self.table(["File", "sha256", "Bytes", "Check"], sums, "num")
            + "</div><div><h3>Dates</h3>"
            + self.table(["What", "When (UTC)"], dates, "kv")
            + "</div></div><h3>Inputs of this page</h3>"
            + self.table(["File", "Status"], inputs, "kv")
        )
        return self.section("reproducibility", "Reproducibility", body)

    def render(self) -> str:
        body = "".join(
            [
                self.header(),
                "<main>",
                self.summary(),
                self.pipeline(),
                self.data(),
                self.training(),
                self.test(),
                self.diagnosis(),
                self.fixes(),
                self.limitations(),
                self.reproducibility(),
                "</main>",
                f"<footer>Generated by <code>scripts/build_model_report.py</code> on {esc(self.built)} from files "
                "written by other tools; numbers are shown as produced (rounded for display, exact value on hover)."
                "</footer>",
            ]
        )
        title = esc((self.f or {}).get("title") or "model-all report")
        return (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>model-all report</title><meta name="description" content="{title}">'
            f"<style>{CSS}</style></head><body>{body}</body></html>\n"
        )


CSS = """
:root{--bg:#fbfbfa;--panel:#ffffff;--text:#1d1f23;--muted:#5b616b;--line:#d9dce1;--grid:#eceef1;--accent:#2457c5;
--on-accent:#ffffff;--t1:#2457c5;--t2:#d0661a;--t3:#16865a;--t4:#8a3fbf;--t5:#c2304a;--t6:#0f7f95;--t7:#7c828c;
--ok:#16794d;--okbg:#e5f4ec;--warn:#9a5b00;--warnbg:#fdf2dd;--bad:#b42335;--badbg:#fbe7ea;--info:#1f5fa8;
--infobg:#e7f0fb;--pend:#5b616b;--pendbg:#eef0f3;color-scheme:light dark}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15171b;--panel:#1d2026;--text:#e7e9ec;
--muted:#a3a9b3;--line:#363b44;--grid:#2a2e35;--accent:#7aa2ff;--on-accent:#0d1220;--t1:#7aa2ff;--t2:#f0a05a;
--t3:#4fcf95;--t4:#c08ef0;--t5:#ff7a8e;--t6:#4fc8dc;--t7:#9aa1ab;--ok:#5fd39a;--okbg:#173326;--warn:#f0b35a;
--warnbg:#3a2c12;--bad:#ff8a9a;--badbg:#3d1c22;--info:#8cb6ff;--infobg:#1a2a44;--pend:#b7bcc5;--pendbg:#2a2e35}}
:root[data-theme="dark"]{--bg:#15171b;--panel:#1d2026;--text:#e7e9ec;--muted:#a3a9b3;--line:#363b44;--grid:#2a2e35;
--accent:#7aa2ff;--on-accent:#0d1220;--t1:#7aa2ff;--t2:#f0a05a;--t3:#4fcf95;--t4:#c08ef0;--t5:#ff7a8e;--t6:#4fc8dc;
--t7:#9aa1ab;--ok:#5fd39a;--okbg:#173326;--warn:#f0b35a;--warnbg:#3a2c12;--bad:#ff8a9a;--badbg:#3d1c22;
--info:#8cb6ff;--infobg:#1a2a44;--pend:#b7bcc5;--pendbg:#2a2e35}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",
Roboto,"Helvetica Neue",Arial,sans-serif}
header,main,footer{max-width:1120px;margin:0 auto;padding:0 16px}
header{padding-top:28px;padding-bottom:8px;border-bottom:1px solid var(--line)}
.kicker{margin:0;color:var(--muted);font-size:13px;letter-spacing:.04em;text-transform:uppercase}
h1{font-size:28px;line-height:1.2;margin:6px 0 10px}
h2{font-size:22px;margin:0 0 8px;padding-top:6px}
h3{font-size:16px;margin:22px 0 8px}
h4{font-size:14px;margin:18px 0 6px}
.meta{display:flex;flex-wrap:wrap;gap:6px 18px;color:var(--muted);font-size:13.5px}
.meta b{color:var(--text);font-weight:600}
.status{margin:10px 0;padding:8px 12px;background:var(--panel);border:1px solid var(--line);border-radius:8px}
nav{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:13.5px;margin:10px 0 6px}
a{color:var(--accent)}
section{padding:26px 0 18px;border-bottom:1px solid var(--line)}
.lead{color:var(--muted);margin:0 0 10px}
.small,.cnote{font-size:13px;color:var(--muted)}
code{font:12.5px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--grid);padding:1px 4px;
border-radius:4px;overflow-wrap:anywhere}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--pend);border-radius:8px;
padding:12px 14px}
.card.yes{border-left-color:var(--ok)}.card.partly{border-left-color:var(--warn)}.card.no{border-left-color:var(--bad)}
.card h3{margin:6px 0 4px;font-size:15px}.card p{margin:0;font-size:14px}
.chip{display:inline-block;white-space:nowrap;font-size:11.5px;font-weight:600;padding:1px 8px;border-radius:999px;
text-transform:uppercase;letter-spacing:.03em;background:var(--pendbg);color:var(--pend)}
.chip.yes{background:var(--okbg);color:var(--ok)}.chip.partly,.chip.medium{background:var(--warnbg);color:var(--warn)}
.chip.no,.chip.high{background:var(--badbg);color:var(--bad)}.chip.low{background:var(--infobg);color:var(--info)}
.callout{border-radius:8px;padding:10px 14px;margin:12px 0;border:1px solid var(--line)}
.callout p{margin:6px 0 0}
.callout.warn{background:var(--warnbg)}.callout.info{background:var(--infobg)}.callout.pending{background:var(--pendbg);
border-style:dashed}
.tablewrap{overflow-x:auto;margin:8px 0}
table{border-collapse:collapse;width:100%;font-size:13.5px;background:var(--panel)}
th,td{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{background:var(--grid);font-weight:600}
table.num td{font-variant-numeric:tabular-nums}
table.kv td:first-child{width:38%;color:var(--muted)}
table.fixes td{font-size:13px}
.sub{color:var(--muted);font-size:12px}
.na{color:var(--muted)}
.n{font-variant-numeric:tabular-nums}
.grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
.grid1{display:grid;grid-template-columns:minmax(0,1fr);max-width:820px}
figure.chart{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
figure.chart.wide{max-width:none}
figure.chart.small svg{max-width:420px}
figcaption{font-weight:600;font-size:13.5px;margin-bottom:4px}
.svgchart{width:100%;height:auto;display:block}
.svgchart text{font-family:inherit}
.grid{stroke:var(--grid)}.axis{stroke:var(--muted)}
.tick{fill:var(--muted);font-size:11px}.axlabel{fill:var(--muted);font-size:11.5px}
.ref{stroke:var(--muted);stroke-dasharray:3 3}.reflabel{fill:var(--muted);font-size:10.5px}
.line{fill:none;stroke-width:2}.pt{stroke:var(--panel);stroke-width:1}
.t1{stroke:var(--t1);fill:var(--t1)}.t2{stroke:var(--t2);fill:var(--t2)}.t3{stroke:var(--t3);fill:var(--t3)}
.t4{stroke:var(--t4);fill:var(--t4)}.t5{stroke:var(--t5);fill:var(--t5)}.t6{stroke:var(--t6);fill:var(--t6)}
.t7{stroke:var(--t7);fill:var(--t7)}
polyline.line{fill:none}
rect.bar{stroke:none}
.whisk{stroke:var(--text);stroke-width:1.2;fill:none}
.vlabel{fill:var(--text);font-size:10.5px}
.inbar{fill:#fff;font-size:11.5px;font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12.5px;color:var(--muted);margin-top:4px}
.key{display:inline-flex;align-items:center;gap:6px}
.sw{display:inline-block;width:16px;height:4px;border-radius:2px;background:currentColor}
.sw.t1{color:var(--t1)}.sw.t2{color:var(--t2)}.sw.t3{color:var(--t3)}.sw.t4{color:var(--t4)}.sw.t5{color:var(--t5)}
.sw.t6{color:var(--t6)}.sw.t7{color:var(--t7)}
.sw.dash{background:repeating-linear-gradient(90deg,currentColor 0 5px,transparent 5px 8px)}
.hcell{fill:var(--accent)}.hin{fill:var(--on-accent);font-size:12px;font-weight:600}.hout{fill:var(--text);font-size:12px}
.pbox{fill:var(--panel);stroke:var(--line);stroke-width:1.5}
.pbox.enc{stroke:var(--t1);stroke-width:2}.pbox.head{stroke:var(--t3)}.pbox.frz{stroke:var(--t1)}.pbox.out{stroke:var(--t2)}
.ptitle{fill:var(--text);font-size:13.5px;font-weight:600}.psub{fill:var(--muted);font-size:12px}
.pgroup{fill:none;stroke:var(--t3);stroke-dasharray:5 4}.pgroup.eval{stroke:var(--t2)}
.pglabel{fill:var(--muted);font-size:11.5px}
.parrow{fill:none;stroke:var(--muted);stroke-width:1.6}.ahead{fill:var(--muted)}
.verdicts{list-style:none;padding:0}.verdicts li{margin:6px 0}
.issue{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--info);border-radius:8px;
padding:10px 14px;margin:10px 0;break-inside:avoid}
.issue.high{border-left-color:var(--bad)}.issue.medium{border-left-color:var(--warn)}
.ihead{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.ihead h3{margin:0;font-size:15px}
.iid{font-weight:700;color:var(--muted)}
.evidence{margin:6px 0;font-size:13.5px}
.issue p{margin:4px 0;font-size:14px}
.roadmap{list-style:none;padding:0;margin:0;position:relative}
.roadmap li{display:flex;gap:12px;align-items:flex-start;position:relative;padding-bottom:12px}
.roadmap li:not(:last-child)::before{content:"";position:absolute;left:13px;top:28px;bottom:0;width:2px;
background:var(--line)}
.roadmap p{margin:3px 0 0}
.node{width:28px;height:28px;flex:none}.node circle{fill:var(--accent)}.node text{fill:var(--on-accent);font-size:13px;
font-weight:700}
dl.cmds{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;font-size:14px}
dl.cmds dt{color:var(--muted)}dl.cmds dd{margin:0}
details summary{cursor:pointer;color:var(--muted);font-size:13.5px}
footer{padding:18px 16px 40px;color:var(--muted);font-size:12.5px}
@media (max-width:760px){.grid2{grid-template-columns:minmax(0,1fr)}h1{font-size:23px}
dl.cmds{grid-template-columns:1fr}dl.cmds dd{margin-bottom:6px}}
@media print{@page{size:A4;margin:14mm}body{background:#fff;font-size:11pt}nav{display:none}
section{break-before:auto;border:none}#training,#test,#diagnosis,#fixes{break-before:page}
figure.chart,table,.card,.issue,.callout{break-inside:avoid}
header,main,footer{max-width:none;padding:0}a{color:inherit;text-decoration:none}
details>summary{display:none}details{display:block}}
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--summary", default=f"{EVAL_DIR}/training_summary.json")
    p.add_argument("--manifest", default="simulation/public/data/detection/manifest.json")
    p.add_argument("--findings", default="docs/reports/model-all-findings.json")
    p.add_argument("--eval-dir", default=EVAL_DIR)
    p.add_argument("--metadata", default="src/data/encoder_input/all/T24/metadata.json")
    p.add_argument("--out", default="model-all-report.html")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = Inputs(args)
    built = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    page = ReportPage(inputs, built).render()
    out = absolute(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(
        f"wrote {rel(out)} ({len(page.encode()):,} bytes); findings {'present' if inputs.findings else 'MISSING'}; "
        f"frozen-probe evaluation: {inputs.eval_status}{' (' + inputs.eval_reason + ')' if inputs.eval_reason else ''}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
