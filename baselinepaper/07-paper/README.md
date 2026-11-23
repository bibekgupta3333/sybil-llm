# `07-paper/` — IEEE conference draft (LaTeX)

First full draft, 2026-11-10 (preliminary table updated 2026-11-11 with the three single-seed studies
`task/autoencoder_benign_t48`, `task/autoencoder_t64`, `task/autoencoder_t32`; updated 2026-11-18: every autoencoder
cell of Table VII as mean ± sd over seeds 0–3, the one-class masked autoencoder `task/masked_autoencoder` (TabMlp;
method §IV-E, results §V-G, Tables VII–VIII; TabResnet marked `% TODO TabResnet`, its runs are not finished), the
attack-weighted length-matched AUROC with masked-AE and one-seed TimesNet examples in §III-D, §V-G, §VI-C; 15 pages),
written so the shape of the paper can be seen. **The one-protocol experiments are not
finished**: results that do not exist are red placeholders, numbers from the earlier single-seed studies are blue and
sit in one "preliminary" subsection that will be removed. Set `\draftfalse` in `main.tex` to switch the colours off.

| File | What it is |
|---|---|
| `main.tex` | IEEEtran conference skeleton, draft macros (`\pending`, `\prelim`, `\todo`, `\tbd`), author block |
| `sections/00-abstract.tex` … `07-conclusion.tex` | one file per section |
| `references.bib` | 79 entries (3 added 2026-11-18: `zaurin2023widedeep`, `widedeepdocs`, `wu2023timesnet`), keys fixed in `citation-keys.md`; unconfirmed fields carry a `% CHECK:` line |
| `references-check.md` | entries to check and metadata that differs from the literature list |
| `main.pdf` | last local build |
| `baseline-paper-overleaf.zip` | upload to Overleaf: New Project → Upload Project; compiler pdfLaTeX, main document `main.tex` |

Local build: `tectonic -X compile main.tex` in this folder.

Open items before submission: fill the result tables from `../experiments/results/`, cut from 10 pages to the venue
limit, resolve every `\todo`, advisor name and emails, full-text checks listed in `../03-gap-analysis/gap-analysis.md` §6.
