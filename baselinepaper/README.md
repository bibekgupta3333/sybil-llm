# `baselinepaper/` — an empirical paper from the baseline studies

Working folder for a short paper on reconstruction-based (autoencoder) anomaly detection for Sybil detection on
VeReMi-Extension, built from the baseline studies in `task/` and `src/baselines/plausibility/`. Started 2026-11-10.
The one-protocol rerun is implemented but not finished (see `experiments/`); the numbers quoted are side experiments, not thesis
results (RULE 1). The work is outside the adopted thesis plan (RULE 7) and proceeds on the user's request.

| Path | What it is |
|---|---|
| `01-our-studies/inventory.md` · `results.json` | Every baseline study we ran: data, split, model, protocol, metrics, with the file each number was read from; what is comparable; what is missing |
| `02-literature/papers.json` · `papers.csv` | Merged list, **76 unique papers** (79 records, 3 duplicates across slices) |
| `02-literature/papers_A.json` · `notes_A.md` | 30 records: learning-based misbehaviour / Sybil detection on VeReMi and VeReMi-Extension, with split protocols |
| `02-literature/papers_B.json` · `notes_B.md` | 21 records: autoencoder / one-class detection in V2X, CAN bus, trajectories |
| `02-literature/papers_C.json` · `notes_C.md` | 28 records: known failure modes and evaluation pitfalls of deep anomaly detection |
| `03-gap-analysis/gap-analysis.md` | Literature vs our work: gaps we can fill, weaknesses in our own evidence, items to check before citing |
| `04-paper-plan/paper-plan.md` | Title, questions, enhancements E1–E10 with priority and effort, outline, schedule, open decisions |
| `05-protocol/protocol.md` · `hyperparameters-*.md` | Evaluation protocol for the one-protocol rerun (E1–E5), fixed before results, with the pre-committed reading rules and a log of changes; frozen hyperparameters |
| `06-reviews/` | Reviews of the protocol before any run: methods (researcher + peer reviewer) and framing (advisor + editor) |
| `experiments/` | Python package for the rerun: cached dataset + verification, detectors (convolutional / LSTM / supervised autoencoder, PCA, Isolation Forest, one-class SVM, kNN, logistic regression, length-only), runner, evaluation, aggregation, tests; `runs/` and `cache/` gitignored. State 2026-11-11: classical sweep 57 / 65 runs (8 one-class SVM runs missing), neural sweep 2 / 45 runs (two trial runs), training stopped by the user; the remaining runs will be done later in the cloud (user, 2026-11-11); no detector result aggregated |
| `07-paper/` | IEEE conference draft in LaTeX (`main.tex`, `sections/`, `references.bib`, `main.pdf`, Overleaf zip); results are placeholders |
| `paper-slides.html` | 20-slide research-paper style deck of the paper draft, presenting the ten trained baseline models as the study (introduction, related work with 15 papers, questions, methodology with pipeline and autoencoder diagrams, preliminary findings and results, discussion, conclusion, status and decisions, suggested additions (proposals), references), dated 12 November 2026; self-contained, open in a browser (arrows / space / Home / End, `N` = speaker notes, light / dark, prints one slide per page); no one-protocol result in it |
| `04-paper-plan/venues.md` | Venues with deadlines read from the official pages on 2026-11-10 |

Verification depth of the literature: bibliographic data of every record was checked online; 22 of the 51 vehicular
papers were read in full, the rest at abstract level (each record's `verified_via` says which).
