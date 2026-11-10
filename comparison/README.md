# `comparison/`: multi-agent literature scan (Sybil, road network, vehicular crowdsensing)

A literature scan for the request "pull the multi-agent problems related to this project (Sybil, road network or
vehicular crowdsensing), 50+ papers, find a novel problem doable in a week". **Proposal-level material:** nothing
here is a thesis result, and every idea is outside the adopted plan (RULE 7) until the user approves it.

## Files

| File | What it is |
|---|---|
| `papers.json` · `papers.csv` | The merged list: **109 unique papers**, one record each (id, title, authors, year, venue, url, area, multi-agent setting, problem, method, dataset / simulator, metrics + result, code, limitation / gap, relevance 1–5 + why, how verified, which search agent found it) |
| `papers_A.json` · `papers_B.json` · `papers_C.json` | The three search agents' raw lists (kept for provenance) |
| `comparison.md` | Counts, a "datasets used" table, one comparison table per area, common gaps |
| `novel-ideas.md` | 7 candidate problems with closest papers, data, metric, risks, effort, scores; the recommendation and a 7-day plan |
| `multi-agent-ideas.md` · `multi-agent-ideas.json` | **Multi-agent ideas:** three of the candidate problems reworked as multi-agent systems (MA1 cooperative receivers with lying Sybil reporters, from I1; MA2 Sybil-aware consensus over linked identities, I2 + I3; MA3 LLM orchestrator with specialist tool agents, from I7): agents, protocol, method, algorithm, threats, experiments, solution, risks, scores. Proposals only. Shown with animated agentic and methodology diagrams in the `comparison.html` section "Multi-agent ideas" |
| `comparison.html` | Self-contained page (works from `file://`, light / dark toggle): sortable / filterable paper table, counts, gaps, idea scorecard, plan |

## Counts per area

| Area (search agent) | Papers | 2019+ | Code public | Relevance 4–5 |
|---|---|---|---|---|
| A · Sybil / misbehaviour detection in VANET / V2X (collaborative, federated, trust, blockchain, MARL, graph, LLM) | 36 | 36 | 3 | 14 |
| B · Road network / traffic (MARL signal control, cooperative perception and its attacks, trajectory-prediction security, LLM traffic agents) | 36 | 34 | 17 | 7 |
| C · Vehicular / mobile crowdsensing (Sybil / fake workers, truth discovery, trust, incentives, MARL scheduling, FL) | 37 | 29 | 1 | 6 |
| **Total** | **109** | 99 | 21 | 27 |

110 records came in; one duplicate (Ghost Riders, Sybil attacks on Waze: B35 = C19) was merged and kept as B35.

## How papers were found and verified

- Three search agents (A, B, C) each covered one area with disjoint focus, using web search plus publisher /
  index pages: arXiv, IEEE Xplore, ACM DL, Springer, MDPI, Semantic Scholar API, Crossref / OpenAlex DOI records.
  Each record names its source in `verified_via` (B and C) or via its URL (A).
- Rule: only papers found online with a working URL (DOI, arXiv or publisher page); fields that the readable source
  did not state are written "unknown" rather than guessed. Most details come from abstracts, so methods and results
  are as the abstracts state them.
- Merge (agent S): records unified to one schema, deduplicated by normalised title and by DOI / arXiv id.
- Spot check (agent S, 2026-11-01): 10 random records re-fetched (B02, A11, A03, C04, B04, B19, B13, B32 on arXiv;
  C36 and C29 through Crossref). All 10 resolve to the stated title, first author and year. Note: `doi.org` links to
  the ACM DL return HTTP 403 to scripts (bot block), but the DOIs are valid in Crossref.

## Known weaknesses of the list

- Four area-A records were verified from title / index pages only; their method and result fields say so (A27, A29,
  A30, A34: "details not verified beyond title" or "abstract not retrieved").
- Ten records are older than 2019 (B33, B35, C08, C15, C16, C18, C23, C24, C29, C33); they are kept as the
  foundational work of their line.
- The "datasets used" table is a keyword classification of free text and is approximate.

## Where to read next

`novel-ideas.md` (recommendation: **I1, cooperative plausibility fusion across receivers with lying Sybil reporters**,
built on `src/baselines/plausibility/`, 6–7 days), then `comparison.md` for the evidence.
