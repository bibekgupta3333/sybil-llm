# agent.md — start here (current state of the project)

> **Every agent reads this first, then root `CLAUDE.md` (rules 1–7).** It says what the project is *now*,
> which files are live vs. legacy, what is decided, and what waits on the user. **Keep it current** (see the
> last section). Last updated: **2026-10-06**.

## 1. The project in five lines

- Master's thesis (Florida Polytechnic): **RoadFM-Lite**, self-supervised trajectory model for **Sybil attack
  detection** on the local **VeReMi-Extension** data (4 Sybil attacks × 2 scenario groups, 0709 / 1416).
- **Adopted method (2026-10-05):** a **TimesNet** encoder pretrained **without attack labels** (Stage 1:
  masked reconstruction + injected-violation physics heads P1–P3 + SimCLR contrastive), then few-shot
  fine-tuning + memory-bank anomaly scoring (Stage 2, **deferred**).
- The method **replaces** the proposal's Transformer + MTR/TCP design. Deviating from `proposal/main.tex` is
  **accepted** (decision D9) — do not flag it as an issue.
- **Nothing of the adopted plan is implemented yet.** Planning is done; code starts in `src/`.
- All previous model numbers (v1 pilot) are **invalid as evidence** (F1 leak, F2 TCP bug) — history only.

## 2. What is live, what is legacy

| Path | Status | What it is |
|---|---|---|
| `agent.md` (this) | **live** | orientation + current state |
| `CLAUDE.md` | **live** | project rules 1–7 (binding) |
| `docs/plan/clarification.md` | **live — the plan** | adopted plan in the professor's format + why it changed + Q&A |
| `docs/plan/stage1-ssl-wbs.md` | **live — the task list** | findings F1–F10, decisions D1–D9, tasks S1.0–S1.4 (38, ≈ 45 d) and S2.0–S2.3 (9, 14.5 d, deferred), sweep plan |
| `docs/plan/research-plan.md` | **live** | thesis-wide phases 0–8 + "Stage 1 plan" decision record; keep in sync with the WBS |
| `src/` + `src/agent.md` | **live — all new code** | TimesNet plan code; layout, OOP 90/10 coding rules, guardrails, per-task workflow |
| `stage1-plan.html` | **live** | per-task tracker for Stage 1/2 + plan review, diagrams, slides |
| `tracker.html` | **live** | whole-thesis tracker (phases, alignment table, critical path) |
| `simulation/` | live tool | TypeScript window simulator (data understanding); see `simulation/README.md` |
| `notebooks/eda_veremi.ipynb`, `benign_vs_attack_maps.ipynb` | live (data understanding) | EDA, raw-data maps |
| `notebooks/refresher_*.ipynb` | study notes | numpy / pandas / pytorch / DL refreshers |
| `docs/research-notes/` | reference | literature + Phase 1 data notes (`data_understanding/`) |
| `models/transformer_model.ipynb`, `models/*.pt`, `models/results/` | **legacy v1** | Transformer pilot; TCP was a no-op; numbers invalid. Do not extend. |
| `scripts/prepare_data.py`, `scripts/export_simulation_sample.py` | legacy v1 / tooling | v1 20 × 13 window prep (+ splice fix), simulator export. House-style reference for new code. |
| `data/` | **protected** (gitignored, 13 GB) | raw `VeReMi-Dataset/` + v1 `prepared_data/` — read-only (RULE 2) |
| `proposal/` | background | original proposal; method superseded (D9) |
| `README.md` | **stale overview** | still describes the v1 method; trust this file + the plan docs |

## 3. Decisions in force (details: WBS §1)

| ID | Decision |
|---|---|
| D1 | Sample = network-heard pseudonym sequence (dedup by `messageID`), **T = 64** (128 sensitivity), fixed length, equal windows per vehicle |
| D2 | "Receiver observed" = receiver's own GPS position/velocity at rcvTime |
| D3 | **13 features**: sender-claimed pos/vel, receiver pos/vel, acl, range, bearing (one wrapped angle, not sin/cos), log-Δτ. **No time of day.** |
| D4 | Physics heads **P1–P3 detect violations we inject** (p = 0.5); real-data rules are diagnostics only |
| D5 | TCP dropped |
| D6 | Contrastive hard negatives: same 50 m grid cell, same group, other pseudonym, ≥ 10 min apart |
| D7 | Normalised losses, λ1 = 1, λ3–λ5 = 0.3, λ2 ∈ {0.1, 0.3, 1}; label-free checkpoint selection |
| D8 | **New split** by (time window, physical vehicle) across the 4 scenarios — RULE 3 change, **adopted, not executed** |
| D9 | Deviation from `proposal/main.tex` accepted |

Stage 2 defaults: train A16 GridSybil, A18 DoSRandom, A19 DoSDisruptive + benign; hold out **A17 DataReplay**;
n ∈ {10, 20, 30, 50}; unlabeled memory bank; K, θ on val; FL = motivation only.

## 4. Facts that trip up new agents

- Beacons are **1 Hz** (not 100 ms). Positions are SUMO **metres**, not lat/lon; there is **no map** in the data.
- Local data has only attack codes **A0, A16–A19** (A16 = GridSybil, A17 = DataReplay). No A1–A15.
- **F1:** the 4 scenarios reuse the same traffic → the v1 split leaks (89.7% of benign test identities have a
  near-copy in train). `scripts/audit_splits.py` passes only because it checks within one scenario.
- **F2:** `models/transformer_model.ipynb` TCP labels are all zero → v1 is MTR-only.
- `GridSybil_0709` pseudonym `1` is a shared sentinel (653 vehicles) — the v1 splice defect.
- Time of day alone separates 0709 from 1416 → never an input.
- Original physics rules: H1 fires only on attackers (label proxy), H2 has no violations > 70 km/h, H3 flags
  33% of benign windows — hence D4.
- Physics heads are **P1–P3**; **H1–H4** are the thesis hypotheses.

## 5. Waiting on the user (do not proceed without an explicit OK)

| Item | Blocks |
|---|---|
| Write `data/prepared_receiver/` (receiver-centric prep) | S1.1.2 onward |
| Add `/src/prepared_data/` to `.gitignore` | any large file in `src/prepared_data/` |
| Install / pin the torch training env (`requirements-train.txt`) | any citable training run (RULE 4, S1.0.4) |
| Optional: download LuST map (`lust.net.xml`) | S1.3.P3 only |
| Optional: download the full 19-attack VeReMi-Extension | original Stage 2 classes only |
| Professor's confirmation of the corrected plan (S1.0.1) | nothing blocks; objections become WBS changes |

## 6. Next steps

1. **S1.1.1** cross-scenario leak audit (read-only) and **S1.2.x** TimesNet encoder — can start now, in `src/`.
2. S1.1.2–S1.1.7 receiver-centric prep + new split — after the `data/` write OK.
3. S1.3 objectives → joint training → S1.4 frozen-encoder evaluation. Then Stage 2.

Follow `src/agent.md` §5 (per-task workflow) for every task.

## 7. Keeping this file current (every agent, every substantive turn)

- When a task completes, a decision is made, or an approval arrives: update §3–§6 here **and** the WBS /
  `research-plan.md` in the same turn, add a line to the changelog below, and remind the user to tick
  `stage1-plan.html` and `tracker.html`.
- If a file's status changes (live → legacy, new doc), update §2.
- Keep it short: this file is orientation, not a duplicate of the plan. Link instead of copying.

## Changelog

- **2026-10-06** — `src/agent.md` added (flat layout, OOP 90/10, guardrails); bearing fixed as one angle
  (13 features); WBS gained Stage 2 tasks + sweep plan; `tracker.html` rewritten for the adopted plan; this file
  created.
- **2026-10-05** — corrected plan adopted (D1–D9); `clarification.md` written; deviation from `main.tex`
  accepted.
- **2026-10-04** — plan review (19 → 18 issues) added to `stage1-plan.html`.
- **2026-09-30** — advisor's TimesNet Stage 1 plan received; WBS + findings F1–F10; F1 leak confirmed.
