# Work Breakdown Structure (WBS) — Thesis Proposal

## RoadFM-Lite: A Self-Supervised Foundation Model for Road-Network-Grounded Trajectory Representation with Application to Sybil Attack Detection

**Student:** Bibek Gupta
**Degree:** Master's Thesis Proposal

---

## 1. Problem Formulation

- [x] 1.1 Define the problem statement (Sybil attacks in vehicular crowdsensing)
- [x] 1.2 Articulate the research gap (trajectory-only models ignore road-network structure)
- [x] 1.3 Formulate the central hypothesis
- [x] 1.4 Define the research question
- [x] 1.5 Write Motivation / Problem Statement section

---

## 2. Literature Review

- [x] 2.1 Survey trajectory foundation models (TrajFM, UniTraj, TimesNet)
- [x] 2.2 Survey Sybil attack detection methods in vehicular crowdsensing
- [x] 2.3 Review graph neural networks for road networks (GraphSAGE, GAT)
- [x] 2.4 Review self-supervised pretraining for spatiotemporal data
- [x] 2.5 Review few-shot and contrastive learning techniques
- [x] 2.6 Identify and document research gaps
- [x] 2.7 Write Related Work section

---

## 3. Proposed Architecture & Methodology

- [x] 3.1 Design two-branch architecture (Road Network Encoder + Trajectory Encoder)
- [x] 3.2 Define Branch 1: GraphSAGE road-network encoder and node features
- [x] 3.3 Define Branch 2: Transformer-based trajectory encoder with fusion mechanism
- [x] 3.4 Define pretraining objectives (masked reconstruction + physical constraint prediction)
- [x] 3.5 Define downstream Sybil detection approach (few-shot contrastive + zero-shot retrieval)
- [x] 3.6 Create architecture diagram
- [x] 3.7 Write Proposed Architecture section
- [x] 3.8 Write Pretraining Objectives section
- [x] 3.9 Write Downstream Task section

---

## 4. Baselines & Evaluation Plan

- [x] 4.1 Define baselines (TimesNet trajectory-only, Feature Engineering + RF, TrajFM)
- [x] 4.2 Define evaluation metrics (Precision, Recall, F1, AUROC)
- [x] 4.3 Define few-shot analysis protocol (5, 10, 20 labeled examples)
- [x] 4.4 Define cross-city transfer experiment design
- [x] 4.5 Define ablation study plan
- [x] 4.6 Write Baselines section
- [x] 4.7 Write Evaluation Plan section

---

## 5. Datasets

- [x] 5.1 Document VeReMi Extension dataset (primary evaluation)
- [x] 5.2 Document real-world trajectory datasets (NGSIM / HighD / OpenDD)
- [x] 5.3 Document OpenStreetMap road network data
- [x] 5.4 Document SUMO-simulated cities for cross-city transfer
- [x] 5.5 Write Datasets section

---

## 6. Timeline & Deliverables

- [x] 6.1 Define semester-wise timeline (2 semesters)
- [x] 6.2 Define expected contributions
- [x] 6.3 Define success criteria
- [x] 6.4 Describe connection to Sybil Guard project
- [x] 6.5 Identify publication target
- [x] 6.6 Write Timeline section
- [x] 6.7 Write Expected Contributions section

---

## 7. Proposal Document Assembly

- [x] 7.1 Write abstract
- [x] 7.2 Compile all sections into proposal document
- [x] 7.3 Format according to department/university template
- [x] 7.4 Add references (IEEE style)
- [x] 7.5 Review and revise draft
- [ ] 7.6 Get advisor feedback and incorporate revisions
- [ ] 7.7 Final proofread
- [ ] 7.8 Submit proposal
