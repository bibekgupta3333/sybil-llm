# references.bib — check list

`references.bib` has 76 entries (one per key in `citation-keys.md`, no key renamed). Metadata was looked up on
2026-11-10 from Crossref (DOI records), the arXiv API, and the PMLR / NeurIPS / USENIX / IFIP proceedings pages.
DBLP could not be queried (bot challenge), so nothing here rests on DBLP. Test compile: `tectonic`, `IEEEtran.bst`,
`\nocite{*}` → 76 `\bibitem`s, 0 BibTeX warnings, 0 errors.

## 1. Entries with a `% CHECK:` note (8)

| Key | What is not confirmed | What the entry does |
|---|---|---|
| `heijden2018veremi` | LNICST series volume (papers.json says 254; Crossref has no volume) | volume left out; pages 318–337 and DOI 10.1007/978-3-030-01701-9_18 are from Crossref |
| `mahmoudi2020reliable` | series volume (Advances in Intelligent Systems and Computing) | volume left out; book title "Vehicular Ad-hoc Networks for Smart Cities", pages 73–86 from Crossref |
| `fdil2024sybil` | series volume (Lecture Notes in Networks and Systems) and the conference name behind the book "Intelligent Computing" | volume left out; pages 428–437 from Crossref |
| `franceschini2025line` | pages (papers.json says 64–71) and DOI | both left out; title, authors, venue (WONS 2025, Hintertux, 27–29 Jan 2025) confirmed from the IFIP digital library index; URL = the IFIP PDF |
| `longari2022candito` | LNCS volume | volume left out; see the disagreement in §2 |
| `wu2021current` | volume / number / pages of the final TKDE issue | left out: the Crossref record still shows early access (pages 1-1, year 2021); papers.json says the issue is 2023 |
| `devika2024vadgan` | first author name form | Crossref form kept: family "S", given "Devika" (prints "D. S") |
| `vishwanath2026federated` | author name forms | written "Vishwanath, B. and Reddy, P. Chandrasekhar" (papers.json form); Crossref has family "Reddy", given "Chandrasekhar P." |

## 2. Looked-up metadata that disagrees with papers.json (title / year / venue)

| Key | papers.json | Looked up | Used in the .bib |
|---|---|---|---|
| `sedar2024knowledge` | year 2024, IEEE TVT (no volume) | IEEE TVT vol. 74, no. 1, pp. 425–440, **2025** (DOI suffix 2024 = online year) | 2025 with volume / number / pages |
| `demir2025distributed` | venue arXiv:2505.23706 only | published: Proc. 2025 ACM Workshop on Wireless Security and Machine Learning (WiseML), pp. 26–31, DOI 10.1145/3733965.3733967 | `@inproceedings` (WiseML 2025) + eprint |
| `kalogiannis2026pampos` | arXiv 2605.06833, "accepted at WiseML 2026" | published: Proc. 2026 ACM WiseML, pp. 37–42, DOI 10.1145/3811880.3815104 | `@inproceedings` + eprint |
| `longari2022candito` | arXiv 2208.06628, year 2022 | published: Cyber Security, Cryptology, and Machine Learning (CSCML), LNCS, Springer, **2023**, pp. 135–150, DOI 10.1007/978-3-031-34671-2_10; author order there is Longari, Pozzoli, Nichelini, Carminati, Zanero (arXiv: Longari, Nichelini, Pozzoli, …) | published version (2023, its author order) + eprint; switch to `@misc` if the arXiv version is the one meant |
| `shahriar2024vehigan` | URL = NSF PAR copy; venue IEEE ICDCS 2024 | ICDCS 2024, pp. 1294–1305, DOI 10.1109/ICDCS60910.2024.00122 (agrees; DOI and pages added). The DOI 10.1145/3745787 in papers.json is the 2025 ACM TCPS extended version (8 authors) | ICDCS 2024 version (7 authors) |
| `yoon2021autoencoding` | "PMLR volume / pages: not verified" | PMLR 139, pp. 12087–12097 (proceedings.mlr.press/v139/yoon21c.html) | added |
| `jacobs2022ai` | "DOI not verified" | DOI 10.1145/3548606.3560609, pp. 1537–1551 (Crossref) | added |
| `flood2024bad` | no DOI | DOI 10.1109/EuroSP60621.2024.00042, pp. 658–675 (Crossref) | added |
| `franceschini2025line` | lists arXiv:2411.07013 as "thesis version" | that arXiv item is a different work (M. Franceschini's MSc thesis, different title, single author) | no eprint on the entry |
| `pang2021deep`, `schmidl2022anomaly`, `jacobs2022ai` | full title with subtitle | Crossref title has no subtitle ("Deep Learning for Anomaly Detection", "Anomaly detection in time series", "AI/ML for Network Security") | papers.json title (matches the paper / arXiv) |

No other title, year or venue differs.

## 3. Author-name differences (for information)

| Key | papers.json | Looked up | Used |
|---|---|---|---|
| `kamel2020veremi` | Rens W. van der Heijden | Crossref record is truncated: "van der Hei, Rens W." | van der Heijden, Rens W. |
| `korba2023federated` | Sidi Mohammed Senouci | Crossref splits it as family "Mohammed Senouci", given "Sidi" | Senouci, Sidi Mohammed |
| `kamel2020simulation`, `kamel2019misbehavior`, `kamel2019misbehaviorb` | Ines Ben Jemaa | Crossref: family "Jemaa", given "Ines Ben" | Ben Jemaa, Ines |
| `youness2025vemisnet` | Mohamed Sobh, Ayman Mohammad Bahaa-Eldin | Crossref: Sobh, Mohamed A.; Bahaa, Ayman M. | Crossref |
| `mahesh2025hyperlstm` | Ali Delham Algarni | Crossref: Algarni, Ali | Crossref |
| `tang2025deep` | Ranesh Kumar Naha | Crossref: Naha, Ranesh | Crossref |
| `han2022adbench` | Mingqi Jiang (also the arXiv API spelling) | NeurIPS proceedings BibTeX: Jiang, Minqi | NeurIPS |
| `campos2025federated` | Enrique Marmol Campos, Aurora Gonzalez-Vidal, Jose L. Hernandez-Ramos | Crossref: Campos, Enrique Mármol; Gonzalez-Vidal, Aurora; Hernández-Ramos, José L. | Crossref (family name "Campos" matches the key) |
| `demir2025distributed` | "Utku Demir et al." | six authors (arXiv + Crossref): Demir, Sagduyu, Erpek, Jafari, Kompella, Xue | full list |
| `yoon2021autoencoding` | Frank Chongwoo Park | PMLR page: Frank Park; arXiv: Frank Chongwoo Park | Park, Frank Chongwoo |
| `pang2021deep`, `gong2019memorizing` | Anton van den Hengel | Crossref: "Hengel, Anton Van Den" / "Van Den Hengel, Anton" | van den Hengel, Anton |

## 4. Conventions

- Conference papers `@inproceedings` (booktitle "Proc. …"), journals `@article`, Springer book chapters without a
  confirmed proceedings name `@incollection`, arXiv-only `@misc` (`hamhoum2024mistralbsm`, `denouden2018improving`,
  `ruff2021rethinking`). `kim2022rigorous` (AAAI) is `@inproceedings` with volume 36, number 7.
- Article-number journals (MDPI, Elsevier, Wiley, Springer IJIS) carry the article number in `pages`.
- `eprint` + `archivePrefix` on every entry with a known arXiv id; `url` only where there is no DOI.
- Acronyms and proper nouns in titles are brace-protected; conference ordinals (35th ICML etc.) come from the
  Crossref / proceedings titles. DOIs are as Crossref returns them (lower case; DOIs are case-insensitive).

## Added 2026-11-18

- `zaurin2023widedeep`: checked on joss.theoj.org (DOI 10.21105/joss.05027) and the library docs citation section.
- `widedeepdocs`: docs page exists (checked 2026-11-18); the code read was the 1.7.0 wheel.
- `wu2023timesnet`: title / authors from arXiv 2210.02186; ICLR 2023 venue not confirmed online (OpenReview blocked), `% CHECK:` line.
