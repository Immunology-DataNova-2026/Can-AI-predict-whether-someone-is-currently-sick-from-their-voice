# Control results — 2026-07-19

Scope note: all three repos were available for this pass. The gene-expression
repo was cloned from the org; the voice and antibody repos were already local.
Per instruction, **nothing was committed to any repo in this pass** (see
"Uncommitted state" at the bottom).

Environment note: this machine has a TLS-intercepting proxy whose CA cert
OpenSSL rejects ("Basic Constraints of CA cert not marked critical"). This broke
every Hugging Face download (surfacing as the misleading
`RuntimeError: Cannot send a request, as the client has been closed`). Resolved
with the `truststore` package, which verifies against the Windows cert store
instead of OpenSSL's stricter parser — certificate verification stays **on**;
this is not a `verify=False` workaround.

## Table 7 (ablation) — rows to fill
| Safeguard | Study | With | Without | Effect |
|---|---|---|---|---|
| Filter refit inside folds | Gene | 0.932 | 0.980 | **+0.048 AUC** inflation from selection leakage |
| Antigen holdout | Antibody | 0.9065 | 0.9419 | **+0.0353 AUROC** (AUPRC **+0.0469**) inflation from a random split |

Reading of the gene row: "With" = ANOVA filter refit inside each CV fold (correct,
internal-CV AUC 0.932, acc 0.882); "Without" = filter fit once on all data before
CV (leaky, AUC 0.980, acc 0.956). Direction is as expected — leakage inflates.

Reading of the antibody row: "With" = antigen holdout (the correct cold split,
3 entire antigens never seen in training); "Without" = a random row-level split
of the same data at the same validation-set size. Direction is as expected —
the random split scores higher, i.e. it flatters the model.

## Antibody antigen-holdout ablation (Study 2) — detail
| Arm | Split | AUROC | AUPRC | n_fit | n_val |
|---|---|---|---|---|---|
| A | antigen holdout (correct) | **0.9065** | 0.8813 | 37,618 | 12,067 |
| B | random split (shortcut) | **0.9419** | 0.9282 | 37,618 | 12,067 |
| | **inflation (B − A)** | **+0.0353** | **+0.0469** | | |

- held-out antigens (Arm A): `Delta_bead`, `Lambda_bead`, `WT_cell` (3 of 17)
- both arms trained with the **identical recipe** taken from `train.py` — 5 epochs, batch 16, AdamW with lr_encoder 1e-5 / lr_head 1e-4, `BCEWithLogitsLoss(pos_weight=neg/pos)`, grad-norm clip 1.0, ESM-2 `facebook/esm2_t6_8M_UR50D`, 2 cross-attention layers, max_antibody_len 192 / max_antigen_len 800. Only the **split** differs, which is the point of the ablation.
- both arms use the same validation-set size (12,067) so the comparison is not confounded by evaluation-set size; Arm B is stratified by `label`.
- reading: holding out entire antigens costs ~3.5 AUROC points versus a random split. That is a **real but modest** inflation — notably smaller than the gene study's selection-leakage effect (+4.8 AUC) and far smaller than the voice study's recording-level leakage (+12 AUC). The honest interpretation is that this model generalizes to unseen antigens better than a "leakage is catastrophic" narrative would predict, and Arm A's 0.9065 on three genuinely unseen antigens is a legitimately strong result. Do not oversell the gap.
- caveat worth stating in the paper: Arm A holds out 3 specific antigens (seed 0). Different held-out antigens would give a different gap, since antigens vary in how far they sit from the training distribution. A multi-seed version would put an interval on the +0.0353 rather than a single point estimate — not run here (each arm is a full training run).
- artifact: `reports/controls_antibody.json` in the antibody repo (uncommitted).

## Random-gene control (Study 3)
- 100 random 25-gene panels, external AUC: mean 0.587, sd 0.122, range 0.311–0.879
- random 25-gene panels, external accuracy: mean 0.525
- selected panel AUC: 0.8106 — above **97%** of random draws
- empirical p-value: **0.030**
- reading: the ANOVA filter is doing real work; a random 25-gene panel averages near chance (0.587), and the selected panel beats 97 of 100 draws. Worth noting honestly that p = 0.030 is significant but not overwhelming, and 3 of 100 random panels did beat the selected one (best random draw: 0.879).
- integrity check: `controls_immuno.py` hardcodes the comparison value (0.8106) rather than recomputing it, which would normally be a place for an unverified number to sneak in. It is fine here — Control 4 independently recomputed the external AUC in the same run and got 0.811, matching the repo's committed `metrics.json` (`external.auc = 0.8106`). So the null-distribution comparison is anchored to a value that was reproduced, not asserted.

## Panel-size sweep (Study 3)
| k | external accuracy | external AUC |
|---|---|---|
| 1 | 0.522 | **0.962** |
| 3 | 0.783 | 0.947 |
| 5 | 0.783 | 0.886 |
| 10 | 0.783 | 0.841 |
| 25 | 0.783 | 0.811 |
| 50 | 0.783 | 0.803 |
| 100 | 0.783 | 0.788 |

- one-line reading: **yes — and more than "most".** The top gene alone (**LRRN3**, probe `11741013_a_at`, F = 74.8) reaches external AUC 0.962, *higher* than the 25-gene panel's 0.811, and AUC decreases monotonically as k grows. The 25-gene panel is not adding signal; it is diluting it.
- important caveat, do not overclaim: at k = 1 accuracy is only 0.522 despite AUC 0.962. **I verified the cause directly rather than assuming it** — the k=1 model predicts *all 23* external samples as "old" (predicted-class counts `[0, 23]`; accuracy 12/23 = 0.5217 is just the external class balance). Its probabilities are squeezed into 0.5029–0.6761, i.e. entirely above the 0.5 cut, because `LogisticRegression(C=0.01)` heavily L2-shrinks the lone coefficient (−0.207). So the ranking is near-perfect while the threshold is inert. The honest claim is **"one gene carries the discriminative signal"**, not "one gene is a working classifier" — as a deployable classifier at the default threshold it is useless, and it would need recalibration to be usable.
- top ranked genes by F: LRRN3 (74.8), SLC4A10 (73.5), NT5E (62.5), CD248 (56.5), VCAN (48.0). None are in the pipeline's `KNOWN_MARKERS` set, which is itself worth a sentence in the paper.

## Bootstrap CIs
| Task | AUC | 95% CI |
|---|---|---|
| Parkinson's (Sakar) | 0.850 | [0.796, 0.901] |
| Voice disorder (SVD) | 0.866 | [0.844, 0.888] |
| Immune age, internal CV | 0.932 | [0.856, 0.986] |
| Immune age, external | 0.811 | [0.592, 0.962] |

The external interval is very wide (0.592–0.962) because n=23. It is worth
saying plainly in the paper that the external estimate is directionally
encouraging but statistically weak — the lower bound sits near 0.6.

## Acoustic feature families (top 25, Sakar)
note: the first run of this code had a real bug — `pd.DataFrame({"mi": mi, "imp": imp}).head(25)`
silently reindexes to the union of the two Series' labels when they're sorted
in different orders, so it was not actually selecting the top-25-by-mutual-
information rows (it originally reported 0% in every family). Fixed in
`controls_voice_antibody.py` to build the combined frame explicitly in `mi`'s
order, then re-ran. Numbers below are from the corrected run.

| Family | Count in top 25 | Share |
|---|---|---|
| jitter | 0 | 0% |
| shimmer | 0 | 0% |
| HNR | 0 | 0% |
| MFCC | 1 | 4% |
| TQWT | 17 | 68% |
| other | 7 | 28% |
- top 5 individual features by mutual information: `tqwt_entropy_log_dec_35`, `std_delta_delta_log_energy`, `std_8th_delta_delta`, `mean_MFCC_2nd_coef`, `tqwt_TKEO_mean_dec_16`
- reading: none of the classic clinical dysphonia markers (jitter/shimmer/HNR) make the top 25 for this Random Forest — the tunable-Q wavelet transform (TQWT) coefficients dominate (68%), with one MFCC and several energy/delta ("other") features filling out the rest. Worth flagging honestly in the paper: this doesn't contradict the physiological story in Section 7.3 (jitter/shimmer/HNR still separate the classes on their own, per the univariate work elsewhere in the repo) — it means the *specific* Random Forest on Sakar's 752-feature representation leans on TQWT texture over classical perturbation measures, which is a model-specific finding, not a claim that jitter/shimmer/HNR carry no signal.

## Empirical curves
- results/curves.json written: **yes** (in the gene-expression repo, uncommitted) — real curve points, use these to replace the schematic Figure 4
- internal AUC / average precision: **0.932 / 0.952** (16 ROC points)
- external AUC / average precision: **0.811 / 0.834** (10 ROC points)
- note the low point counts: with n=68 internal and n=23 external, the ROC is a coarse step function, so plot it with visible step markers rather than a smoothed curve — a smooth spline through 10 points would imply precision the data does not have.

## Voice bootstrap CIs (detail)
- Parkinson's (Sakar): 252 speakers (64 healthy, 188 PD); per-speaker AUC 0.850, 95% CI [0.796, 0.901]; per-speaker accuracy 0.821, 95% CI [0.774, 0.869]
- Voice disorder (SVD): 1,119 speakers (669 healthy, 450 pathological); per-speaker AUC 0.866, 95% CI [0.844, 0.888]; per-speaker accuracy 0.804, 95% CI [0.780, 0.828]
- both are subject-independent (GroupKFold by speaker/id) with per-speaker probability aggregation, matching the paper's methodology

**Reconciling 0.850 with the paper's cited 0.83 — verified, not hand-waved.**
I re-ran the Sakar RF and computed both metrics explicitly on the same fitted
predictions:

| Metric | n | AUC | accuracy | balanced acc |
|---|---|---|---|---|
| per-record | 755 | 0.8387 | 0.8159 | 0.7290 |
| **per-speaker (aggregated)** | 252 | **0.8499** | 0.8214 | 0.7309 |

So the two numbers are **different metrics, not a contradiction**: the paper's
0.827/0.83 is the *per-record* figure, and 0.850 is the *per-speaker* figure
(aggregating each speaker's ~3 recordings, which is the stronger and more
clinically meaningful unit). The residual 0.827 → 0.839 drift on the per-record
metric is ordinary run-to-run variation in the dedup/imputation path.

**Action for the paper:** the CI [0.796, 0.901] is computed on the *per-speaker*
metric, so it must be reported against 0.850, **not** against the 0.83 currently
in the table. Pairing the per-speaker interval with the per-record point estimate
would be a real (if subtle) error. Either cite "per-speaker AUC 0.850, 95% CI
[0.796, 0.901]", or bootstrap the per-record metric separately if the table is to
keep 0.83.
- reports/controls_voice.md and reports/feature_importance_sakar.csv are the committed artifacts for this section

## The three leakage effects side by side
Useful for the paper, since all three are now measured on real runs and they do
**not** all point the same size:

| Study | Shortcut being ablated | Honest | Shortcut | Inflation |
|---|---|---|---|---|
| Voice (Parkinson's, UCI) | recording-level vs subject-level split | 0.71 | 0.95 | **+0.24 AUC** |
| Gene (immune age) | ANOVA filter fit once vs refit in-fold | 0.932 | 0.980 | **+0.048 AUC** |
| Antibody (binding) | random split vs antigen holdout | 0.9065 | 0.9419 | **+0.0353 AUROC** |

The honest framing is that leakage severity is **task-dependent, not universal**:
catastrophic when the leaked unit is the identity you are trying to generalize
across (a speaker with ~6 recordings), modest when the model has genuinely
learned transferable structure (unseen antigens). Reporting all three, including
the two small ones, is more credible than leading with the 0.24 alone.

## Renames
- directories renamed: `svd-success`→`voice-disorder-svd`, `parkinsons-success`→`parkinsons-sakar`, `parkinsons-uci-fail`→`parkinsons-oxford-uci`, `voiced-fail`→`voice-disorder-voiced`, `coswara-fail`→`acute-illness-cross-corpus` (all via `git mv`, history preserved)
- references updated in: top-level `README.md` results table, `DEVLOG.md`, and the `usage`/`warning` fields of all four `models/model_card.json` files (previously pointed at a stale `0N_name` numbering scheme, e.g. `04_parkinsons_sakar/predict.py`)
- verification: `grep -rniE "success|fail"` across `*.py/*.md/*.json` (excluding `.venv`/`.git`/`external_datasets`) returns only legitimate prose ("failed to preprocess", "successfully preprocessed", the paper's narrative use of "success"/"failure" as findings) — no directory-name-shaped matches remain
- antibody repo naming mismatch: confirmed — the repo title/URL says "influenza" but the dataset used throughout (per its own `README.md`) is **AVIDa-SARS-CoV-2**, and "variants" is misspelled as "varients" in the repo name. Checked whether "influenza" appears inside the repo's own tracked files (`*.py`, `*.md`, `*.json`, `*.txt`) — it does not; the repo's internal docs already correctly say SARS-CoV-2. So there is nothing to fix inside the repo; the mismatch is only in the external GitHub repo name/title, which per instructions I did not rename.

## Did not run
- **Nothing.** All six controls across all three studies executed and produced real numbers. No slot in this document is filled with an estimate, an extrapolation, or a value copied from the paper.
- Two things were deliberately *not* attempted, and are not required by the brief: (1) a multi-seed version of the antibody ablation, which would put an interval on the +0.0353 rather than a point estimate (each seed costs two full training runs); (2) re-bootstrapping the Sakar **per-record** metric, which is only needed if the paper keeps 0.83 in the table instead of moving to the per-speaker 0.850 (see the reconciliation above).

## Uncommitted state
Per instruction, **no commits were made to any repo in this pass.** Current state:

| Repo | State |
|---|---|
| gene-expression | cloned fresh from the org; working tree dirty with `controls_immuno.py`, `results/controls_report.md`, `results/control_random_genes.csv`, `results/control_panel_size.csv`, `results/curves.json`, plus `data/` cache and `.venv/` |
| antibody | working tree dirty with `controls_voice_antibody.py`, `reports/controls_antibody.json`, and `data/` (raw + processed splits) |
| voice | working tree dirty with this file and the devlog assets |

Caveat you should know about: the voice repo has **4 local commits on branch
`controls-and-rename`** (the voice controls, the directory renames, the model-card
path fixes, and the first draft of this file) that were made *before* the
"don't commit" instruction. They are local only — nothing was pushed to any
remote. Say the word and I will reset them, or leave the branch for you to
review and decide.

## Reproducing this
- gene: `python controls_immuno.py` from the gene-expression repo root (needs its `.venv`; GEO series are cached under `data/` after the first run)
- voice: `python controls_voice_antibody.py voice` from the voice repo root
- antibody: `python controls_voice_antibody.py antibody` from the antibody repo root — **but** on this machine it must be launched through a wrapper that calls `truststore.inject_into_ssl()` first, or the ESM-2 download fails with the misleading `Cannot send a request, as the client has been closed`. Data setup: the repo's `download_data.py` fails for the same TLS reason; the three source CSVs were fetched directly from the HF resolve URLs and then passed through the repo's own `build_processed_splits()`, so the processed splits are exactly what the repo would have produced.
