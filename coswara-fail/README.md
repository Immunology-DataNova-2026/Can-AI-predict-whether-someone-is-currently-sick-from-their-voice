# coswara-fail · respiratory infection detection ❌

**Question I was asking:** can a model tell if someone's currently sick (like a cold or COVID) just from a cough/breath/voice clip?
**What I actually got: AUC ~0.54 across independent datasets. Basically a coin flip.**

I trained this on Coswara (2,746 people, symptomatic vs healthy-asymptomatic) with a full stacked ensemble — per-audio-type RF + gradient boosting, a ResNet CNN on mel-spectrograms, a wav2vec2 embedding model. Looked amazing in-distribution. Fell apart the second I actually checked it properly:

| Evaluation | AUC | What it means |
|---|---|---|
| Random split (in-distribution) | 0.835 | looks like a win |
| **Metadata only (no audio!)** | **≈0.93** | yeah, it's a confound |
| Temporal holdout (within Coswara) | 0.669 | some real signal, weak |
| Cross-dataset → COUGHVID / Sound-Dr | 0.54 / 0.53 | **chance** |
| Leave-one-dataset-out (mean) | 0.545 | doesn't generalize |

## Why this one didn't work
The model was picking up on which dataset/recording batch a clip came from, not whether the person was actually sick — I proved this by training a model on literally no audio (just age, sex, date) and it matched the "real" model. A cold/COVID barely changes your voice, and every crowd-sourced dataset here was collected differently enough that the model just learns the collection artifact instead. I tried harmonizing labels, standardizing features per dataset, rebalancing — none of it moved the needle off ~0.54.

## What's in here
Kept as the original multi-stage pipeline since this one predates the single-file setup I used everywhere else:
`config.py · preprocessing.py · features.py · splitting.py · training.py · evaluation.py · analysis.py`
Outputs are in `data/` and `reports/`; the full trained ensemble (~928MB, Git LFS) is in `models/`. To run: `python preprocessing.py index`, then `features.py extract/embed`, `training.py train`, `evaluation.py evaluate`, and the honesty checks `analysis.py confound / datesplit / crossdataset / leaveoneout`.
