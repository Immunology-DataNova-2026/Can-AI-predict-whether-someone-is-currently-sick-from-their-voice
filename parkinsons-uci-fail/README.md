# parkinsons-uci-fail · small Oxford/UCI Parkinson's set ⚠️

**Question:** can I detect Parkinson's from a sustained vowel, on a tiny dataset?
**What I got: AUC ~0.71 — real signal, but the estimate is shaky, and the famous "95%" everyone quotes is leakage.**

This is the classic Oxford/UCI set (Little et al.) — only **31 speakers, 8 of them healthy controls**, 195 recordings, 22 dysphonia features.

| Split | AUC | Accuracy | Note |
|---|---|---|---|
| **Honest — split by subject** | **0.708** | 0.718 | *below* the 0.754 baseline |
| Leaky — split by recording | 0.952 | 0.888 | just memorizes the 31 voices |

## Why this one didn't work
Two things going on. First, the "~95%" everyone cites is **leakage** — each speaker has ~6 recordings, and if you split by recording instead of by person, the model just learns to recognize the individual's voice instead of the disease. Second, it's **too small** — only 8 healthy controls means each CV fold only sees 1-2 of them, so "healthy" barely gets estimated and accuracy actually falls below baseline. The signal is real (Parkinson's does leave a strong signature in the voice), this dataset is just too small to prove it. The bigger Sakar dataset does prove it, honestly, at AUC 0.83 — see `../parkinsons-success/`.

## Run it
```bash
python parkinsons-uci-fail/pipeline.py             # prints both the honest and leaky splits
python parkinsons-uci-fail/predict.py features.csv # score a 22-feature vector
```
Model: `models/parkinsons_uci_rf.pkl` (Git LFS). Data in `../external_datasets/Parkinsons-UCI/`.
