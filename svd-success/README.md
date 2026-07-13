# svd-success · voice-disorder detection (Saarbrücken / SVD) ✅

**Question:** can a model detect an organic voice disorder from sustained vowels?
**What I got: AUC 0.86, ~80% accuracy — the best result in the whole project.**

| Model (speaker-independent, vowels aggregated per speaker) | Per-record AUC | Speaker AUC |
|---|---|---|
| **SVM-RBF** | 0.76 | **0.861** |
| Random Forest | 0.76 | 0.842 |
| Logistic regression | 0.74 | 0.823 |

## Why this one actually worked
Organic voice disorders (dysphonia, vocal-fold paralysis, laryngitis, neoplasia) mess with the vocal cords **directly**, so the acoustic signature is big and permanent, not subtle or temporary. SVD recorded everyone — patients and controls — on the exact same standardized 50kHz setup, so there's no device confound to accidentally learn. Averaging each speaker's ~14 vowel recordings pushed AUC from 0.78 (single vowel) up to **0.86**. And this is using **only acoustic eGeMAPS features** — no VHI/RSI clinical scores, which is the shortcut that inflates a lot of the published numbers on this exact dataset (see the small-corpus version that failed without that shortcut: `../voiced-fail/`). 1,119 speakers total.

## Run it
```bash
python svd-success/pipeline.py            # full pipeline -> AUC 0.86, saves the model
python svd-success/predict.py a_vowel.wav # score a new recording
```
Model: `models/svd_voice_disorder_svm.pkl` (Git LFS). Data/cache in `../external_datasets/SVD/`.
