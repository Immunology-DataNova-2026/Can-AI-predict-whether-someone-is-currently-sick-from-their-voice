# parkinsons-success · Parkinson's detection (Sakar 2018) ✅

**Question:** can a model detect Parkinson's from a sustained vowel?
**What I got: AUC 0.83, ~80% accuracy — real, strong, no leakage.**

| Model (subject-independent 5-fold CV) | AUC | Bal. acc | Accuracy | Recall (PD) |
|---|---|---|---|---|
| **Random Forest** | **0.827** | 0.704 | 0.804 | 0.908 |
| SVM-RBF | 0.804 | 0.702 | 0.796 | 0.894 |
| Logistic regression | 0.803 | 0.720 | 0.779 | 0.840 |

## Why this one actually worked
Parkinson's causes **hypokinetic dysarthria** — a permanent motor signature (monotone pitch, breathiness, imprecise articulation, tremor) that shows up in literally every phonation, because it's rooted in basal-ganglia damage, not something transient. That plus controlled clinical recording and real clinician labels means the signal actually holds up. 252 speakers, 752 acoustic features, evaluated by speaker so there's zero leakage. I also print the leaky recording-split (AUC 0.955) right in the pipeline so you can see for yourself how much the "~95%" everyone quotes is inflated by leakage. The 31-speaker corpus that was too small to prove this is in `../parkinsons-uci-fail/`.

## Run it
```bash
python parkinsons-success/pipeline.py             # full pipeline -> AUC 0.83, saves the model
python parkinsons-success/predict.py features.csv # score a 752-feature vector
```
Model: `models/parkinsons_sakar_rf.pkl` (Git LFS). Data in `../external_datasets/Parkinsons-Sakar/`.
