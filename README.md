# Can AI tell if you're sick from your voice?

This is my project testing whether ML can actually detect disease from voice recordings. I wasn't just trying to get a good accuracy number — I wanted to know what number actually holds up once you stop cheating yourself with bad evaluation (random splits that leak, datasets with a confound baked in, features that basically already contain the answer). So every result here got put through subject-independent splits and cross-dataset tests before I trusted it.

One folder per experiment, named for how it turned out. Each one is self-contained — just a `pipeline.py`, a `predict.py`, a trained model in `models/`, and its own README.

## The experiments

| Folder | Task | Honest AUC | Outcome |
|---|---|---|---|
| `coswara-fail/` | "am I currently sick?" (respiratory) | **0.54** | ❌ recording-date confound |
| `voiced-fail/` | voice disorders (small corpus) | **0.65** | ⚠️ corpus too small |
| `parkinsons-uci-fail/` | parkinson's (31-speaker set) | **0.71** | ⚠️ too small / leakage demo |
| `parkinsons-success/` | **parkinson's (252-speaker set)** | **0.83** | ✅ **works** |
| `svd-success/` | **voice disorders (Saarbrücken)** | **0.86** | ✅ **works** |

## What I actually found
Voice models work when the disease leaves a **permanent physical mark** on how you talk — Parkinson's changes your motor control, laryngeal disorders change the vocal cords themselves. They don't work when you're asking about something **temporary**, like a current cold, because that signal is way too faint and gets swamped by whatever confound is hiding in the dataset (recording date, device, whatever). The three failures below each show a different way this breaks: a date confound, a corpus too small with features that secretly leak the label, and straight-up data leakage.

## Run it
```bash
pip install -r requirements.txt
python parkinsons-success/pipeline.py   # parkinson's      -> AUC 0.83
python svd-success/pipeline.py          # voice disorders  -> AUC 0.86
python voiced-fail/pipeline.py          # the 0.65 ceiling
python parkinsons-uci-fail/pipeline.py  # the leakage demo
```
Each `pipeline.py` runs the whole thing (acquisition → preprocessing → EDA → feature engineering → stats → modeling → interpretation → validation), prints the metrics, and saves the model. `coswara-fail/` still has its original multi-stage pipeline (`config.py`, `preprocessing.py`, …, `analysis.py`) since that one predates the single-file setup.

## Grab a trained model
Every folder ships a portable model in `models/` (Git LFS):
```bash
python svd-success/predict.py  a_vowel.wav        # audio -> healthy / pathological
python parkinsons-success/predict.py features.csv # features -> healthy / parkinsons
```
The `.pkl` only needs `joblib` + `scikit-learn` to load (plus `librosa`/`opensmile` if you're feeding it audio). `coswara-fail/models/` has the full respiratory model set too.

## Datasets
Raw data and feature caches are in `external_datasets/` (git-ignored, ~62GB); credits in `THIRD_PARTY_LICENSES/`. Using Sakar 2018 & Oxford Parkinson's (UCI), Saarbrücken Voice Database (Zenodo mirror), VOICED (PhysioNet), and Coswara/COUGHVID/Sound-Dr.
