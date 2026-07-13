# voiced-fail · VOICED voice disorders ⚠️

**Question:** can I detect a voice disorder from one sustained "aaah" on a small dataset?
**What I got: AUC ~0.65 — real, but weak, and it wouldn't move no matter what I tried.**

VOICED is a clean dataset (208 speakers: 57 healthy, 151 pathological, all recorded on the same 8kHz device), but honest speaker-independent classification capped out around 0.65 across **twelve** different feature setups I threw at it:

| Approach | AUC | Note |
|---|---|---|
| eGeMAPS (88 clinical features) | 0.61–0.66 | best I got; caps at ~0.66 |
| ComParE (6,373 features) | 0.49 | overfits, only 208 samples |
| wav2vec2 embeddings | 0.49 | a vowel has no phonetic content for it to grab |
| CPPS + Praat jitter/shimmer/HNR | 0.64 | even the gold-standard markers cap out |
| **+ VHI/RSI clinical scores** | **0.77** | **cheating — this basically encodes the label** |

## Why this one didn't work
Not enough healthy speakers (57), and the pathological cases are a mixed bag of mostly mild stuff. Every individual clinical marker tops out around 0.66 and combining them doesn't help. That 0.85-0.95 you see in papers on this dataset comes from throwing in the VHI/RSI questionnaire scores — which are basically a restatement of the diagnosis, so of course it looks great. Same exact task on the Saarbrücken dataset (10x more speakers) gets 0.86 honestly — see `../svd-success/`. So it's a scale problem, not a method problem.

## Run it
```bash
python voiced-fail/pipeline.py             # full pipeline (~0.65), saves the model
python voiced-fail/predict.py a_vowel.wav  # score a new recording
```
Model: `models/voiced_egemaps_logreg.pkl` (Git LFS). Data/cache in `../external_datasets/VOICED/`.
