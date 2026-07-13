# 15 Date-Stratified (Repeated Temporal Holdout) Evaluation - Binary Only

Pooled across block counts [6, 8, 10, 12, 15] (51 block-holdout estimates). Each block:
train on all other blocks, test on a chronological block never seen in training.

## Pooled results
- Mean accuracy: 0.671 +/- 0.186  (95% CI 0.279-0.902)
- Mean block-own-baseline accuracy: 0.596 +/- 0.357
- Mean balanced accuracy: 0.556 +/- 0.078  (95% CI 0.445-0.703)  [0.5 = chance]
- Mean AUC: 0.669 +/- 0.121  (95% CI 0.453-0.865)  [0.5 = chance]
- t-test on pooled block AUCs vs chance: t=9.840, p=0.0000 (**not independent samples**)
- t-test on the 5 per-configuration mean AUCs vs chance (more defensible): t=19.523, p=0.0000
- Blocks beating their own baseline accuracy: 26/51

## Per-configuration summary

| n_blocks | mean accuracy | mean balanced accuracy | mean AUC |
|---|---|---|---|
| 6 | 0.590 | 0.540 | 0.633 |
| 8 | 0.656 | 0.573 | 0.671 |
| 10 | 0.675 | 0.573 | 0.663 |
| 12 | 0.675 | 0.535 | 0.675 |
| 15 | 0.705 | 0.557 | 0.681 |

## How to read this
- Pooled mean AUC CI excluding 0.5 (and p<0.05) => real signal beyond the date shortcut.
- CI straddling 0.5 => earlier random-split accuracy was inflated by the recording-date confound.