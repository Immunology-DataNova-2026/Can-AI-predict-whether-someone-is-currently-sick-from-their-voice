# 16 Cross-Dataset Evaluation - train coswara -> test coughvid

Zero-shot: model never saw a single clip from the test dataset. Harmonized
label = currently symptomatic vs healthy-asymptomatic. Cough clips only.
Metrics are participant-aggregated.

- Train: **coswara** (4296 cough clips)
- Test:  **coughvid** (9992 cough clips)
- Shared numeric features: 109; embedding dim: 768

| model | AUC | balanced acc | accuracy | recall(sick) | f1-macro |
|---|---|---|---|---|---|
| gbm_features | 0.526 | 0.514 | 0.514 | 0.307 | 0.492 |
| embedding | 0.535 | 0.523 | 0.523 | 0.441 | 0.520 |
| ensemble | 0.537 | 0.520 | 0.520 | 0.332 | 0.503 |

Test-set baseline (majority-class) accuracy: 0.500.