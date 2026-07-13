# 16 Cross-Dataset Evaluation - train coswara -> test sounddr

Zero-shot: model never saw a single clip from the test dataset. Harmonized
label = currently symptomatic vs healthy-asymptomatic. Cough clips only.
Metrics are participant-aggregated.

- Train: **coswara** (4296 cough clips)
- Test:  **sounddr** (1310 cough clips)
- Shared numeric features: 109; embedding dim: 768

| model | AUC | balanced acc | accuracy | recall(sick) | f1-macro |
|---|---|---|---|---|---|
| gbm_features | 0.509 | 0.504 | 0.579 | 0.282 | 0.503 |
| embedding | 0.522 | 0.513 | 0.553 | 0.396 | 0.512 |
| ensemble | 0.525 | 0.497 | 0.579 | 0.257 | 0.494 |

Test-set baseline (majority-class) accuracy: 0.670.