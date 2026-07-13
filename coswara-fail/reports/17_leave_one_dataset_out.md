# 17 Leave-One-Dataset-Out Evaluation

Each dataset is held out once as the test set while the model trains on the
pooled remaining datasets. Harmonized symptomatic-vs-healthy label, cough
clips only, participant-aggregated metrics. This is the strongest
generalization test in the project.

| held-out test | train on | AUC | balanced acc | accuracy | recall(sick) | baseline acc | n |
|---|---|---|---|---|---|---|---|
| **coswara** | coughvid+sounddr | 0.623 | 0.587 | 0.585 | 0.597 | 0.596 | 2154 |
| **coughvid** | coswara+sounddr | 0.539 | 0.522 | 0.522 | 0.322 | 0.500 | 9992 |
| **sounddr** | coswara+coughvid | 0.473 | 0.472 | 0.481 | 0.447 | 0.670 | 1310 |

Mean held-out ensemble AUC across datasets: **0.545**.