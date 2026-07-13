# 14 Repeated Cross-Validation (tabular GBM, participant-level)

- 5-fold x 3 repeats = 15 folds
- Participant-disjoint; per-participant aggregation; GBM base model.

## multiclass
- Accuracy: 0.713 +/- 0.015  (95% CI 0.688-0.742)
- Macro F1: 0.531 +/- 0.022  (95% CI 0.502-0.579)
- AUC:      0.778 +/- 0.016  (95% CI 0.756-0.806)

## binary
- Accuracy: 0.768 +/- 0.015  (95% CI 0.739-0.794)
- Macro F1: 0.622 +/- 0.034  (95% CI 0.550-0.679)
- AUC:      0.808 +/- 0.017  (95% CI 0.778-0.832)
