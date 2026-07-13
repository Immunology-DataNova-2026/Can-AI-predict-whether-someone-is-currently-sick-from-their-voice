# 12 Confound Audit

Metadata-only models (age + gender + recording date), trained on the same
participant-disjoint split as the voice models. If these rival the voice
ensemble, the headline accuracy is partly explained by *who/when* was
recorded rather than *how their voice sounds*.

| Target | Baseline | Metadata-only acc | Metadata-only AUC | Voice ensemble acc | Voice ensemble AUC |
|---|---|---|---|---|---|
| multiclass | 0.709 | 0.810 | 0.906 | 0.752 | 0.816 |
| binary | 0.709 | 0.876 | 0.947 | 0.749 | 0.848 |

## Univariate association with is_sick (test set)
- Age alone (AUC): 0.544
- Recording date alone (AUC): 0.924
- Gender alone (AUC): 0.568

## How to read this
- Metadata-only AUC near 0.5 and accuracy near baseline => the confounds are
  weak and the voice models are likely learning real signal.
- Metadata-only accuracy/AUC approaching the voice ensemble => a chunk of the
  apparent performance may be recruitment/demographic confound; the voice
  model's *marginal* value over metadata is the honest headline.