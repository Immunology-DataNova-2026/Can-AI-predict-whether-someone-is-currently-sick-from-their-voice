# Voice controls — Parkinson's (Sakar)

- speakers: 252  (64 healthy, 188 PD)
- per-speaker AUC: 0.850  95% CI [0.796, 0.901]
- per-speaker accuracy: 0.821  95% CI [0.774, 0.869]

## Top 15 features by mutual information

 1. tqwt_entropy_log_dec_35      MI=0.1078  RF=0.0047
 2. std_delta_delta_log_energy   MI=0.1044  RF=0.0159
 3. std_8th_delta_delta          MI=0.0969  RF=0.0079
 4. mean_MFCC_2nd_coef           MI=0.0961  RF=0.0069
 5. tqwt_TKEO_mean_dec_16        MI=0.0958  RF=0.0021
 6. tqwt_entropy_shannon_dec_35  MI=0.0946  RF=0.0015
 7. tqwt_maxValue_dec_12         MI=0.0936  RF=0.0055
 8. tqwt_TKEO_std_dec_12         MI=0.0917  RF=0.0085
 9. tqwt_TKEO_mean_dec_12        MI=0.0916  RF=0.0096
10. tqwt_stdValue_dec_12         MI=0.0912  RF=0.0062
11. tqwt_entropy_log_dec_11      MI=0.0912  RF=0.0039
12. tqwt_stdValue_dec_15         MI=0.0910  RF=0.0007
13. tqwt_entropy_log_dec_12      MI=0.0906  RF=0.0100
14. tqwt_energy_dec_27           MI=0.0891  RF=0.0051
15. tqwt_TKEO_std_dec_13         MI=0.0886  RF=0.0078

Saved: reports/feature_importance_sakar.csv

## Top-25 features grouped by acoustic family

| family | count in top 25 | share |
|---|---|---|
| jitter | 0 | 0% |
| shimmer | 0 | 0% |
| HNR | 0 | 0% |
| MFCC | 1 | 4% |
| TQWT | 17 | 68% |
| other | 7 | 28% |

# Voice controls — voice disorder (SVD)

- speakers: 1119  (669 healthy, 450 pathological)
- per-speaker AUC: 0.866  95% CI [0.844, 0.888]
- per-speaker accuracy: 0.804  95% CI [0.780, 0.828]