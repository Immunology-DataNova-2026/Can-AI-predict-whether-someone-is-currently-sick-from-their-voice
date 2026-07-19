# 13 Error Analysis

## multiclass (overall ensemble accuracy: 0.752)

### Accuracy by true class
| class | n | accuracy |
|---|---|---|
| moderate_infection | 24 | 0.292 |
| mild_infection | 77 | 0.377 |
| not_infected | 246 | 0.915 |

### Accuracy by age band
| age_band | n | accuracy |
|---|---|---|
| 60+ | 21 | 0.571 |
| <30 | 166 | 0.759 |
| 45-59 | 60 | 0.767 |
| 30-44 | 100 | 0.770 |

### Accuracy by gender
| gender | n | accuracy |
|---|---|---|
| female | 111 | 0.685 |
| male | 236 | 0.784 |

### Accuracy by recording month (wave)
| record_year_month | n | accuracy |
|---|---|---|
| 2022-01 | 35 | 0.314 |
| 2021-03 | 3 | 0.333 |
| 2021-09 | 14 | 0.429 |
| 2020-09 | 7 | 0.429 |
| 2021-05 | 13 | 0.462 |
| 2022-02 | 18 | 0.500 |
| 2021-07 | 17 | 0.529 |
| 2021-06 | 14 | 0.643 |
| 2021-04 | 18 | 0.722 |
| 2020-08 | 17 | 0.765 |
| 2020-12 | 5 | 0.800 |
| 2020-06 | 6 | 0.833 |
| 2020-10 | 9 | 0.889 |
| 2020-05 | 43 | 0.907 |
| 2020-04 | 115 | 0.974 |
| 2020-11 | 1 | 1.000 |
| 2021-01 | 2 | 1.000 |
| 2021-02 | 1 | 1.000 |
| 2020-07 | 7 | 1.000 |
| 2021-10 | 1 | 1.000 |
| 2021-12 | 1 | 1.000 |

### Confidence
- Mean confidence on correct: 0.829
- Mean confidence on errors: 0.671
- High-confidence errors (conf>0.8): 23 of 86 errors

- Misclassified participants written to `misclassified_multiclass.csv` (86 rows)

## binary (overall ensemble accuracy: 0.749)

### Accuracy by true class
| class | n | accuracy |
|---|---|---|
| not_sick | 246 | 0.715 |
| sick | 101 | 0.832 |

### Accuracy by age band
| age_band | n | accuracy |
|---|---|---|
| <30 | 166 | 0.729 |
| 30-44 | 100 | 0.760 |
| 60+ | 21 | 0.762 |
| 45-59 | 60 | 0.783 |

### Accuracy by gender
| gender | n | accuracy |
|---|---|---|
| female | 111 | 0.676 |
| male | 236 | 0.784 |

### Accuracy by recording month (wave)
| record_year_month | n | accuracy |
|---|---|---|
| 2021-03 | 3 | 0.333 |
| 2021-05 | 13 | 0.462 |
| 2021-04 | 18 | 0.500 |
| 2020-07 | 7 | 0.571 |
| 2020-12 | 5 | 0.600 |
| 2022-02 | 18 | 0.611 |
| 2021-09 | 14 | 0.714 |
| 2020-09 | 7 | 0.714 |
| 2022-01 | 35 | 0.714 |
| 2020-05 | 43 | 0.721 |
| 2020-10 | 9 | 0.778 |
| 2021-06 | 14 | 0.786 |
| 2020-08 | 17 | 0.824 |
| 2020-04 | 115 | 0.826 |
| 2020-06 | 6 | 0.833 |
| 2020-11 | 1 | 1.000 |
| 2021-07 | 17 | 1.000 |
| 2021-02 | 1 | 1.000 |
| 2021-01 | 2 | 1.000 |
| 2021-10 | 1 | 1.000 |
| 2021-12 | 1 | 1.000 |

### Confidence
- Mean confidence on correct: 0.855
- Mean confidence on errors: 0.690
- High-confidence errors (conf>0.8): 15 of 87 errors

- Misclassified participants written to `misclassified_binary.csv` (87 rows)
