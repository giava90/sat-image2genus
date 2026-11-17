### A. Overall test scores

| test set | model | n | accuracy (95% CI) | weighted precision | macro F1 | conifer vs broadleaf accuracy (95% CI) |
|---|---|---|---|---|---|---|
| common | M1 LightGBM, S2 statistics | 2322 | 78.8% (77.0% to 80.4%) | 79.8% | 0.775 | 87.8% (86.5% to 89.1%) |
| common | M2 small CNN, S2 patches | 2322 | 89.9% (88.7% to 91.1%) | 90.1% | 0.891 | 95.3% (94.4% to 96.1%) |
| common | M3 ResNet-18, aerial | 2322 | 91.9% (90.8% to 93.0%) | 92.1% | 0.911 | 96.6% (95.8% to 97.3%) |
| full | M1 LightGBM, S2 statistics | 2738 | 78.9% (77.5% to 80.4%) | 78.9% | 0.784 | 87.9% (86.7% to 89.1%) |
| full | M2 small CNN, S2 patches | 2738 | 89.5% (88.3% to 90.6%) | 89.5% | 0.894 | 95.4% (94.6% to 96.2%) |

### B. Paired accuracy differences (same test patches, 1,000 paired bootstrap resamples)

| test set | comparison | difference in accuracy | 95% CI | share of resamples where the first model is better |
|---|---|---|---|---|
| common | M2 small CNN, S2 patches minus M1 LightGBM, S2 statistics | +11.1% | +9.3% to +12.8% | 1.000 |
| common | M3 ResNet-18, aerial minus M1 LightGBM, S2 statistics | +13.1% | +11.4% to +15.0% | 1.000 |
| common | M3 ResNet-18, aerial minus M2 small CNN, S2 patches | +2.0% | +0.7% to +3.4% | 0.997 |
| full | M2 small CNN, S2 patches minus M1 LightGBM, S2 statistics | +10.6% | +9.0% to +12.0% | 1.000 |

### C. Per-genus precision and recall, common test set

| model | Picea P | Picea R | Pinus P | Pinus R | Fagus P | Fagus R | Quercus P | Quercus R | conifer P | conifer R | broadleaf P | broadleaf R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 LightGBM, S2 statistics | 0.777 | 0.758 | 0.893 | 0.868 | 0.807 | 0.709 | 0.612 | 0.807 | 0.904 | 0.880 | 0.845 | 0.875 |
| M2 small CNN, S2 patches | 0.898 | 0.905 | 0.947 | 0.926 | 0.906 | 0.870 | 0.800 | 0.884 | 0.963 | 0.954 | 0.939 | 0.951 |
| M3 ResNet-18, aerial | 0.932 | 0.940 | 0.970 | 0.964 | 0.912 | 0.854 | 0.816 | 0.907 | 0.970 | 0.971 | 0.961 | 0.960 |

### D. Training runs (device, time, selected epoch)

| model | device | training patches | training time | epochs run | selected epoch / rounds | best validation accuracy |
|---|---|---|---|---|---|---|
| M1 LightGBM, S2 statistics | cpu | 22425 | 0.2 min | - | 270 | - |
| M2 small CNN, S2 patches | cpu | 22425 | 11.0 min | 40 | 37 | 89.8% |
| M3 ResNet-18, aerial | cpu | 11790 | 60.6 min | 12 | 12 | 91.0% |

### E. Volume error in % of true volume, common test set, all three models (placeholder volumes)

Point estimate on the test set, with the 95% bootstrap interval in brackets. 'count' = area by counting predictions; 'adjusted' = after confusion-matrix adjustment. The validation matrix used for the adjustment is held fixed, so adjusted intervals leave out validation-set noise and are too narrow. The last column averages |error| over the four genera. Bootstrapping an absolute value biases it upward, so its interval can exclude a point estimate near zero; read it as the spread of the error size, not as a confidence interval.

| scenario | model | estimator | Picea | Pinus | Fagus | Quercus | total volume | mean abs. genus error |
|---|---|---|---|---|---|---|---|---|
| S1 dataset shares | M1 LightGBM, S2 statistics | count | -1.6% [-6.9%, +3.2%] | +0.9% [-3.1%, +4.6%] | -4.2% [-9.9%, +1.7%] | +3.8% [-1.1%, +8.6%] | -0.2% [-0.6%, +0.0%] | 2.6% [1.1%, 5.7%] |
| S1 dataset shares | M1 LightGBM, S2 statistics | adjusted | -4.8% [-12.5%, +2.3%] | +1.6% [-3.4%, +6.4%] | -1.4% [-10.5%, +8.1%] | +3.2% [-4.4%, +10.8%] | -0.4% [-0.8%, +0.0%] | 2.7% [1.2%, 7.7%] |
| S1 dataset shares | M2 small CNN, S2 patches | count | +2.0% [-1.9%, +6.2%] | -0.8% [-3.6%, +2.1%] | +1.2% [-3.1%, +5.5%] | -1.7% [-5.5%, +2.1%] | +0.2% [-0.1%, +0.4%] | 1.4% [0.6%, 3.6%] |
| S1 dataset shares | M2 small CNN, S2 patches | adjusted | +1.6% [-2.8%, +6.2%] | -2.3% [-5.4%, +1.0%] | -1.4% [-6.9%, +4.0%] | +2.1% [-2.6%, +7.0%] | +0.1% [-0.2%, +0.3%] | 1.8% [0.8%, 5.0%] |
| S1 dataset shares | M3 ResNet-18, aerial | count | +1.7% [-1.5%, +4.8%] | +0.4% [-1.7%, +2.5%] | -1.7% [-6.1%, +2.4%] | -0.2% [-3.8%, +3.5%] | +0.0% [-0.2%, +0.3%] | 1.0% [0.4%, 3.3%] |
| S1 dataset shares | M3 ResNet-18, aerial | adjusted | +0.7% [-2.8%, +4.2%] | -1.2% [-3.5%, +1.1%] | -5.6% [-11.5%, -0.3%] | +5.3% [+0.7%, +9.7%] | -0.2% [-0.4%, +0.1%] | 3.2% [1.2%, 6.0%] |
| S2 equal shares | M1 LightGBM, S2 statistics | count | -5.1% [-9.9%, -0.8%] | +2.6% [-1.4%, +6.5%] | -6.1% [-11.3%, -0.6%] | +8.5% [+3.3%, +13.9%] | -0.6% [-0.9%, -0.3%] | 5.6% [3.1%, 8.5%] |
| S2 equal shares | M1 LightGBM, S2 statistics | adjusted | -3.1% [-10.1%, +3.3%] | +1.7% [-3.3%, +6.6%] | -1.7% [-10.2%, +7.0%] | +3.1% [-5.1%, +11.4%] | -0.3% [-0.7%, +0.1%] | 2.4% [1.0%, 7.2%] |
| S2 equal shares | M2 small CNN, S2 patches | count | -0.1% [-3.5%, +3.5%] | +0.1% [-2.9%, +3.2%] | -0.1% [-4.1%, +3.7%] | +0.1% [-3.9%, +4.1%] | -0.0% [-0.2%, +0.2%] | 0.1% [0.4%, 3.0%] |
| S2 equal shares | M2 small CNN, S2 patches | adjusted | +1.3% [-2.6%, +5.3%] | -2.4% [-5.8%, +1.0%] | -0.6% [-5.6%, +4.3%] | +1.7% [-3.3%, +6.8%] | +0.1% [-0.2%, +0.4%] | 1.5% [0.7%, 4.6%] |
| S2 equal shares | M3 ResNet-18, aerial | count | +0.3% [-2.4%, +3.1%] | +0.8% [-1.3%, +3.0%] | -3.0% [-7.0%, +1.0%] | +1.8% [-2.1%, +5.6%] | -0.1% [-0.3%, +0.1%] | 1.5% [0.4%, 3.7%] |
| S2 equal shares | M3 ResNet-18, aerial | adjusted | +0.7% [-2.4%, +3.8%] | -1.8% [-4.1%, +0.6%] | -4.4% [-9.6%, +0.7%] | +5.5% [+0.6%, +10.2%] | -0.1% [-0.4%, +0.1%] | 3.1% [1.2%, 5.9%] |

**Spruce sentences (common test set, all three models, scenario S1, counting predictions (placeholder volumes)):**

- A classifier with 78.8% accuracy (M1 LightGBM, S2 statistics) misstates spruce volume by -1.6% (95% bootstrap interval -6.9% to +3.2%) (placeholder volumes).
- A classifier with 89.9% accuracy (M2 small CNN, S2 patches) misstates spruce volume by +2.0% (95% bootstrap interval -1.9% to +6.2%) (placeholder volumes).
- A classifier with 91.9% accuracy (M3 ResNet-18, aerial) misstates spruce volume by +1.7% (95% bootstrap interval -1.5% to +4.8%) (placeholder volumes).

### E_full. Volume error in % of true volume, full test set, models 1 and 2 (placeholder volumes)

Point estimate on the test set, with the 95% bootstrap interval in brackets. 'count' = area by counting predictions; 'adjusted' = after confusion-matrix adjustment. The validation matrix used for the adjustment is held fixed, so adjusted intervals leave out validation-set noise and are too narrow. The last column averages |error| over the four genera. Bootstrapping an absolute value biases it upward, so its interval can exclude a point estimate near zero; read it as the spread of the error size, not as a confidence interval.

| scenario | model | estimator | Picea | Pinus | Fagus | Quercus | total volume | mean abs. genus error |
|---|---|---|---|---|---|---|---|---|
| S1 dataset shares | M1 LightGBM, S2 statistics | count | +1.8% [-3.2%, +7.3%] | +0.7% [-2.8%, +4.2%] | -6.2% [-11.2%, -1.3%] | +3.1% [-0.9%, +7.4%] | -0.1% [-0.4%, +0.2%] | 3.0% [1.2%, 5.8%] |
| S1 dataset shares | M1 LightGBM, S2 statistics | adjusted | +0.5% [-6.8%, +8.6%] | +2.6% [-1.8%, +7.1%] | -0.7% [-8.4%, +7.4%] | -2.3% [-8.0%, +4.0%] | +0.0% [-0.4%, +0.4%] | 1.5% [0.9%, 5.9%] |
| S1 dataset shares | M2 small CNN, S2 patches | count | +2.0% [-1.6%, +5.6%] | -0.9% [-3.6%, +1.7%] | +2.2% [-1.7%, +6.1%] | -2.4% [-5.6%, +0.9%] | +0.2% [-0.0%, +0.4%] | 1.9% [0.7%, 3.9%] |
| S1 dataset shares | M2 small CNN, S2 patches | adjusted | +1.0% [-3.0%, +5.1%] | -1.3% [-4.4%, +1.6%] | +2.5% [-2.3%, +7.4%] | -1.6% [-5.6%, +2.6%] | +0.2% [-0.1%, +0.4%] | 1.6% [0.7%, 4.1%] |
| S2 equal shares | M1 LightGBM, S2 statistics | count | -2.6% [-7.1%, +2.5%] | +2.4% [-1.4%, +6.4%] | -7.7% [-12.6%, -2.9%] | +7.9% [+3.4%, +12.8%] | -0.5% [-0.8%, -0.2%] | 5.1% [2.9%, 7.6%] |
| S2 equal shares | M1 LightGBM, S2 statistics | adjusted | +0.8% [-5.8%, +8.2%] | +2.7% [-2.1%, +7.6%] | -1.1% [-8.7%, +6.6%] | -2.4% [-8.9%, +4.9%] | +0.0% [-0.4%, +0.4%] | 1.7% [1.0%, 6.0%] |
| S2 equal shares | M2 small CNN, S2 patches | count | -0.1% [-3.4%, +3.2%] | +0.1% [-2.8%, +2.9%] | +0.7% [-3.1%, +4.4%] | -0.6% [-4.1%, +3.0%] | +0.0% [-0.2%, +0.3%] | 0.4% [0.4%, 2.8%] |
| S2 equal shares | M2 small CNN, S2 patches | adjusted | +0.8% [-2.8%, +4.6%] | -1.5% [-4.7%, +1.7%] | +2.6% [-2.0%, +7.1%] | -2.0% [-6.2%, +2.6%] | +0.2% [-0.1%, +0.4%] | 1.7% [0.7%, 4.2%] |

**Spruce sentences (full test set, models 1 and 2, scenario S1, counting predictions (placeholder volumes)):**

- A classifier with 78.9% accuracy (M1 LightGBM, S2 statistics) misstates spruce volume by +1.8% (95% bootstrap interval -3.2% to +7.3%) (placeholder volumes).
- A classifier with 89.5% accuracy (M2 small CNN, S2 patches) misstates spruce volume by +2.0% (95% bootstrap interval -1.6% to +5.6%) (placeholder volumes).

### G. Accuracy by distance from a test patch to the nearest training patch (Chebyshev distance between patch centres)

200 m Sentinel-2 windows overlap when the distance is under 200 m.


common test set:

| distance band | n | M1 LightGBM, S2 statistics | M2 small CNN, S2 patches | M3 ResNet-18, aerial |
|---|---|---|---|---|
| 0-200 m | 1587 | 80.4% | 90.3% | 92.4% |
| 200-600 m | 706 | 76.6% | 89.7% | 91.2% |
| 600-2000 m | 26 | 46.2% | 73.1% | 84.6% |
| 2000-inf m | 3 | 33.3% | 100.0% | 66.7% |
| all | 2322 | 78.8% | 89.9% | 91.9% |

full test set:

| distance band | n | M1 LightGBM, S2 statistics | M2 small CNN, S2 patches |
|---|---|---|---|
| 0-200 m | 1951 | 80.3% | 89.8% |
| 200-600 m | 752 | 76.5% | 89.1% |
| 600-2000 m | 31 | 54.8% | 77.4% |
| 2000-inf m | 4 | 50.0% | 100.0% |
| all | 2738 | 78.9% | 89.5% |
