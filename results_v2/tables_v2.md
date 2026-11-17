### T1. The CNN's lead over LightGBM, three ways (accuracy points; 95% intervals: patch bootstrap / 2 km block bootstrap)

Best fair baseline = `lgbm_centre_ring`, chosen among stage 2 variants a-d by validation accuracy.

| comparison | test set | n test | CNN minus LightGBM | patch-bootstrap 95% CI | block-bootstrap 95% CI |
|---|---|---|---|---|---|
| original split, run 1 LightGBM (200 m summary stats) | full | 2738 | +10.6 | +9.1 to +12.1 | +9.1 to +12.1 |
| original split, best fair LightGBM baseline | full | 2738 | +3.1 | +1.9 to +4.2 | +1.9 to +4.1 |
| blocked seed0 split (all retrained), run 1 LightGBM features | full | 5536 | +12.5 | +11.4 to +13.7 | +11.1 to +14.1 |
| blocked seed0 split (all retrained), best fair LightGBM baseline | full | 5536 | +3.7 | +2.8 to +4.6 | +2.7 to +4.8 |
| original split, run 1 LightGBM (200 m summary stats) | common | 2322 | +11.1 | +9.3 to +12.8 | +9.4 to +12.9 |
| original split, best fair LightGBM baseline | common | 2322 | +3.1 | +1.8 to +4.2 | +1.9 to +4.2 |
| blocked seed0 split (all retrained), run 1 LightGBM features | common | 4632 | +12.6 | +11.5 to +13.8 | +10.9 to +14.4 |
| blocked seed0 split (all retrained), best fair LightGBM baseline | common | 4632 | +3.6 | +2.7 to +4.5 | +2.5 to +4.7 |

### T2. Stage 2: is the LightGBM baseline fair? Test accuracy on the original split

Differences against run 1's M1 and M2 are paired (same test patches, same 1,000 patch resamples).


common test set:

| model | description | validation acc. | test accuracy (patch 95% CI) | minus M1 (95% CI) | minus M2 (95% CI) |
|---|---|---|---|---|---|
| M1 | run 1 model | - | 78.8% (77.0% to 80.4%) | - | - |
| M2 | run 1 model | - | 89.9% (88.7% to 91.1%) | +11.1 (+9.3 to +12.8) | - |
| M3 | run 1 model | - | 91.9% (90.8% to 93.0%) | - | - |
| cnn_centre6 | e. CNN, centre 6x6 only | 87.4% | 88.2% (86.8% to 89.4%) | +9.3 (+7.5 to +11.1) | -1.8 (-2.9 to -0.6) |
| cnn_masked_centre | f. CNN, 200 m with centre 6x6 masked | 86.0% | 86.9% (85.5% to 88.2%) | +8.1 (+6.5 to +9.6) | -3.1 (-4.1 to -2.0) |
| lgbm_centre_ring | b. LightGBM, centre 6x6 + ring summary stats | 87.2% | 86.9% (85.4% to 88.3%) | +8.1 (+6.5 to +9.6) | -3.1 (-4.2 to -1.8) |
| lgbm_raw200 | d. LightGBM, raw pixels of 200 m patch | 84.9% | 84.4% (82.9% to 85.9%) | +5.6 (+3.9 to +7.1) | -5.6 (-6.8 to -4.2) |
| lgbm_raw60 | c. LightGBM, raw pixels of 60 m patch | 86.0% | 84.1% (82.6% to 85.6%) | +5.3 (+3.4 to +6.9) | -5.9 (-7.2 to -4.4) |
| lgbm_summary200_repro | LightGBM run 1 features, retrained (reproduction check) | 79.6% | 78.8% (77.0% to 80.4%) | +0.0 (+0.0 to +0.0) | -11.1 (-12.8 to -9.3) |
| lgbm_summary60 | a. LightGBM, summary stats of 60 m patch | 86.3% | 85.5% (84.0% to 87.0%) | +6.7 (+5.0 to +8.4) | -4.4 (-5.6 to -3.1) |

full test set:

| model | description | validation acc. | test accuracy (patch 95% CI) | minus M1 (95% CI) | minus M2 (95% CI) |
|---|---|---|---|---|---|
| M1 | run 1 model | - | 78.9% (77.5% to 80.3%) | - | - |
| M2 | run 1 model | - | 89.5% (88.3% to 90.6%) | +10.6 (+9.1 to +12.1) | - |
| cnn_centre6 | e. CNN, centre 6x6 only | 87.4% | 87.8% (86.5% to 89.1%) | +8.9 (+7.4 to +10.5) | -1.7 (-2.7 to -0.6) |
| cnn_masked_centre | f. CNN, 200 m with centre 6x6 masked | 86.0% | 86.4% (85.1% to 87.9%) | +7.5 (+6.1 to +8.9) | -3.0 (-4.1 to -2.0) |
| lgbm_centre_ring | b. LightGBM, centre 6x6 + ring summary stats | 87.2% | 86.4% (85.2% to 87.7%) | +7.5 (+6.1 to +8.9) | -3.1 (-4.2 to -1.9) |
| lgbm_raw200 | d. LightGBM, raw pixels of 200 m patch | 84.9% | 83.6% (82.2% to 85.1%) | +4.7 (+3.2 to +6.3) | -5.8 (-7.1 to -4.6) |
| lgbm_raw60 | c. LightGBM, raw pixels of 60 m patch | 86.0% | 83.3% (81.8% to 84.7%) | +4.3 (+2.6 to +6.1) | -6.2 (-7.5 to -5.0) |
| lgbm_summary200_repro | LightGBM run 1 features, retrained (reproduction check) | 79.6% | 78.9% (77.5% to 80.3%) | +0.0 (+0.0 to +0.0) | -10.6 (-12.1 to -9.1) |
| lgbm_summary60 | a. LightGBM, summary stats of 60 m patch | 86.3% | 84.9% (83.5% to 86.2%) | +6.0 (+4.5 to +7.5) | -4.6 (-5.8 to -3.5) |

### T3. Stage 3: accuracy on the original split and on blocked splits

Blocked splits regroup all four-genus patches into 2 km blocks (20% of patches test, 10% validation) and drop training patches within 200 m of a held-out patch. Model 3 was not retrained on blocked splits.


full test set:

| model | split | n test | blocks | accuracy | patch-bootstrap 95% CI | block-bootstrap 95% CI |
|---|---|---|---|---|---|---|
| LightGBM, run 1 features (200 m summary stats) | original | 2738 | 1220 | 78.9% | 77.5% to 80.3% | 77.2% to 80.6% |
| small CNN, 200 m patch | original | 2738 | 1220 | 89.5% | 88.3% to 90.6% | 88.3% to 90.7% |
| best stage 2 LightGBM variant (lgbm_centre_ring) | original | 2738 | 1220 | 86.4% | 85.2% to 87.7% | 85.1% to 87.7% |
| 1-NN on patch centre | original | 2738 | 1220 | 71.4% | 69.6% to 73.0% | 69.5% to 73.2% |
| 5-NN on patch centre | original | 2738 | 1220 | 68.3% | 66.5% to 69.9% | 66.5% to 70.3% |
| LightGBM on (x, y) | original | 2738 | 1220 | 67.3% | 65.6% to 69.1% | 65.5% to 69.4% |
| LightGBM, run 1 features (200 m summary stats) | blocked seed0 | 5536 | 463 | 76.0% | 74.9% to 77.1% | 74.3% to 77.8% |
| best stage 2 LightGBM variant (lgbm_centre_ring) | blocked seed0 | 5536 | 463 | 84.9% | 84.0% to 85.9% | 83.6% to 86.3% |
| small CNN, 200 m patch | blocked seed0 | 5536 | 463 | 88.6% | 87.7% to 89.4% | 87.4% to 89.7% |
| 1-NN on patch centre | blocked seed0 | 5536 | 463 | 53.9% | 52.6% to 55.2% | 51.1% to 57.2% |
| 5-NN on patch centre | blocked seed0 | 5536 | 463 | 54.7% | 53.5% to 56.0% | 51.5% to 58.0% |
| LightGBM on (x, y) | blocked seed0 | 5536 | 463 | 57.4% | 56.2% to 58.7% | 54.3% to 60.4% |
| 1-NN on patch centre | blocked seed1 | 5547 | 422 | 54.5% | 53.1% to 55.7% | 50.9% to 58.2% |
| 5-NN on patch centre | blocked seed1 | 5547 | 422 | 56.1% | 54.8% to 57.3% | 52.6% to 60.0% |

common test set:

| model | split | n test | blocks | accuracy | patch-bootstrap 95% CI | block-bootstrap 95% CI |
|---|---|---|---|---|---|---|
| LightGBM, run 1 features (200 m summary stats) | original | 2322 | 1103 | 78.8% | 77.0% to 80.4% | 77.0% to 80.6% |
| small CNN, 200 m patch | original | 2322 | 1103 | 89.9% | 88.7% to 91.1% | 88.7% to 91.2% |
| best stage 2 LightGBM variant (lgbm_centre_ring) | original | 2322 | 1103 | 86.9% | 85.4% to 88.3% | 85.5% to 88.2% |
| 1-NN on patch centre | original | 2322 | 1103 | 70.3% | 68.5% to 72.1% | 68.4% to 72.4% |
| 5-NN on patch centre | original | 2322 | 1103 | 69.1% | 67.2% to 71.0% | 66.9% to 71.4% |
| LightGBM on (x, y) | original | 2322 | 1103 | 68.5% | 66.5% to 70.4% | 66.3% to 70.9% |
| ResNet-18, aerial | original | 2322 | 1103 | 91.9% | 90.8% to 93.0% | 90.8% to 93.1% |
| LightGBM, run 1 features (200 m summary stats) | blocked seed0 | 4632 | 451 | 76.2% | 75.0% to 77.4% | 74.2% to 78.0% |
| best stage 2 LightGBM variant (lgbm_centre_ring) | blocked seed0 | 4632 | 451 | 85.2% | 84.2% to 86.3% | 83.8% to 86.5% |
| small CNN, 200 m patch | blocked seed0 | 4632 | 451 | 88.8% | 87.9% to 89.7% | 87.5% to 89.9% |
| 1-NN on patch centre | blocked seed0 | 4632 | 451 | 56.2% | 54.8% to 57.6% | 52.8% to 59.8% |
| 5-NN on patch centre | blocked seed0 | 4632 | 451 | 56.4% | 55.0% to 57.8% | 52.7% to 60.0% |
| LightGBM on (x, y) | blocked seed0 | 4632 | 451 | 59.9% | 58.5% to 61.2% | 56.4% to 63.6% |
| 1-NN on patch centre | blocked seed1 | 4652 | 413 | 57.7% | 56.3% to 59.1% | 53.8% to 62.0% |
| 5-NN on patch centre | blocked seed1 | 4652 | 413 | 58.4% | 57.1% to 59.8% | 54.0% to 62.9% |

### T4. Stage 3: paired differences, patch vs block bootstrap

| split | test set | comparison | difference (points) | patch-bootstrap 95% CI | block-bootstrap 95% CI |
|---|---|---|---|---|---|
| original | common | M2 - M1 | +11.1 | +9.3 to +12.8 | +9.4 to +12.9 |
| original | common | M3 - M1 | +13.1 | +11.4 to +15.0 | +11.2 to +15.1 |
| original | common | M3 - M2 | +2.0 | +0.7 to +3.4 | +0.6 to +3.4 |
| original | common | M2 - best_lgbm | +3.1 | +1.8 to +4.2 | +1.9 to +4.2 |
| original | common | M3 - best_lgbm | +5.1 | +3.5 to +6.5 | +3.6 to +6.6 |
| original | common | M1 - knn1 | +8.5 | +6.3 to +10.6 | +6.3 to +10.6 |
| original | common | M2 - knn1 | +19.6 | +17.5 to +21.7 | +17.4 to +21.8 |
| original | full | M2 - M1 | +10.6 | +9.1 to +12.1 | +9.1 to +12.1 |
| original | full | M2 - best_lgbm | +3.1 | +1.9 to +4.2 | +1.9 to +4.1 |
| original | full | M1 - knn1 | +7.6 | +5.7 to +9.4 | +5.8 to +9.6 |
| original | full | M2 - knn1 | +18.1 | +16.4 to +20.1 | +16.1 to +20.1 |
| blocked seed0 | common | M2 - M1 | +12.6 | +11.5 to +13.8 | +10.9 to +14.4 |
| blocked seed0 | common | M2 - best_lgbm | +3.6 | +2.7 to +4.5 | +2.5 to +4.7 |
| blocked seed0 | common | best_lgbm - M1 | +9.0 | +7.9 to +10.1 | +7.7 to +10.4 |
| blocked seed0 | common | M1 - knn1 | +19.9 | +18.3 to +21.6 | +16.5 to +23.3 |
| blocked seed0 | common | M2 - knn1 | +32.6 | +31.0 to +34.3 | +29.0 to +35.9 |
| blocked seed0 | full | M2 - M1 | +12.5 | +11.4 to +13.7 | +11.1 to +14.1 |
| blocked seed0 | full | M2 - best_lgbm | +3.7 | +2.8 to +4.6 | +2.7 to +4.8 |
| blocked seed0 | full | best_lgbm - M1 | +8.9 | +7.8 to +9.9 | +7.6 to +10.2 |
| blocked seed0 | full | M1 - knn1 | +22.1 | +20.6 to +23.7 | +19.0 to +25.1 |
| blocked seed0 | full | M2 - knn1 | +34.7 | +33.3 to +36.2 | +31.6 to +37.6 |

### T5-common. Stage 1: per-genus volume error by estimator, common test set (placeholder volumes, placeholder regions)

1000 replicates of 5,000 stands; test patches and validation set resampled in every replicate. Cell = bias / RMSE, in % of the true genus volume. Scenario shares (spruce/pine/beech/oak): S1 dataset shares 25/32/28/15; S2 equal shares 25/25/25/25; A 60/25/10/5; B 15/70/10/5; C 40/35/23/2.

| scenario | model | estimator | Picea | Pinus | Fagus | Quercus | total volume | mean RMSE over genera |
|---|---|---|---|---|---|---|---|---|
| S1 dataset shares | M1 LightGBM | CC | -2.7 / 3.2% | -2.7 / 2.9% | -12.3 / 12.4% | +33.3 / 33.4% | -0.7 / 0.7% | 13.0% |
| S1 dataset shares | M1 LightGBM | AC | -0.3 / 4.6% | +1.5 / 3.1% | -2.0 / 5.1% | +1.1 / 8.3% | -0.1 / 0.3% | 5.3% |
| S1 dataset shares | M1 LightGBM | PCC | -5.0 / 5.1% | -4.4 / 4.5% | -10.1 / 10.1% | +36.7 / 36.8% | -0.8 / 0.8% | 14.1% |
| S1 dataset shares | M1 LightGBM | PAC | +3.4 / 5.2% | +0.2 / 2.3% | -0.4 / 4.0% | -5.3 / 8.8% | +0.2 / 0.3% | 5.1% |
| S1 dataset shares | M1 LightGBM | EM | -0.5 / 2.0% | +0.6 / 1.3% | +0.2 / 2.0% | -1.0 / 3.6% | -0.0 / 0.1% | 2.2% |
| S1 dataset shares | M2 CNN | CC | +0.7 / 1.4% | -2.2 / 2.4% | -4.0 / 4.2% | +11.2 / 11.3% | -0.1 / 0.1% | 4.8% |
| S1 dataset shares | M2 CNN | AC | +1.4 / 2.8% | -1.7 / 2.5% | +0.3 / 2.8% | +0.7 / 5.0% | +0.1 / 0.2% | 3.3% |
| S1 dataset shares | M2 CNN | PCC | -0.4 / 1.1% | -3.4 / 3.4% | -3.5 / 3.6% | +14.6 / 14.6% | -0.2 / 0.2% | 5.7% |
| S1 dataset shares | M2 CNN | PAC | +2.5 / 3.3% | -3.0 / 3.4% | +3.6 / 4.3% | -4.3 / 5.9% | +0.3 / 0.4% | 4.2% |
| S1 dataset shares | M2 CNN | EM | +1.1 / 1.6% | -1.5 / 1.7% | +3.4 / 3.6% | -4.8 / 5.2% | +0.2 / 0.2% | 3.1% |
| S1 dataset shares | M3 ResNet-18 | CC | +0.8 / 1.3% | -0.5 / 0.8% | -6.4 / 6.5% | +11.6 / 11.8% | -0.2 / 0.2% | 5.1% |
| S1 dataset shares | M3 ResNet-18 | AC | +0.7 / 2.1% | -1.1 / 1.8% | -2.5 / 3.8% | +5.9 / 7.6% | -0.0 / 0.2% | 3.8% |
| S1 dataset shares | M3 ResNet-18 | PCC | +0.8 / 1.2% | -1.4 / 1.5% | -6.1 / 6.2% | +13.1 / 13.2% | -0.2 / 0.2% | 5.5% |
| S1 dataset shares | M3 ResNet-18 | PAC | +0.6 / 1.9% | -0.5 / 1.4% | -2.4 / 3.5% | +4.6 / 6.2% | -0.1 / 0.1% | 3.2% |
| S1 dataset shares | M3 ResNet-18 | EM | +0.2 / 1.0% | +0.3 / 0.6% | -3.1 / 3.3% | +4.9 / 5.2% | -0.1 / 0.1% | 2.5% |
| S2 equal shares | M1 LightGBM | CC | -5.0 / 5.3% | +2.6 / 3.0% | -6.1 / 6.4% | +8.6 / 8.7% | -0.6 / 0.6% | 5.9% |
| S2 equal shares | M1 LightGBM | AC | -2.9 / 5.5% | +1.6 / 3.8% | -1.7 / 5.6% | +3.1 / 6.5% | -0.3 / 0.4% | 5.4% |
| S2 equal shares | M1 LightGBM | PCC | -6.5 / 6.6% | +2.9 / 3.1% | -1.4 / 1.9% | +5.0 / 5.1% | -0.5 / 0.5% | 4.2% |
| S2 equal shares | M1 LightGBM | PAC | +1.1 / 4.2% | +0.4 / 3.0% | -0.1 / 4.6% | -1.4 / 5.1% | +0.1 / 0.3% | 4.3% |
| S2 equal shares | M1 LightGBM | EM | -2.3 / 3.1% | +1.9 / 2.4% | +2.6 / 3.5% | -2.2 / 3.1% | -0.1 / 0.1% | 3.0% |
| S2 equal shares | M2 CNN | CC | -0.1 / 1.2% | +0.1 / 1.1% | -0.1 / 1.4% | +0.1 / 1.3% | -0.0 / 0.1% | 1.2% |
| S2 equal shares | M2 CNN | AC | +1.3 / 2.7% | -2.4 / 3.4% | -0.6 / 3.6% | +1.7 / 4.0% | +0.1 / 0.2% | 3.4% |
| S2 equal shares | M2 CNN | PCC | -1.7 / 2.0% | +0.1 / 0.9% | +1.8 / 2.0% | -0.1 / 1.0% | -0.1 / 0.1% | 1.5% |
| S2 equal shares | M2 CNN | PAC | +2.4 / 3.2% | -3.8 / 4.3% | +2.8 / 4.1% | -1.4 / 3.4% | +0.3 / 0.3% | 3.7% |
| S2 equal shares | M2 CNN | EM | +0.6 / 1.3% | -1.3 / 1.7% | +4.7 / 4.9% | -4.0 / 4.2% | +0.2 / 0.2% | 3.0% |
| S2 equal shares | M3 ResNet-18 | CC | +0.4 / 1.0% | +0.8 / 1.1% | -3.0 / 3.3% | +1.8 / 2.2% | -0.1 / 0.1% | 1.9% |
| S2 equal shares | M3 ResNet-18 | AC | +0.7 / 2.2% | -1.8 / 2.6% | -4.6 / 5.8% | +5.6 / 6.5% | -0.1 / 0.2% | 4.3% |
| S2 equal shares | M3 ResNet-18 | PCC | +0.2 / 0.9% | +0.5 / 0.9% | -1.5 / 1.9% | +0.7 / 1.2% | -0.0 / 0.1% | 1.2% |
| S2 equal shares | M3 ResNet-18 | PAC | +0.6 / 1.9% | -1.3 / 2.1% | -4.4 / 5.4% | +5.1 / 5.9% | -0.1 / 0.2% | 3.8% |
| S2 equal shares | M3 ResNet-18 | EM | +0.3 / 1.0% | +0.4 / 0.8% | -2.7 / 3.0% | +2.0 / 2.3% | -0.1 / 0.1% | 1.8% |
| A | M1 LightGBM | CC | -19.8 / 19.8% | +6.2 / 6.4% | +39.0 / 39.2% | +128.9 / 129.1% | -2.7 / 2.7% | 48.6% |
| A | M1 LightGBM | AC | -0.9 / 3.2% | +0.9 / 4.4% | +0.3 / 14.7% | +5.3 / 24.5% | -0.1 / 0.4% | 11.7% |
| A | M1 LightGBM | PCC | -25.7 / 25.7% | +5.1 / 5.2% | +61.6 / 61.7% | +159.4 / 159.5% | -3.4 / 3.4% | 63.0% |
| A | M1 LightGBM | PAC | +0.9 / 2.9% | -1.2 / 3.8% | +0.5 / 12.4% | -6.0 / 20.4% | +0.2 / 0.4% | 9.9% |
| A | M1 LightGBM | EM | -4.7 / 4.8% | +2.5 / 2.9% | +15.0 / 15.8% | +14.3 / 16.3% | -0.6 / 0.6% | 10.0% |
| A | M2 CNN | CC | -7.2 / 7.2% | +2.9 / 3.1% | +19.9 / 20.1% | +31.9 / 32.2% | -0.9 / 0.9% | 15.6% |
| A | M2 CNN | AC | -0.6 / 1.7% | -3.0 / 4.2% | +9.9 / 12.0% | +1.8 / 10.1% | +0.0 / 0.2% | 7.0% |
| A | M2 CNN | PCC | -10.5 / 10.5% | +3.4 / 3.5% | +28.5 / 28.6% | +51.8 / 51.9% | -1.3 / 1.3% | 23.6% |
| A | M2 CNN | PAC | +0.7 / 1.6% | -5.0 / 5.7% | +11.7 / 13.0% | -6.5 / 10.6% | +0.3 / 0.3% | 7.7% |
| A | M2 CNN | EM | -0.8 / 0.9% | -1.4 / 1.7% | +10.7 / 11.1% | -5.3 / 6.5% | +0.0 / 0.1% | 5.1% |
| A | M3 ResNet-18 | CC | -4.6 / 4.6% | +1.1 / 1.4% | +14.3 / 14.6% | +21.1 / 21.4% | -0.6 / 0.6% | 10.5% |
| A | M3 ResNet-18 | AC | +0.5 / 1.3% | -3.1 / 3.8% | +6.6 / 9.2% | -4.0 / 10.0% | +0.2 / 0.2% | 6.1% |
| A | M3 ResNet-18 | PCC | -6.6 / 6.6% | +0.7 / 1.1% | +23.0 / 23.1% | +28.9 / 29.1% | -0.8 / 0.8% | 15.0% |
| A | M3 ResNet-18 | PAC | +0.0 / 1.1% | -1.6 / 2.6% | +7.6 / 9.5% | -7.3 / 10.4% | +0.1 / 0.2% | 5.9% |
| A | M3 ResNet-18 | EM | -1.0 / 1.1% | +0.6 / 1.0% | +2.7 / 3.7% | +3.7 / 5.1% | -0.1 / 0.1% | 2.7% |
| B | M1 LightGBM | CC | +8.1 / 8.4% | -10.8 / 10.8% | +0.8 / 3.2% | +125.8 / 126.0% | +0.4 / 0.4% | 37.1% |
| B | M1 LightGBM | AC | -0.7 / 7.6% | +0.9 / 2.1% | -1.6 / 8.3% | -7.6 / 25.5% | -0.1 / 0.3% | 10.9% |
| B | M1 LightGBM | PCC | +11.0 / 11.1% | -15.3 / 15.3% | +11.7 / 11.9% | +158.2 / 158.3% | +0.7 / 0.7% | 49.2% |
| B | M1 LightGBM | PAC | +10.1 / 11.8% | -0.4 / 1.7% | -7.1 / 9.7% | -11.1 / 23.2% | +0.4 / 0.4% | 11.6% |
| B | M1 LightGBM | EM | +4.3 / 5.1% | -1.2 / 1.3% | -2.6 / 4.3% | +9.0 / 11.6% | +0.2 / 0.2% | 5.6% |
| B | M2 CNN | CC | +10.5 / 10.7% | -6.2 / 6.2% | +2.1 / 3.1% | +51.0 / 51.3% | +0.5 / 0.5% | 17.8% |
| B | M2 CNN | AC | +1.6 / 5.0% | -0.8 / 1.5% | -1.8 / 5.3% | +10.1 / 16.8% | +0.0 / 0.2% | 7.2% |
| B | M2 CNN | PCC | +14.9 / 15.0% | -9.1 / 9.1% | +8.5 / 8.7% | +65.2 / 65.4% | +0.8 / 0.8% | 24.5% |
| B | M2 CNN | PAC | +6.0 / 7.3% | -1.8 / 2.2% | +3.3 / 5.3% | +1.0 / 11.0% | +0.3 / 0.4% | 6.4% |
| B | M2 CNN | EM | +5.4 / 5.7% | -1.4 / 1.5% | +3.2 / 3.9% | -2.5 / 4.7% | +0.3 / 0.3% | 4.0% |
| B | M3 ResNet-18 | CC | +4.8 / 5.0% | -2.9 / 3.0% | +1.0 / 2.7% | +24.7 / 25.0% | +0.2 / 0.2% | 8.9% |
| B | M3 ResNet-18 | AC | +0.5 / 3.3% | +0.8 / 1.3% | -1.3 / 5.4% | -10.6 / 15.5% | +0.0 / 0.2% | 6.4% |
| B | M3 ResNet-18 | PCC | +6.5 / 6.6% | -4.8 / 4.9% | +7.0 / 7.2% | +34.4 / 34.5% | +0.4 / 0.4% | 13.3% |
| B | M3 ResNet-18 | PAC | +0.2 / 2.9% | +1.1 / 1.4% | -2.0 / 5.0% | -12.1 / 15.4% | -0.0 / 0.1% | 6.2% |
| B | M3 ResNet-18 | EM | +1.8 / 2.3% | -0.4 / 0.5% | -1.7 / 2.9% | +3.8 / 5.2% | +0.1 / 0.1% | 2.7% |
| C | M1 LightGBM | CC | -12.5 / 12.5% | -2.8 / 3.1% | -9.0 / 9.2% | +402.5 / 403.0% | -1.7 / 1.7% | 106.9% |
| C | M1 LightGBM | AC | +0.7 / 3.6% | +1.5 / 3.1% | -2.5 / 6.0% | -10.7 / 54.8% | -0.0 / 0.3% | 16.9% |
| C | M1 LightGBM | PCC | -17.5 / 17.5% | -5.7 / 5.7% | -5.3 / 5.5% | +509.6 / 509.8% | -2.2 / 2.2% | 134.6% |
| C | M1 LightGBM | PAC | +2.9 / 4.2% | -0.2 / 2.3% | -1.2 / 5.0% | -41.0 / 59.3% | +0.3 / 0.4% | 17.7% |
| C | M1 LightGBM | EM | -2.0 / 2.5% | +0.4 / 1.2% | -0.7 / 2.3% | +40.7 / 44.5% | -0.3 / 0.3% | 12.6% |
| C | M2 CNN | CC | -3.9 / 4.0% | -1.9 / 2.0% | -3.4 / 3.7% | +150.8 / 151.4% | -0.6 / 0.6% | 40.3% |
| C | M2 CNN | AC | +0.3 / 1.9% | -1.5 / 2.4% | +2.5 / 3.9% | -9.4 / 30.0% | +0.1 / 0.2% | 9.6% |
| C | M2 CNN | PCC | -6.2 / 6.2% | -3.2 / 3.3% | -2.8 / 3.0% | +212.1 / 212.3% | -0.8 / 0.8% | 56.2% |
| C | M2 CNN | PAC | +1.5 / 2.3% | -3.1 / 3.5% | +5.3 / 5.9% | -35.7 / 42.8% | +0.3 / 0.4% | 13.6% |
| C | M2 CNN | EM | +0.3 / 0.8% | -1.6 / 1.8% | +2.9 / 3.2% | -10.8 / 13.5% | +0.1 / 0.1% | 4.8% |
| C | M3 ResNet-18 | CC | -2.4 / 2.5% | -0.7 / 0.9% | -5.9 / 6.1% | +128.8 / 129.4% | -0.5 / 0.5% | 34.7% |
| C | M3 ResNet-18 | AC | +0.7 / 1.7% | -1.3 / 2.0% | +1.0 / 3.1% | -1.2 / 26.8% | +0.1 / 0.2% | 8.4% |
| C | M3 ResNet-18 | PCC | -3.6 / 3.7% | -1.8 / 1.9% | -5.3 / 5.5% | +164.7 / 164.9% | -0.6 / 0.6% | 44.0% |
| C | M3 ResNet-18 | PAC | +0.3 / 1.4% | -0.4 / 1.4% | +1.1 / 2.9% | -12.0 / 25.4% | +0.1 / 0.2% | 7.8% |
| C | M3 ResNet-18 | EM | -0.7 / 0.9% | +0.1 / 0.6% | -1.6 / 2.0% | +29.5 / 30.9% | -0.1 / 0.1% | 8.6% |

### T5-full. Stage 1: per-genus volume error by estimator, full test set (placeholder volumes, placeholder regions)

1000 replicates of 5,000 stands; test patches and validation set resampled in every replicate. Cell = bias / RMSE, in % of the true genus volume. Scenario shares (spruce/pine/beech/oak): S1 dataset shares 21/27/23/29; S2 equal shares 25/25/25/25; A 60/25/10/5; B 15/70/10/5; C 40/35/23/2.

| scenario | model | estimator | Picea | Pinus | Fagus | Quercus | total volume | mean RMSE over genera |
|---|---|---|---|---|---|---|---|---|
| S1 dataset shares | M1 LightGBM | CC | +1.8 / 2.6% | +0.7 / 1.5% | -6.2 / 6.4% | +3.1 / 3.5% | -0.1 / 0.2% | 3.5% |
| S1 dataset shares | M1 LightGBM | AC | +0.6 / 5.0% | +2.6 / 3.9% | -0.9 / 5.0% | -2.2 / 4.6% | +0.0 / 0.3% | 4.6% |
| S1 dataset shares | M1 LightGBM | PCC | +0.5 / 1.5% | +0.6 / 1.2% | +0.2 / 1.3% | -1.1 / 1.5% | +0.0 / 0.1% | 1.4% |
| S1 dataset shares | M1 LightGBM | PAC | +3.3 / 5.4% | +1.9 / 3.2% | +0.6 / 4.4% | -4.7 / 5.8% | +0.2 / 0.3% | 4.7% |
| S1 dataset shares | M1 LightGBM | EM | +0.7 / 2.4% | +0.9 / 1.6% | +0.7 / 2.4% | -1.9 / 2.7% | +0.1 / 0.1% | 2.3% |
| S1 dataset shares | M2 CNN | CC | +2.1 / 2.5% | -0.9 / 1.3% | +2.2 / 2.7% | -2.5 / 2.8% | +0.2 / 0.2% | 2.3% |
| S1 dataset shares | M2 CNN | AC | +1.1 / 2.8% | -1.3 / 2.3% | +2.6 / 4.1% | -1.7 / 3.1% | +0.2 / 0.2% | 3.1% |
| S1 dataset shares | M2 CNN | PCC | +1.5 / 1.9% | -1.3 / 1.6% | +3.6 / 3.8% | -2.8 / 3.0% | +0.2 / 0.2% | 2.6% |
| S1 dataset shares | M2 CNN | PAC | +2.6 / 3.5% | -2.6 / 3.1% | +4.8 / 5.5% | -3.4 / 4.0% | +0.3 / 0.4% | 4.0% |
| S1 dataset shares | M2 CNN | EM | +1.6 / 2.1% | -1.5 / 1.7% | +4.8 / 5.0% | -3.7 / 3.9% | +0.3 / 0.3% | 3.2% |
| S2 equal shares | M1 LightGBM | CC | -2.5 / 3.1% | +2.4 / 2.8% | -7.6 / 7.8% | +7.8 / 8.0% | -0.5 / 0.5% | 5.4% |
| S2 equal shares | M1 LightGBM | AC | +0.8 / 4.5% | +2.6 / 4.1% | -0.9 / 4.8% | -2.6 / 5.2% | +0.0 / 0.3% | 4.7% |
| S2 equal shares | M1 LightGBM | PCC | -5.2 / 5.4% | +2.5 / 2.8% | -2.2 / 2.6% | +5.0 / 5.1% | -0.5 / 0.5% | 4.0% |
| S2 equal shares | M1 LightGBM | PAC | +3.0 / 4.8% | +1.8 / 3.3% | +0.7 / 4.1% | -5.5 / 6.8% | +0.2 / 0.4% | 4.7% |
| S2 equal shares | M1 LightGBM | EM | -0.2 / 2.0% | +1.3 / 2.0% | +0.8 / 2.4% | -1.9 / 3.0% | +0.0 / 0.1% | 2.4% |
| S2 equal shares | M2 CNN | CC | -0.1 / 1.2% | +0.1 / 1.0% | +0.7 / 1.6% | -0.7 / 1.5% | +0.0 / 0.1% | 1.3% |
| S2 equal shares | M2 CNN | AC | +0.8 / 2.4% | -1.4 / 2.5% | +2.7 / 4.0% | -2.1 / 3.6% | +0.2 / 0.2% | 3.1% |
| S2 equal shares | M2 CNN | PCC | -1.4 / 1.7% | -0.1 / 0.9% | +1.7 / 2.0% | -0.2 / 1.1% | -0.0 / 0.1% | 1.4% |
| S2 equal shares | M2 CNN | PAC | +2.1 / 2.9% | -3.0 / 3.5% | +4.9 / 5.6% | -4.1 / 4.8% | +0.3 / 0.4% | 4.2% |
| S2 equal shares | M2 CNN | EM | +1.0 / 1.6% | -1.5 / 1.8% | +4.4 / 4.6% | -3.9 / 4.2% | +0.2 / 0.3% | 3.0% |
| A | M1 LightGBM | CC | -19.6 / 19.6% | +6.2 / 6.4% | +38.1 / 38.3% | +128.2 / 128.5% | -2.7 / 2.7% | 48.2% |
| A | M1 LightGBM | AC | -0.5 / 3.0% | +0.9 / 4.4% | +0.9 / 14.4% | -0.7 / 23.9% | -0.1 / 0.4% | 11.4% |
| A | M1 LightGBM | PCC | -25.6 / 25.6% | +5.0 / 5.1% | +61.2 / 61.2% | +159.4 / 159.5% | -3.4 / 3.4% | 62.8% |
| A | M1 LightGBM | PAC | +1.2 / 2.9% | -1.1 / 3.8% | +1.0 / 12.1% | -10.4 / 22.1% | +0.2 / 0.4% | 10.2% |
| A | M1 LightGBM | EM | -4.4 / 4.5% | +2.4 / 2.8% | +14.0 / 14.7% | +13.0 / 15.2% | -0.5 / 0.5% | 9.3% |
| A | M2 CNN | CC | -7.1 / 7.1% | +2.8 / 3.1% | +20.1 / 20.3% | +31.1 / 31.4% | -0.9 / 0.9% | 15.5% |
| A | M2 CNN | AC | -0.5 / 1.6% | -3.0 / 4.2% | +11.3 / 13.1% | -1.8 / 10.0% | +0.1 / 0.2% | 7.2% |
| A | M2 CNN | PCC | -10.4 / 10.5% | +3.4 / 3.5% | +28.3 / 28.4% | +51.8 / 51.9% | -1.3 / 1.3% | 23.6% |
| A | M2 CNN | PAC | +0.7 / 1.6% | -5.0 / 5.6% | +12.4 / 13.7% | -8.9 / 12.1% | +0.3 / 0.4% | 8.2% |
| A | M2 CNN | EM | -0.7 / 0.8% | -1.5 / 1.9% | +9.7 / 10.1% | -3.8 / 5.3% | +0.0 / 0.1% | 4.5% |
| B | M1 LightGBM | CC | +9.0 / 9.3% | -10.9 / 10.9% | -0.2 / 3.2% | +125.6 / 125.9% | +0.4 / 0.4% | 37.3% |
| B | M1 LightGBM | AC | +0.7 / 7.2% | +0.9 / 2.1% | -1.8 / 8.3% | -11.6 / 26.4% | +0.0 / 0.3% | 11.0% |
| B | M1 LightGBM | PCC | +11.4 / 11.5% | -15.4 / 15.4% | +11.2 / 11.4% | +158.8 / 158.9% | +0.7 / 0.7% | 49.3% |
| B | M1 LightGBM | PAC | +10.8 / 12.2% | -0.4 / 1.7% | -7.1 / 9.7% | -13.1 / 23.7% | +0.4 / 0.5% | 11.8% |
| B | M1 LightGBM | EM | +5.1 / 5.8% | -1.3 / 1.4% | -3.5 / 5.1% | +9.6 / 12.1% | +0.2 / 0.2% | 6.1% |
| B | M2 CNN | CC | +10.5 / 10.7% | -6.2 / 6.2% | +2.6 / 3.5% | +50.2 / 50.5% | +0.5 / 0.5% | 17.7% |
| B | M2 CNN | AC | +1.3 / 4.6% | -0.7 / 1.4% | +0.1 / 5.0% | +6.2 / 14.0% | +0.1 / 0.2% | 6.3% |
| B | M2 CNN | PCC | +15.0 / 15.1% | -9.1 / 9.1% | +8.5 / 8.7% | +65.2 / 65.3% | +0.8 / 0.8% | 24.5% |
| B | M2 CNN | PAC | +5.8 / 7.0% | -1.8 / 2.1% | +4.3 / 6.0% | -1.3 / 10.5% | +0.3 / 0.4% | 6.4% |
| B | M2 CNN | EM | +5.6 / 5.9% | -1.5 / 1.6% | +2.5 / 3.4% | -0.6 / 4.1% | +0.3 / 0.3% | 3.7% |
| C | M1 LightGBM | CC | -12.4 / 12.4% | -2.8 / 3.0% | -9.1 / 9.3% | +401.8 / 402.2% | -1.7 / 1.7% | 106.7% |
| C | M1 LightGBM | AC | +0.9 / 3.7% | +1.5 / 3.2% | -2.6 / 6.0% | -15.0 / 53.2% | +0.0 / 0.3% | 16.5% |
| C | M1 LightGBM | PCC | -17.4 / 17.4% | -5.7 / 5.8% | -5.4 / 5.5% | +509.9 / 510.1% | -2.2 / 2.2% | 134.7% |
| C | M1 LightGBM | PAC | +3.1 / 4.4% | -0.2 / 2.4% | -1.3 / 4.8% | -42.3 / 59.1% | +0.3 / 0.4% | 17.7% |
| C | M1 LightGBM | EM | -1.9 / 2.3% | +0.4 / 1.1% | -0.8 / 2.3% | +40.0 / 43.9% | -0.2 / 0.3% | 12.4% |
| C | M2 CNN | CC | -3.9 / 4.0% | -1.9 / 2.1% | -3.3 / 3.6% | +150.7 / 151.3% | -0.6 / 0.6% | 40.2% |
| C | M2 CNN | AC | +0.3 / 1.9% | -1.5 / 2.5% | +2.9 / 4.2% | -13.6 / 31.8% | +0.1 / 0.2% | 10.1% |
| C | M2 CNN | PCC | -6.2 / 6.2% | -3.3 / 3.3% | -2.8 / 2.9% | +212.4 / 212.6% | -0.8 / 0.8% | 56.3% |
| C | M2 CNN | PAC | +1.5 / 2.3% | -3.2 / 3.6% | +5.4 / 6.0% | -37.9 / 44.9% | +0.4 / 0.4% | 14.2% |
| C | M2 CNN | EM | +0.3 / 0.9% | -1.7 / 1.9% | +2.6 / 2.8% | -5.5 / 9.7% | +0.1 / 0.1% | 3.8% |

### T6. Stage 1: estimator with the lowest mean per-genus RMSE, common test set (placeholder volumes, placeholder regions)

| scenario | M1 LightGBM | M2 CNN | M3 ResNet-18 |
|---|---|---|---|
| S1 dataset shares | EM 2.2% (CC 13.0%) | EM 3.1% (CC 4.8%) | EM 2.5% (CC 5.1%) |
| S2 equal shares | EM 3.0% (CC 5.9%) | CC 1.2% (CC 1.2%) | PCC 1.2% (CC 1.9%) |
| A | PAC 9.9% (CC 48.6%) | EM 5.1% (CC 15.6%) | EM 2.7% (CC 10.5%) |
| B | EM 5.6% (CC 37.1%) | EM 4.0% (CC 17.8%) | EM 2.7% (CC 8.9%) |
| C | EM 12.6% (CC 106.9%) | EM 4.8% (CC 40.3%) | PAC 7.8% (CC 34.7%) |

### T7. Stage 1: oak volume error against true oak share, common test set (placeholder volumes, placeholder regions)

Other genera in S1 (common-set) proportions. Cell = bias / RMSE in % of true oak volume.

| true oak share | model | CC | AC | PCC | PAC | EM |
|---|---|---|---|---|---|---|
| 1% | M1 LightGBM | +893.2 / 894.2% | -23.6 / 85.8% | +1135.5 / 1135.8% | -70.3 / 85.6% | +87.1 / 93.9% |
| 1% | M2 CNN | +383.3 / 384.3% | -32.1 / 65.5% | +516.8 / 517.2% | -76.9 / 83.7% | -11.5 / 17.8% |
| 1% | M3 ResNet-18 | +355.4 / 356.4% | +21.4 / 62.9% | +443.9 / 444.4% | -5.8 / 50.1% | +79.2 / 81.4% |
| 2% | M1 LightGBM | +432.2 / 432.7% | -23.6 / 58.5% | +546.7 / 546.8% | -53.8 / 67.8% | +40.1 / 44.3% |
| 2% | M2 CNN | +182.9 / 183.5% | -20.5 / 40.3% | +246.8 / 247.1% | -50.4 / 57.7% | -10.9 / 13.7% |
| 2% | M3 ResNet-18 | +170.2 / 170.8% | +13.0 / 34.5% | +212.4 / 212.7% | -0.6 / 27.3% | +41.7 / 43.0% |
| 3% | M1 LightGBM | +279.3 / 279.6% | -12.0 / 41.6% | +350.8 / 350.9% | -36.0 / 48.6% | +24.0 / 27.5% |
| 3% | M2 CNN | +116.3 / 116.7% | -10.7 / 25.9% | +157.4 / 157.5% | -31.4 / 37.0% | -9.9 / 11.6% |
| 3% | M3 ResNet-18 | +109.3 / 109.7% | +9.9 / 22.8% | +135.9 / 136.1% | +1.8 / 17.8% | +28.5 / 29.4% |
| 5% | M1 LightGBM | +155.6 / 155.8% | -6.5 / 25.3% | +193.0 / 193.1% | -22.5 / 31.1% | +10.5 / 13.5% |
| 5% | M2 CNN | +63.9 / 64.1% | -5.9 / 15.3% | +85.9 / 86.0% | -18.5 / 21.9% | -8.0 / 9.1% |
| 5% | M3 ResNet-18 | +60.5 / 60.7% | +8.1 / 14.8% | +74.4 / 74.6% | +3.1 / 11.2% | +17.5 / 18.1% |
| 7.5% | M1 LightGBM | +94.0 / 94.2% | -2.7 / 15.7% | +114.4 / 114.5% | -14.0 / 19.6% | +4.3 / 7.4% |
| 7.5% | M2 CNN | +37.0 / 37.2% | -2.7 / 9.6% | +49.8 / 49.9% | -11.4 / 13.8% | -6.9 / 7.6% |
| 7.5% | M3 ResNet-18 | +35.7 / 35.9% | +6.7 / 10.6% | +43.5 / 43.6% | +3.7 / 8.3% | +11.2 / 11.7% |
| 10% | M1 LightGBM | +63.8 / 63.9% | -0.3 / 12.4% | +75.6 / 75.6% | -8.8 / 13.8% | +1.8 / 5.2% |
| 10% | M2 CNN | +24.1 / 24.3% | -0.8 / 7.4% | +32.2 / 32.3% | -7.4 / 9.6% | -5.7 / 6.3% |
| 10% | M3 ResNet-18 | +23.7 / 23.9% | +6.3 / 8.9% | +28.3 / 28.4% | +4.2 / 6.9% | +8.2 / 8.6% |
| 15% | M1 LightGBM | +32.9 / 33.0% | +1.3 / 8.4% | +36.2 / 36.3% | -4.7 / 8.3% | -0.8 / 3.5% |
| 15% | M2 CNN | +10.8 / 11.0% | +0.4 / 5.0% | +14.2 / 14.3% | -4.3 / 5.9% | -4.9 / 5.2% |
| 15% | M3 ResNet-18 | +11.5 / 11.6% | +5.9 / 7.5% | +12.9 / 13.0% | +4.8 / 6.3% | +4.9 / 5.2% |
| 20% | M1 LightGBM | +17.7 / 17.8% | +2.8 / 7.0% | +16.6 / 16.7% | -2.2 / 5.8% | -1.7 / 3.1% |
| 20% | M2 CNN | +4.3 / 4.5% | +1.8 / 4.4% | +5.3 / 5.4% | -2.1 / 3.9% | -4.3 / 4.5% |
| 20% | M3 ResNet-18 | +5.4 / 5.6% | +5.7 / 6.9% | +5.3 / 5.4% | +4.9 / 6.0% | +3.1 / 3.5% |
| 25% | M1 LightGBM | +8.4 / 8.6% | +2.9 / 6.2% | +4.7 / 4.9% | -1.5 / 4.8% | -2.3 / 3.2% |
| 25% | M2 CNN | +0.3 / 1.4% | +1.9 / 4.0% | -0.1 / 1.0% | -1.3 / 3.3% | -3.9 / 4.1% |
| 25% | M3 ResNet-18 | +1.7 / 2.1% | +5.2 / 6.3% | +0.6 / 1.2% | +4.7 / 5.6% | +1.9 / 2.3% |
| 30% | M1 LightGBM | +2.1 / 2.6% | +3.3 / 6.1% | -3.1 / 3.3% | -0.6 / 4.4% | -2.4 / 3.0% |
| 30% | M2 CNN | -2.4 / 2.6% | +2.3 / 4.0% | -3.7 / 3.8% | -0.5 / 2.8% | -3.7 / 3.9% |
| 30% | M3 ResNet-18 | -0.8 / 1.3% | +5.2 / 6.1% | -2.5 / 2.6% | +4.9 / 5.7% | +1.1 / 1.5% |

### T8. Stage 4: test accuracy by label purity (run 1 models; Wilson 95% intervals)

Purity = area share of the labelled genus in the archive's multi-label JSON.

| test set | grouping | level | n | M1 LightGBM | M2 CNN | M3 ResNet-18 |
|---|---|---|---|---|---|---|
| common | purity | < 0.6 | 232 | 59.1% (52.6% to 65.2%) | 72.0% (65.9% to 77.4%) | 76.7% (70.9% to 81.7%) |
| common | purity | 0.6 to < 0.9 | 576 | 61.5% (57.4% to 65.3%) | 85.2% (82.1% to 87.9%) | 88.9% (86.1% to 91.2%) |
| common | purity | >= 0.9 | 1514 | 88.4% (86.7% to 90.0%) | 94.5% (93.2% to 95.5%) | 95.4% (94.3% to 96.4%) |
| common | json_top_agrees | agrees | 2236 | 79.8% (78.1% to 81.4%) | 90.9% (89.7% to 92.0%) | 92.7% (91.5% to 93.7%) |
| common | json_top_agrees | disagrees | 86 | 53.5% (43.0% to 63.7%) | 64.0% (53.4% to 73.3%) | 73.3% (63.1% to 81.5%) |
| common | all | all | 2322 | 78.8% | 89.9% | 91.9% |
| full | purity | < 0.6 | 335 | 64.2% (58.9% to 69.1%) | 75.2% (70.3% to 79.5%) | - |
| full | purity | 0.6 to < 0.9 | 683 | 63.4% (59.7% to 66.9%) | 84.6% (81.7% to 87.1%) | - |
| full | purity | >= 0.9 | 1720 | 88.0% (86.3% to 89.4%) | 94.2% (93.0% to 95.2%) | - |
| full | json_top_agrees | agrees | 2632 | 80.1% (78.5% to 81.6%) | 90.5% (89.3% to 91.6%) | - |
| full | json_top_agrees | disagrees | 106 | 50.0% (40.6% to 59.4%) | 64.2% (54.7% to 72.6%) | - |
| full | all | all | 2738 | 78.9% | 89.5% | - |
