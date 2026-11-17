# STATUS

Run started: 2026-11-16 16:47 UTC. Wall-clock limit: 5 h, so hard stop at 21:47 UTC.
Device: CPU only (Intel i7-11850H, 8 cores / 16 threads, 32 GB RAM, no CUDA, no MPS).

## Stages

| # | Stage | State |
|---|-------|-------|
| 1 | Environment and folder layout | done 16:56 UTC |
| 2 | Inspect Zenodo record, choose download level, download | done 17:01 UTC (level 1, 7.616 GB) |
| 3 | Data loading, metric and volume functions, with unit tests | done 17:03 UTC |
| 4 | Timed pilot per model, size the runs | done 17:09 UTC |
| 5 | Train and evaluate models 1, 2, 3; save predictions | done 18:28 UTC |
| 6 | Volume step, figures, RESULTS.md, README.md | done 18:35 UTC |
| 7 | Final status line | done 18:35 UTC |

## Log

- 2026-11-16 16:47 UTC: Stage 1 started.
- 2026-11-16 16:56 UTC: Stage 1 done. venv with torch 2.15.0+cpu, lightgbm 4.7.0, rasterio 1.5.2; 15 unit tests pass.
- 2026-11-16 16:52 UTC: Stage 2 started (record listed at 16:52; downloads running since 16:53).
- 2026-11-16 17:01 UTC: Stage 2 done. Fallback level 1: S2 + labels + split lists + aerial for Picea abies, Pinus sylvestris/nigra/strobus, Fagus sylvatica, Quercus robur. 7.616 GB, all md5 ok. See DECISIONS D6.
- 2026-11-16 16:58 UTC: Stage 3 started (overlapping with downloads). Index and S2 cache built (27,656 patches); LightGBM features written; 18 tests pass.
- 2026-11-16 17:03 UTC: Stage 3 done (index, S2 cache, features, metric and volume functions; 18 tests pass). Stage 4 started: timed pilot.
- 2026-11-16 17:09 UTC: Stage 4 done. Pilot timings in results/pilot_timing.json; sizing in DECISIONS D11. Model 3 smoke test (32 px, 1 epoch) passed; its outputs were deleted.
- 2026-11-16 17:09 UTC: Stage 5 started. Model 1 (LightGBM) done 17:07 UTC: full-test accuracy 0.789, 9.8 s training.
- 2026-11-16 17:20 UTC: Model 2 (small CNN) done 17:20 UTC: full-test accuracy 0.895, 658 s training, best epoch 37 of 40. Model 3 started 17:20 UTC.
- 2026-11-16 18:28 UTC: Model 3 (ResNet-18, aerial, 128 px, full fine-tune) done 18:27:50 UTC: common-test accuracy 0.919, 3,637 s training (60.6 min), best epoch 12 of 12. Aerial cache took 356 s. Total training time of the three models: about 72 min. Stage 5 done.
- 2026-11-16 18:28 UTC: Stage 6 started.
- 2026-11-16 18:35 UTC: Stage 6 done. Evaluation (results/metrics.json), volume step (results/volume_summary*.csv; reruns in about 4 s), spatial diagnostic, figures 1-3 (+ fig3 for the full test set), RESULTS.md, README.md. 18 tests pass. Elapsed wall clock: about 1 h 48 min of the 5 h limit; download 7.663 GB of the 8 GB cap; training 71.8 min of the 2 h budget.

DONE. All three models trained and evaluated on the same (common) test patches; volume step, three figures, RESULTS.md and README.md are complete. Deviations (all in DECISIONS.md and RESULTS.md): fallback level 1 with aerial data for Quercus robur only among oaks, so the three-way comparison uses the common test set (2,322 patches) instead of the full four-genus test split (2,738), on which only models 1 and 2 were scored; model 3 ran at 128 px on CPU. Volume figures use placeholder volumes.

# Run 2

Follow-up run testing three doubts from the review of run 1. Start 2026-11-16 21:17 UTC; wall-clock limit 2.5 h (hard stop 23:47 UTC); training budget about 45 min; no downloads. Outputs go to `results_v2/` and `RESULTS_V2.md`; `results/` and `RESULTS.md` are not touched.

| Stage | Description | State |
|---|---|---|
| 1 | Volume errors under realistic genus shares, five estimators (no training) | done 21:44 UTC |
| 2 | Fairness of the LightGBM baseline (60 m / centre-ring / raw-pixel variants, centre-only and centre-masked CNN) | done 21:59 UTC |
| 3 | Spatial independence (coordinate baselines, block bootstrap, blocked split) | done 22:11 UTC (one block draw) |
| 4 | Label purity | done 21:26 UTC |
| 5 | RESULTS_V2.md | done 22:14 UTC |
| 6 | BLOG_POST.md | done 22:15 UTC |

## Run 2 log

- 2026-11-16 21:17 UTC: Run 2 started. Read CLAUDE.md, BRIEF.md, RESULTS.md, DECISIONS.md and the run 1 code.
- 2026-11-16 21:24 UTC: Stage 1 started. regions.yaml created (A 60/25/10/5, B 15/70/10/5, C 40/35/23/2, placeholder). Estimators and region sampler unit-tested (tests/test_quantify.py, 10 pass). Stage 2/3 training queue running in the background since 21:21 UTC (logs/run2_train_queue.log); feature and block helpers unit-tested (tests/test_v2_features.py, tests/test_blocks.py).
- 2026-11-16 21:44 UTC: Stage 1 done (1,214 s of simulation). results_v2/stage1_volume_summary.csv, stage1_oak_sweep.csv, stage1_meta.json, stage1_volume_replicates.parquet, fig_v2_rmse_by_estimator_{common,full}.png, fig_v2_oak_sweep.png. Stage 4 (label purity, run 1 predictions only) also done: results_v2/stage4_purity.csv.
- 2026-11-16 21:59 UTC: Stage 2 done. All six variants trained (plus a reproduction of run 1 model 1: identical test accuracy 0.7893). Best LightGBM variant by validation accuracy: centre + ring statistics. results_v2/stage2_accuracy.csv, stage2_paired.csv, stage2_best_variant.json. The masked-centre CNN hit its 15-min cap at epoch 38/40 (DECISIONS R2-11).
- 2026-11-16 22:11 UTC: Stage 3 done. Coordinate baselines (kNN, LightGBM on x/y) on the original split and two blocked draws; block bootstrap of run 1; blocked split seed0 with LightGBM (run 1 features), the centre+ring variant, LightGBM on x/y, kNN and the CNN. Total run 2 training time 2,863 s (47.7 min, over the 45-min budget by 2.7 min); the second block draw (seed1) was therefore scored only with the kNN baselines (which need no training), not with LightGBM or the CNN (DECISIONS R2-12). results_v2/stage3_accuracy.csv, stage3_paired.csv, tables_v2.md.
- 2026-11-16 22:14 UTC: Stage 5 done. RESULTS_V2.md (tables appended verbatim from results_v2/tables_v2.md), results_v2/fig_v2_cnn_lead.png, results_v2/run2_facts.json, README run 2 section. 41 unit tests pass.
- 2026-11-16 22:15 UTC: Stage 6 done. BLOG_POST.md (three title options, 778 words, two figures referenced, not published); departures from style_notes.md listed in DECISIONS R2-14. Run 2 elapsed wall clock: about 1 h of the 2.5 h limit; training 47.7 min (budget about 45 min).

DONE (run 2). All six stages completed. RESULTS_V2.md, BLOG_POST.md, results_v2/ (tables, figures fig_v2_rmse_by_estimator_{common,full}.png, fig_v2_oak_sweep.png, fig_v2_cnn_lead.png). What is missing or reduced: the second blocked-split draw was scored only with the kNN baselines, not with LightGBM or the CNN (training budget, DECISIONS R2-12); model 3 was not retrained on the blocked split (as the prompt specified); the masked-centre CNN stopped at epoch 38 of 40 (R2-11). results/ and RESULTS.md are unchanged. All volume figures use placeholder volumes and placeholder regions.
