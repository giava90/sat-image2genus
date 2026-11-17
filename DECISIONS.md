# DECISIONS

Choices made without the user during the unattended run. Each entry: what came up, what was chosen, why, how to redo it differently.

## D1. Python interpreter
- **Came up:** no Python on PATH (only the Microsoft Store stub). Found MSYS2 Python 3.11.7, an Anaconda install, and a uv-managed CPython 3.12.13.
- **Chose:** create `.venv` from the uv-managed CPython 3.12.13 (`%APPDATA%\uv\python\cpython-3.12.13-windows-x86_64-none\python.exe`). Nothing was installed outside the folder.
- **Why:** MSYS2 Python cannot install the official PyPI torch wheels; Anaconda would mix in a conda base. A plain CPython ≥ 3.11 is what the brief asks for.
- **Redo:** delete `.venv` and run `<other python> -m venv .venv`, then `pip install -r requirements.txt`.

## D2. Compute device
- **Came up:** no CUDA GPU (Intel UHD only) and no MPS (Windows).
- **Chose:** CPU for all models (Intel i7-11850H, 8 cores / 16 threads, 32 GB RAM).
- **Why:** it is the only device available. This makes model 3 the binding constraint for the 2 h compute budget.
- **Redo:** on a CUDA machine the scripts pick `cuda` automatically (`src/common.py: device()`).

## D3. Where the label lives
- **Came up:** the brief asked to check whether the single label is in the file name or in JSON.
- **Found:** `labels/TreeSatBA_v9_60m_multi_labels.json` holds, for every patch, a list of `[genus, area proportion]` pairs (multi-label, genus level). The species single label is the file name prefix, e.g. `Picea_abies_3_1234_WEFL_NLF.tif`. Aerial zips are split by this file-name species.
- **Chose:** the class of a patch is the genus of its file-name species. The JSON proportions are not used for the class.
- **Why:** the file name is the archive's single label, and it is how the aerial files are organised, so all three models see the same label. Across all 50,381 patches the file-name genus differs from the JSON's largest-share genus for 3,418 patches (6.8%). This label noise is part of the task for every model.
- **Redo:** change `genus_of()` in `src/data.py` to use the JSON top genus.

## D4. Data inside a OneDrive-synced folder
- **Came up:** the project folder is under OneDrive, so `data/` (several GB) will be synced.
- **Chose:** keep data under `data/` as the rules require. Aerial zips are not extracted. They are read directly from the zip and turned into a small downscaled cache, which limits disk use to about the size of the zips.
- **Redo:** move the folder outside OneDrive; nothing in the code depends on the location.

## D5. Sentinel-2 band order
- **Came up:** the S2 GeoTIFFs carry no band descriptions, and the archive's documentation PDF is password-protected.
- **Chose:** the order B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09 (0-based: blue=0, green=1, red=2, NIR=3, red-edge 4-7, SWIR 8-9). This is the order I remember from the TreeSatAI documentation. It also fits the spectral signature of a spruce patch: means 167, 372, 251, 2593, 685, 2035, 2476, 2744, 1367, 670, 159, 2743 (low red, high NIR, rising red-edge, two SWIR bands).
- **Why it matters:** only for the hand-made indices in model 1 (NDVI = B08/B04, NDRE = B8A/B05, NDMI = B08/B11). The CNN of model 2 uses all bands and does not depend on the order.
- **Redo:** change `BANDS` in `src/features.py`.

## D6. Download level: fallback level 1 (one or more aerial files per genus)
- **Came up:** the record has 27 files, 16.27 GB in total. The aerial files of the four genera are: Picea abies 1.78 GB; Pinus sylvestris 2.02, P. nigra 0.12, P. strobus 0.16; Fagus sylvatica 1.96; Quercus robur 1.07, Q. petraea 0.81, Q. rubra 0.58. All of them plus Sentinel-2 would come to about 9.0 GB, which is over the 8 GB cap.
- **Chose:** level 1, read as the brief describes it: "pick aerial files starting with the most frequent species of each genus". Patch counts from the labels: Picea abies 5,783; Pinus sylvestris 6,591; Fagus sylvatica 6,459; Quercus robur 3,453 (Q. petraea 2,608, Q. rubra 1,841). I downloaded S2 + labels + split lists + doc PDF + the most frequent species of each genus, and then the two small Pinus files (nigra, strobus) because they still fitted. Q. petraea (+0.81 GB) and Q. rubra (+0.58 GB) would have exceeded the cap.
- **Total downloaded:** 7.616 GB (ledger in `data/download_ledger.csv`). Every Zenodo md5 matched. No file failed, so the level-2 fallback (conifer vs broadleaf only for model 3) was not needed.
- **Consequence:** aerial data covers all Picea, Pinus and Fagus patches, but only Quercus robur among oaks (3,453 of 7,902 Quercus patches). Model 3 is still a four-genus model.
- **Alternative reading:** if "aerial files for the four genera" meant every species file, step 1 exceeds the cap and level 2 applies. To redo it that way: delete the Pinus and Quercus zips, rerun `src/prepare.py`, and set `--task binary` in `src/train_m3_resnet.py`.

## D7. "Same test split" when aerial coverage is partial
- **Came up:** model 3 can only be evaluated on test patches whose species has aerial data (no Q. petraea or Q. rubra).
- **Chose:** two evaluation sets.
  - **common test set:** test patches of the downloaded species (Picea abies, Pinus sylvestris/nigra/strobus, Fagus sylvatica, Quercus robur). All three models are scored on exactly these patches. This is the headline comparison and the basis of the volume step.
  - **full test set:** all test patches of the four genera. Only models 1 and 2 can be scored on it.
  - Validation is restricted the same way for the confusion-matrix adjustment, so the validation matrix matches the population it is applied to.
- **Why:** the brief requires the same test split for every model. Restricting all three models to the common subset keeps the comparison exact; reporting models 1 and 2 on the full set as well shows what the restriction changes.
- **Redo:** `--eval-set full` in `src/volume_step.py` uses the full set for models 1 and 2.

## D8. Validation set
- **Chose:** 10% of the archive's training list, drawn at random within each species, seed 0. This gives 2,493 validation patches for the four genera.
- **Why:** the brief asks for a carve-out from the training split. Stratifying by species keeps the Quercus species mix the same as in training. The archive's lists carry no stand IDs, so neighbouring patches from one stand may sit on both sides of the split. That would inflate validation and test scores equally for all models. I cannot check this without the geojson, which was not downloaded to stay inside the cap.
- **Redo:** change `VAL_FRACTION` / `SEED` in `src/data.py`.

## D9. S2 patch size irregularities
- **Came up:** 48 of the 27,656 target 200 m patches are 21 px on one side.
- **Chose:** keep the first 20 rows and columns.

## D10. Aerial band order and first-layer adaptation (model 3)
- **Came up:** aerial GeoTIFFs are 4 x 304 x 304 uint8, tagged R, G, B, alpha by GDAL default. The brief says the bands are infrared, green, blue, red.
- **Checked:** mean band values over the training patches (bands 0-3) are Picea 126/88/82/78, Pinus 149/88/79/77, Fagus 155/102/93/90, Quercus 153/92/87/80. Band 0 is by far the brightest over vegetation and brighter over broadleaf, as NIR should be, so I kept the brief's order NIR, G, B, R.
- **Chose:** keep all 4 bands and widen the first convolution to 4 inputs (not drop a band). The pretrained R, G and B filters go to the matching aerial bands; the NIR filter starts as the mean of the three RGB filters; all four are scaled by 0.75 so first-layer activations keep their size. Inputs are standardised per band with training-set statistics, not ImageNet statistics.
- **Redo:** `resnet18_4band()` in `src/nets.py`.

## D11. Sizing from the timed pilot (CPU only)
- **Pilot (`results/pilot_timing.json`):** model 2 trains at 1,705 img/s. ResNet-18 full fine-tune: 64 img/s at 112 px, 50 at 128, 32 at 160, 16 at 224. Training only the last block and head at 128 px: 122 img/s.
- **Chose for model 3:** full fine-tune (all layers) at 128 px (304 px area-downscaled, about 47 cm/pixel), training capped at 3,000 patches per genus (Quercus robur has only 2,790, so 11,790 in total), 12 epochs, batch 64, AdamW lr 5e-4, cosine schedule with 1 warm-up epoch. Expected about 50-55 min. There is a hard time budget of 80 min, after which training stops at the end of an epoch.
- **Why:** a full fine-tune fits the 2 h budget at 128 px, so freezing was not needed. 224 px would have allowed only about 4 epochs on the capped set.
- **Chose for model 2:** all 22,425 training patches, 40 epochs, batch 256, AdamW lr 2e-3, wd 1e-4, cosine schedule, flips and transposes as augmentation. Expected about 10 min.
- **Chose for model 1:** LightGBM, lr 0.05, 63 leaves, up to 3,000 rounds, early stopping after 100 rounds without improvement on validation log-loss. No hyperparameter search for any model. The first complete run is the result.
- **Model selection:** for models 2 and 3, the epoch with the best validation accuracy is kept.
- **Class balance:** models 1 and 2 train on the natural class mix; model 3's cap makes its training mix nearly balanced. No class weights for any model. The prior shift this causes in model 3 is part of what the confusion-matrix adjustment in the volume step has to correct.
- **Redo:** command-line flags of `src/train_m2_cnn.py` and `src/train_m3_resnet.py`.

## D12. Added diagnostic: train-test spatial proximity (not in the brief)
- **Came up:** model 2 beat model 1 by about 10 accuracy points. The split lists carry no stand IDs, and neighbouring patches' 200 m Sentinel-2 windows overlap. A CNN could exploit overlapping pixels it saw in training.
- **Chose:** read each patch centre from the GeoTIFF georeferencing (no extra download), compute for each test patch the distance to the nearest training patch, and report every model's accuracy by distance band (`src/spatial_check.py`, `results/spatial_check.csv`). This is a diagnostic only. The headline numbers are unchanged and nothing was retrained because of it.
- **Redo:** delete `results/spatial_check.csv`; nothing else depends on it.

## D13. Volume-step design (brief silent on the details)
- **Region:** 10,000 ha (only relative errors are reported, so the size does not matter). Stands of true genus i are labelled j with the test-set rate P(j|i). The expected labelled area is used, so the region is treated as large and its own sampling noise is ignored.
- **Scenario S1, "the dataset's shares":** genus shares among all 27,656 four-genus patches of the archive (train + validation + test).
- **Adjustment:** solve C_val^T x = counted area, with C_val the model's row-normalised validation confusion matrix on the same evaluation population. Negative areas are clipped to 0 and rescaled so the total area is unchanged.
- **Bootstrap:** 1,000 resamples of the test patches, with the validation matrix held fixed. The brief asks for a test-set bootstrap; resampling validation as well would widen the adjusted intervals. This is stated as a limitation.
- **Why:** these are the simplest choices that implement the brief, and they keep the step to a few seconds.
- **Redo:** `src/volume.py` (estimators) and `src/volume_step.py` (`TOTAL_AREA_HA`, `scenarios()`, `--n-boot`).

# Run 2

Follow-up run (started 2026-11-16 21:17 UTC) testing three doubts from a review of run 1. Nothing in `results/` or `RESULTS.md` was changed; run 2 outputs are in `results_v2/` and `RESULTS_V2.md`. Same entry format as above.

## R2-1. regions.yaml
- **Came up:** the file did not exist.
- **Chose:** created it with the three prompt scenarios, marked `placeholder: true`: A 60/25/10/5, B 15/70/10/5, C 40/35/23/2 (spruce/pine/beech/oak). S1 and S2 are added by the script. S1 for the common set uses the genus shares of all common-set patches (train + val + test, 23,207 patches of species with aerial data); for the full set it uses all 27,656 four-genus patches as in run 1.
- **Redo:** edit `regions.yaml` and rerun `src/v2_stage1_volume.py` (about 15 min on this machine).

## R2-2. Synthetic region construction (stage 1)
- **Came up:** "weighted resampling of test patches to the target shares" leaves open whether the genus counts are fixed or random.
- **Chose:** fixed counts. Each replicate draws exactly round(5,000 x share) stands of each genus (largest-remainder rounding), with replacement from that genus's test patches. The true shares are therefore known exactly in every replicate; the noise comes from which patches are drawn. In the same replicate the validation set is bootstrapped (resampled with replacement, all genera together) and the validation matrices for AC and PAC are recomputed from it. The same random draws are used for every model within a scenario (paired comparison).
- **Error measure:** relative error of the estimated genus share = relative error of genus area = relative error of genus volume (growing stock cancels). Total volume error uses `volumes.yaml` (placeholder values). Bias = mean error over replicates, RMSE = root mean squared error, 95% interval = 2.5th-97.5th percentile.
- **Redo:** `simulate()` in `src/v2_stage1_volume.py`; use `rng.multinomial` for random counts.

## R2-3. EM starting prior and the definition of "trained on"
- **Chose:** models 1 and 2: genus shares of run 1's training split (natural mix). Model 3: its capped training mix from `results/run_m3_resnet18.json` (`n_train_per_genus`). EM tolerance 1e-8 on the max change of the prior, at most 1,000 iterations. The same starting prior is used on the common and full evaluation sets, since the models were trained once.
- **Redo:** `train_prior()` in `src/v2_stage1_volume.py`.

## R2-4. Probabilistic adjusted count
- **Chose:** PAC solves M_val^T x = PCC, where M_val[i, j] is the mean predicted probability of genus j over validation patches of true genus i (Bella et al. 2010 style). Negative solutions are clipped and renormalised, as for AC in run 1.

## R2-5. Position of the 60 m patch inside the 200 m patch (stage 2)
- **Came up:** the prompt asked to verify how `s2/60m` relates to the centre of `s2/200m`.
- **Found:** in 1,000 randomly sampled four-genus patches, the 60 m patch (12 x 6 x 6) equals pixels [7:13, 7:13] of the 200 m patch exactly, in every band, and the georeferenced top-left corner sits exactly 7 pixels right and down (`results_v2/s2_60m_vs_200m.json`). No other offset matched.
- **Chose:** take the 60 m patch as the centre crop [7:13, 7:13] of run 1's 200 m cache instead of reading 27,656 more files. Caveat: the 48 patches that were 21 px wide on one side and cropped to 20 in run 1 were not specifically checked; if their extra row or column is at the top or left, their centre is off by one pixel (0.17% of patches).
- **Redo:** read `s2/60m/<file>` directly in `src/v2_features.py`.

## R2-6. CNN variants (stage 2 e, f)
- **e, centre 6 x 6 only:** the same SmallCNN, with the second max-pool replaced by identity, so the 6 x 6 input becomes 3 x 3 after the first pool and the last two convolutions plus global average pooling run on 3 x 3. Same parameter count (290,564). Per-band standardisation uses training statistics of the 6 x 6 crops.
- **f, centre masked:** the 200 m patch after per-band standardisation, with the centre 6 x 6 set to 0 (= the training mean) in train, validation and test. The mask is symmetric under the dihedral augmentation.
- Same recipe as run 1: 40 epochs, AdamW lr 2e-3, batch 256, weight decay 1e-4, cosine schedule, best epoch by validation accuracy, seed 0. A 15-minute time cap per run was set; it is recorded in each `run_*.json` whether it was reached.

## R2-7. Spatial blocks and blocked split (stage 3)
- **Blocks:** 2 km x 2 km grid cells anchored at (0, 0) in the patches' CRS (one EPSG code for all patches), on the patch centres in `results/patch_centres.parquet`. 2,281 non-empty blocks.
- **Blocked split:** the whole four-genus pool (run 1 train + val + test, 27,656 patches) is regrouped. Blocks are shuffled with a fixed seed and taken in order until 20% of patches are in test, then 10% in validation; all other blocks are training. Training patches whose centre is within 200 m (Chebyshev distance, so their 200 m windows would overlap a held-out window) of any test or validation patch are dropped. Seed 0: 18,639 train / 2,791 val / 5,536 test / 690 dropped. Seed 1: 18,642 / 2,785 / 5,547 / 682 (`results_v2/blocked_splits_summary.json`).
- **Common subset on the blocked split:** test patches of species with aerial data, for comparability with run 1's common set. Model 3 is not retrained on the blocked split (brief for run 2), so this subset only matters for comparing set composition.
- **Redo:** `src/v2_splits.py --seeds ...`; block size and buffer in `src/blocks.py`.

## R2-8. Block bootstrap
- **Chose:** resample the 2 km blocks that contain test patches, with replacement, as many blocks as there are, keeping all test patches of each drawn block. 1,000 resamples, shared by all models in a comparison (paired). Reported next to the patch bootstrap, which is recomputed in run 2 with the same code.

## R2-9. Coordinate-only baselines
- **Chose:** k-nearest-neighbour vote (k = 1 and 5) on patch centres, Euclidean distance, training patches only; LightGBM with run 1's settings on (x, y) with early stopping on validation. They use no image data at all.

## R2-10. Label purity bins (stage 4)
- **Found:** `json_share_of_label` exceeds 1 for 19 of the 27,656 four-genus patches (maximum 2.0), an anomaly in the archive's JSON. They are put in the top bin (>= 0.9). Bins: < 0.6, 0.6 to < 0.9, >= 0.9; Wilson 95% intervals.

## R2-11. Masked-centre CNN stopped at epoch 38 of 40
- **Came up:** variant f ran while the stage 1 simulation used the same CPU, so epochs took about 24 s instead of 16 s, and the 15-minute cap per run (R2-6) was reached after epoch 38 of 40.
- **Chose:** keep the run as is (first complete run; best epoch 36, chosen on validation as always). The cosine schedule had not fully decayed, so this variant is at a slight disadvantage against run 1's model 2, which ran all 40 epochs. Its validation curve had flattened (see `results_v2/stage2/run_cnn_masked_centre.json`).
- **Redo:** `python src/v2_train_cnn.py --variant masked --name cnn_masked_centre --budget-min 30`.

## R2-12. One block draw for the trained models
- **Came up:** the prompt allows a second block draw "if time allows". After stage 2 and the seed 0 blocked runs, training time was 2,863 s (47.7 min), already above the 45-min budget, mostly because two runs (masked CNN and raw-pixel 200 m LightGBM) were slowed by the concurrent stage 1 simulation.
- **Chose:** `src/v2_queue2.py` skipped seed 1 for LightGBM and the CNN by its pre-set rule (skip if used + 14 min > 45 min). The seed 1 split exists and the kNN baselines (no training) were scored on it.
- **Redo:** run the four seed 1 commands listed in `src/v2_queue2.py`, then `src/v2_stage3_eval.py` and `src/v2_make_tables.py`.

## R2-13. Choice of the "best fair baseline"
- **Chose:** the LightGBM variant with the highest validation accuracy among a-d: centre + ring statistics (validation 87.2%; a 86.3%, c 86.0%, d 84.9%). It also has the highest test accuracy of the four, but test accuracy was not used for the choice.

## R2-14. Blog post (BLOG_POST.md): check against style_notes.md
- **Audience and sources:** technical non-specialist readers; written from RESULTS.md and RESULTS_V2.md only; numbers checked against `results/` and `results_v2/` files (27,656 patches and the run 1 spruce sentence from `results/`, the rest from `results_v2/`). 778 words after the three title options. Two figures named: `fig_v2_cnn_lead.png` and `fig_v2_oak_sweep.png`. Not published or sent anywhere.
- **Followed:** plain declarative opening; order context, turn, what was done, findings, limits, implication; findings counted out as First, Second, Third (as paragraphs, no bullet list); exactly one question; one worked numerical example (the 60/25/10/5 region); "I" for the author, with one sentence stating that an AI coding agent ran the experiments unattended; limitations in their own paragraph, followed by what still holds; closing "To conclude" paragraph with a practical implication; "that is" and "for example" instead of i.e./e.g.; each technical term defined at first use; no hooks, slogans, exclamation marks, metaphors, novelty claims, citations, roadmap or closing question.
- **Departures:**
  1. Passive voice in several sentences ("three models were trained", "errors were converted"). Chosen so the text does not imply that the author ran the training, since the agent did; the notes ask for active voice.
  2. 8 of 56 sentences are longer than 20 words (measured before the final edits, which added a few words to two sentences), for example the question and the closing recommendation. The notes say "mostly under 20 words", which holds, but these sentences carry more than one idea.
  3. Markers: 12 of about 56 sentences start with a marker, within the "about one in three" limit, but some (Nevertheless, Instead, To conclude, First/Second/Third) are outside the preferred four (However, Indeed, Specifically, In other words). "First/Second/Third" and "To conclude" are required by other parts of the notes.
  4. The "one concrete numerical example" is accompanied by many other numbers in the findings. They report results rather than illustrate a mechanism, but the post is number-heavy for a non-specialist reader.
  5. One short stand-alone sentence ("Three findings remain.") is used as a transition; the notes say such sentences are rare in the papers but do not forbid them, and it is not a one-line paragraph.

## R2-15. Placeholder labels in the blog post (after run 2, at the user's request)
- **Came up:** the user found the repeated "(placeholder volumes)" and "(placeholder volumes and regions)" parentheticals awkward to read.
- **Chose:** removed the inline labels and added one sentence before the volume numbers stating that they use placeholder growing stocks and synthetic regions, and why that suffices (per-genus percentage errors do not depend on the growing stock). The limitations paragraph still names the placeholders. RESULTS.md and RESULTS_V2.md keep the per-number labels.
