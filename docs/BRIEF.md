# Project B: tree genus from satellite and aerial images, and what the error does to wood volume

This brief is written for an unattended run. Read `CLAUDE.md` first: it holds the rules for working without a human.

## Question

How well can tree genus be classified from open imagery, does that need deep learning, and what does the remaining error do to a wood-volume estimate?

Background: the dataset's own paper reports about 74% weighted precision on Sentinel-2 alone with a plain MLP or LightGBM, and about 79% with a ResNet on 20 cm aerial photos (Ahlswede et al., https://essd.copernicus.org/preprints/essd-2022-312/essd-2022-312.pdf). So "this needs deep learning" is a claim to test, not an assumption.

## Data

TreeSatAI Benchmark Archive, Zenodo record 6780578: https://zenodo.org/records/6780578

- 50,381 labelled forest patches from Lower Saxony, Germany.
- Aerial: 20 cm, 60 m patches, 4 bands (infrared, green, blue, red).
- Sentinel-2: 10 m, 12 bands, patches of 6 x 6 px (60 m) or 20 x 20 px (200 m).
- Labels: 20 species in 15 genera. Fixed 90/10 train/test split, given as file lists.
- Full archive is about 20 GB, mostly aerial. Aerial files are split per species, roughly 0.8 GB each.

Not verified, check first: exact file names, exact sizes, and how the single-label species or genus is stored (file name or JSON). List the record's files before downloading anything, for example through the Zenodo REST API.

### Target classes

Four genera: spruce (Picea), pine (Pinus), beech (Fagus), oak (Quercus). Also report the two-class split conifer (Picea, Pinus) versus broadleaf (Fagus, Quercus).

Aerial files are per species and a genus can span several species. Map species to genus from the labels, then pick aerial files starting with the most frequent species of each genus.

### Download plan and fallbacks

Download cap: 8 GB in total. Decide in this order and log the choice in `DECISIONS.md`.

1. Sentinel-2 archive, labels, split lists, and aerial files for the four genera.
2. If step 1 exceeds the cap or an aerial file fails: aerial for one species each of Picea and Fagus only. Model 3 then becomes conifer versus broadleaf.
3. If no aerial data can be obtained: skip model 3 and run models 1 and 2 on Sentinel-2.
4. If TreeSatAI cannot be downloaded at all: use EuroSAT RGB (https://zenodo.org/record/7711810, 27,000 patches of 64 x 64 px, 10 classes, under 100 MB). The project then becomes forest versus non-forest. Skip the volume step and say why.

Verify checksums when Zenodo provides them. Keep all downloads under `data/`.

## Three models, same test split

Use the archive's fixed test split for every model, restricted to the target classes. Carve a validation set out of the training split.

1. **No deep learning.** LightGBM on per-band summary statistics (mean, standard deviation, a few percentiles, simple indices such as NDVI) of the 200 m Sentinel-2 patches.
2. **Small network on satellite data.** A compact CNN trained from scratch on the 20 x 20 x 12 Sentinel-2 patches.
3. **Pretrained network on aerial data.** ImageNet-pretrained ResNet-18 fine-tuned on aerial patches. Either adapt the first convolution to four bands or drop one band, and log which.

Compute budget: about 2 hours of training in total across the three models. Run a timed pilot first and size model 3 to fit: cap patches per genus, downscale the images, and fine-tune only the last block and the head if a full fine-tune is too slow. Use MPS or CUDA when available, otherwise CPU. Record the device.

## From accuracy to wood volume

1. Take each model's confusion matrix on the test split.
2. Build a synthetic region of stands with known true genus shares. Scenario 1: the dataset's shares. Scenario 2: equal shares.
3. Estimate area per genus two ways: by counting predictions, and with a confusion-matrix adjustment (the matrix estimated on the validation set, applied to test predictions).
4. Convert area to volume with the growing stock per genus in `volumes.yaml`. Those values are PLACEHOLDERS. Print "placeholder volumes" in the title or caption of every figure that uses them, and in `RESULTS.md`.
5. Bootstrap the test set (at least 1,000 resamples) for the distribution of volume error, per genus and in total.

Keep the volume step in its own script that reads saved predictions, so it reruns in seconds when the user replaces `volumes.yaml`.

## Figures

1. Accuracy and per-genus precision and recall for the three models, plus the conifer versus broadleaf result.
2. Confusion matrices side by side.
3. Volume error per genus for each model, with and without the adjustment.

## RESULTS.md must state

- Whether the deep models beat LightGBM on the same test split, and by how much.
- Whether the accuracy gap between models changes the volume estimate, or disappears after the adjustment.
- One sentence of the form: a classifier with X% accuracy misstates spruce volume by Y% (placeholder volumes).
- Which fallback level was used and every deviation from this brief.
- Limitations: one German state, summer imagery only, patches from administratively labelled stands, synthetic region, placeholder volumes.

## Environment and layout

Python 3.11 or newer in a fresh virtual environment inside this folder. Packages: torch, torchvision, pandas, scikit-learn, lightgbm, rasterio, matplotlib, pytest, pyarrow.

```
src/          experiment code, small pure functions
tests/        unit tests for the metric and volume functions
data/         downloads and caches (git-ignored)
results/      predictions (parquet) and figures (png)
RESULTS.md    numbers, figures, limitations
README.md     how to reproduce in three commands
DECISIONS.md  every choice made without the user
STATUS.md     stage-by-stage progress
```
