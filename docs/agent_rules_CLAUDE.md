# Rules for this folder: unattended run

The user is away for the whole run and will read the results afterwards. Nobody can answer questions.

## Never block on a question

- Do not ask the user anything. When a choice comes up, take the fallback written in `BRIEF.md`. If the brief is silent, take the cheapest reasonable option.
- Log every such choice in `DECISIONS.md`: what came up, what you chose, why, and how to redo it differently.

## Stay inside this folder

- Read, write and delete only inside this project folder.
- Install Python packages only into the virtual environment in this folder. No sudo, no global installs, no system changes.
- Network use is limited to: PyPI, Zenodo, and the download host for pretrained torchvision weights. Nothing else.
- Use git locally only: commit after each completed stage. Never push, never add a remote.

## Budget and stopping

- Wall-clock limit: 5 hours from the start. Check elapsed time before each stage.
- Download limit: 8 GB in total, as set in `BRIEF.md`.
- If a stage fails twice, record the error in `STATUS.md`, take the next fallback, and continue with the stages that do not depend on it.
- If time runs short, finish with fewer models rather than leaving no write-up. A `RESULTS.md` covering what completed is required in every case.

## Honesty about results

- Report what the first complete run shows, including a null or disappointing result. Do not tune until a hoped-for effect appears.
- Never write a number in `RESULTS.md` that is not produced by code in this folder and saved under `results/`.
- A stage that did not run is reported as not run. No placeholder figures, no estimated numbers.
- Label every use of the placeholder volumes as placeholder.

## Working order

1. Create the environment and the folder layout. Write `STATUS.md` with the stage list.
2. Inspect the Zenodo record, decide the download level, log it, download.
3. Write and test the data loading, metric and volume functions before any training.
4. Timed pilot for each model, then size the runs to the compute budget.
5. Train and evaluate models 1, 2, 3. Save predictions after each.
6. Volume step, figures, `RESULTS.md`, `README.md`.
7. Final line in `STATUS.md`: DONE or PARTIAL, with what is missing.

Update `STATUS.md` when each stage starts and ends, with a timestamp.
