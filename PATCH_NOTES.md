# Patch notes: v4 clean ablation

This patch is designed for controlled experiments, not for claiming a guaranteed score improvement.

## Main code changes

1. `train.py`
   - Added `--neutral-weight-multiplier`.
   - Added `--input-format {term,term_desc}`.
   - Prints effective class weights at the start of each run.

2. `absa_pipeline.py`
   - Added optional `aspects.json` loading.
   - `term_desc` input format appends description, but the aspect mask still covers only the raw aspect term.
   - `describe_samples()` now also reports unique aspect counts and unseen-aspect ratio for non-train splits.

3. `tune.py`
   - Added `ablation_core`, `ablation_desc`, and `ablation_all` search spaces.
   - `ablation_core` isolates pooling and neutral weighting from description usage.

4. Kaggle
   - Added `kaggle_ablation_training.ipynb`.
   - Added `run_ablation_core.sh` and `run_ablation_desc.sh`.

## Recommended order

1. Run `ablation_core` first.
2. Inspect neutral precision/recall/F1 and seen/unseen metrics.
3. Run `ablation_desc` only as a separate secondary ablation.
