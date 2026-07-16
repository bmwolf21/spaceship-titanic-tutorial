# Codex pipeline

Independent baseline for the ROGII Wellbore Geology Prediction competition.

## Approach

This pipeline predicts post-PS `TVT` in two stages:

1. Anchor each lateral at the last known heel `TVT_input` and extrapolate by
   vertical displacement:

   `base_tvt = last_TVT_input + (last_Z - current_Z)`

2. Learn a global ridge-regression correction to the base prediction on the
   shared whole-well folds. Features are restricted to inference-available
   columns (`MD, X, Y, Z, GR, TVT_input`) plus the paired type-well
   (`TVT, GR`). Train-only formation marker columns are not used.

The model is intentionally dependency-light: `numpy`, `pandas`, and `scipy`
are enough.

## Run

From the repository root:

```bash
python3 codex/train_predict.py
```

Outputs:

- `codex/oof.csv`: shared-fold out-of-fold predictions
- `codex/test_pred.csv`: test row predictions
- `codex/metrics.json`: CV metrics and configuration
- `outputs/submissions/codex_ridge_residual_YYYYMMDD.csv`: Kaggle submission

