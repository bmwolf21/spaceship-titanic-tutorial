# House Prices (Advanced Regression) - Tutorial Log

> Competition: <https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques>
> Goal: predict `SalePrice` for each home (regression).
> Metric: **RMSE of log(SalePrice)** (so we model on the log scale).
>
> Running record of every step, detailed enough to become a standalone tutorial.
> Newest entries at the bottom. Competition #2 in the workspace, run with the
> same three-angle playbook as Spaceship Titanic.

## The three angles

1. **Compete** - build models and submit to the leaderboard.
2. **Document** - this file.
3. **Translate** - `wildlife_translation/` reframes the pipeline as predicting a
   continuous ecological quantity (animal abundance / density) from habitat
   covariates.

## Why this competition maps well to ecology

Spaceship Titanic was detection/occupancy (binary). House Prices is the
**abundance/density** analogue:

- **Continuous, right-skewed target** (`SalePrice`) modeled on the log scale,
  exactly like animal density or count data (often log or sqrt transformed).
- **Many mixed covariates** (numeric + ordinal quality ratings + nominal
  categories) is what a habitat covariate table looks like.
- **Structural missingness**: many NAs here mean "feature absent" (no garage, no
  basement, no pool), not "unrecorded". The ecological parallel is a covariate
  that is genuinely absent at a site (no canopy, no water) versus one that was
  not measured. Getting this distinction right is the crux of the imputation.

## Step log

### Step 0 - Setup

- Downloaded data to `data/raw/` (train 1460x81, test 1459x80, plus
  `data_description.txt`, the data dictionary).
- Created the standard project structure and this log.

### Step 1 - Exploratory Data Analysis (`src/01_eda.py`)

**Target is right-skewed.** `SalePrice` skewness is 1.88 raw, 0.12 after
`log1p`. We model on the log scale, which also matches the competition metric
(RMSE of log). Figures saved to `outputs/figures/`.

**Mixed covariates:** 36 numeric, 43 categorical (including many ordinal quality
ratings like Ex/Gd/TA/Fa/Po).

**Missingness is mostly STRUCTURAL (the key insight).** 34 columns have NAs, but
for most, NA means "the feature does not exist," not "unrecorded":
- PoolQC 99.7% NA (almost no house has a pool), MiscFeature, Alley, Fence,
  FireplaceQu, and all Garage*/Bsmt* columns -> NA = no pool / alley / fence /
  fireplace / garage / basement.
- The right fix is to fill these with an explicit "None" category (and the paired
  numeric columns, like GarageArea, with 0), NOT to impute a plausible value.
- Only a few are genuinely missing and need real imputation, chiefly
  `LotFrontage` (16.6%), which we will fill from the neighborhood (a structural,
  neighbor-based fill, the same idea as group-mode imputation in Spaceship
  Titanic).

**Strongest predictors** (correlation with SalePrice): OverallQual 0.79,
GrLivArea 0.71, GarageCars 0.64, GarageArea 0.62, TotalBsmtSF 0.61, 1stFlrSF
0.61, YearBuilt 0.52. Size and overall quality dominate.

**Ecological read:** the structural-vs-genuine missingness distinction is exactly
the covariate-absent-vs-unmeasured problem in habitat data, and neighbor-based
`LotFrontage` imputation is the same move as filling a site covariate from nearby
sites.

### Step 2 - Feature engineering (`src/features.py`)

From 79 raw features to 88, modeling `log1p(SalePrice)`:

- **Structural NA handling:** 15 categorical columns where NA means "absent"
  filled with an explicit `"None"`; paired numeric columns (GarageArea, MasVnrArea,
  Bsmt* areas) filled with 0. This is the central move and it prevents the model
  from treating "no garage" as "unknown garage".
- **Ordinal quality encoding:** Ex/Gd/TA/Fa/Po/None mapped to 5..0 for the quality
  columns, preserving order that one-hot encoding would discard.
- **Neighbor-based imputation:** `LotFrontage` filled with the median of its
  Neighborhood (the House Prices analogue of Spaceship Titanic's group-mode fills).
- **Engineered features:** TotalSF, TotalBath, TotalPorch, HouseAge, RemodAge,
  IsRemodeled, HasPool/HasGarage/Has2ndFloor.
- **Skew correction:** `log1p` applied to skewed numeric features (skew > 0.75).

### Step 3 - Baseline model + submission (`src/02_baseline_model.py`)

LightGBM regressor, 5-fold CV with early stopping, RMSE on the log scale.

| Fold | RMSE(log) |
|------|-----------|
| 1 | 0.12962 |
| 2 | 0.10873 |
| 3 | 0.15620 |
| 4 | 0.12332 |
| 5 | 0.10725 |
| **CV mean** | **0.12503 +/- 0.01776** |

**Submitted -> public LB 0.12502**, essentially identical to CV. Honest
evaluation confirmed. Fold 3's higher error (0.156) points to the known
high-leverage outliers in this dataset (a few very large homes sold cheaply);
handling those is the natural next iteration.

**Where we sit:** a single GBM at ~0.125 is competitive for this competition
(honest scores cluster near 0.11 to 0.13). Next: outlier handling, a linear model
(Ridge/Lasso, which do well here) blended with the GBM, then the wildlife
translation as an abundance-regression cookbook entry.
