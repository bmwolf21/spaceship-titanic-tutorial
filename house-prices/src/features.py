"""
features.py - shared feature engineering for House Prices.

Driven by the EDA (TUTORIAL.md Step 1):
- Structural NA handling: NA that means "none" is filled explicitly, not imputed.
- Ordinal quality ratings encoded in their true order.
- Neighbor-based LotFrontage imputation (structural, like group-mode fills).
- A few engineered size/age features.
Target is modeled as log1p(SalePrice).
"""
import numpy as np
import pandas as pd

# Categorical columns where NA genuinely means "feature absent" -> "None".
NONE_CAT = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
            "GarageType", "GarageFinish", "GarageQual", "GarageCond",
            "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
            "BsmtFinType2", "MasVnrType"]
# Numeric columns where NA means "none of it" -> 0.
ZERO_NUM = ["GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1",
            "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath",
            "BsmtHalfBath", "MasVnrArea"]
# Ordered quality scales -> integers (higher is better).
QUAL_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
QUAL_COLS = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
             "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]


def build_features(train: pd.DataFrame, test: pd.DataFrame):
    y = np.log1p(train["SalePrice"])
    both = pd.concat([train.drop(columns=["SalePrice"]), test],
                     ignore_index=True)
    n_train = len(train)
    both = both.drop(columns=["Id"])

    # Structural NA -> explicit "none"
    for c in NONE_CAT:
        both[c] = both[c].fillna("None")
    for c in ZERO_NUM:
        both[c] = both[c].fillna(0)

    # Neighbor-based LotFrontage fill (median within Neighborhood).
    both["LotFrontage"] = both.groupby("Neighborhood")["LotFrontage"].transform(
        lambda s: s.fillna(s.median()))

    # Ordinal quality encodings.
    for c in QUAL_COLS:
        both[c] = both[c].map(QUAL_MAP).fillna(0).astype(int)

    # Remaining categoricals: fill NA with mode; remaining numerics: median.
    cat_cols = both.select_dtypes(exclude="number").columns
    for c in cat_cols:
        both[c] = both[c].fillna(both[c].mode()[0])
    num_cols = both.select_dtypes(include="number").columns
    for c in num_cols:
        both[c] = both[c].fillna(both[c].median())

    # Engineered features
    both["TotalSF"] = both["TotalBsmtSF"] + both["1stFlrSF"] + both["2ndFlrSF"]
    both["TotalBath"] = (both["FullBath"] + 0.5 * both["HalfBath"]
                         + both["BsmtFullBath"] + 0.5 * both["BsmtHalfBath"])
    both["TotalPorch"] = (both["OpenPorchSF"] + both["EnclosedPorch"]
                          + both["3SsnPorch"] + both["ScreenPorch"]
                          + both["WoodDeckSF"])
    both["HouseAge"] = both["YrSold"] - both["YearBuilt"]
    both["RemodAge"] = both["YrSold"] - both["YearRemodAdd"]
    both["IsRemodeled"] = (both["YearBuilt"] != both["YearRemodAdd"]).astype(int)
    both["HasPool"] = (both["PoolArea"] > 0).astype(int)
    both["HasGarage"] = (both["GarageArea"] > 0).astype(int)
    both["Has2ndFloor"] = (both["2ndFlrSF"] > 0).astype(int)

    # Log-transform skewed numeric features (skew > 0.75), excluding flags.
    skew_cols = [c for c in both.select_dtypes(include="number").columns
                 if both[c].min() >= 0 and both[c].nunique() > 10]
    for c in skew_cols:
        if both[c].skew() > 0.75:
            both[c] = np.log1p(both[c])

    # Encode remaining nominal categoricals as integer codes (tree-friendly).
    for c in both.select_dtypes(exclude="number").columns:
        both[c] = both[c].astype("category").cat.codes

    X = both.iloc[:n_train].reset_index(drop=True)
    X_test = both.iloc[n_train:].reset_index(drop=True)
    return X, y, X_test, list(X.columns)
