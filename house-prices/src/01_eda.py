"""
01_eda.py - Exploratory Data Analysis for House Prices.

Angle 1 (compete): understand the target, missingness, and covariate structure.
Angle 2 (document): findings printed here feed TUTORIAL.md.

Run:  python src/01_eda.py
Saves figures to outputs/figures/, prints a findings summary.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "data", "raw")
FIG = os.path.join(HERE, "outputs", "figures")
os.makedirs(FIG, exist_ok=True)

train = pd.read_csv(os.path.join(RAW, "train.csv"))
test = pd.read_csv(os.path.join(RAW, "test.csv"))

print("=" * 70)
print(f"SHAPES  train {train.shape} | test {test.shape}")
print("=" * 70)

# --- Target -----------------------------------------------------------------
y = train["SalePrice"]
print("\nTARGET (SalePrice):")
print(f"  min {y.min():,} | median {y.median():,.0f} | max {y.max():,}")
print(f"  skewness raw: {y.skew():.2f} | skewness log1p: {np.log1p(y).skew():.2f}")
print("  -> right-skewed; log transform makes it near-normal (and matches the")
print("     RMSE-of-log metric).")

# --- Feature types ----------------------------------------------------------
feat = train.drop(columns=["Id", "SalePrice"])
num = feat.select_dtypes(include="number").columns
cat = feat.select_dtypes(exclude="number").columns
print(f"\nFEATURE TYPES: {len(num)} numeric, {len(cat)} categorical")

# --- Missingness ------------------------------------------------------------
both = pd.concat([train.drop(columns=["SalePrice"]), test], ignore_index=True)
miss = (both.isna().mean() * 100).round(2)
miss = miss[miss > 0].sort_values(ascending=False)
print(f"\nMISSINGNESS: {len(miss)} columns have missing values. Top 12:")
print(miss.head(12).to_string())
print("\nNOTE: for many of these, NA is STRUCTURAL (means 'none'):")
print("  PoolQC, MiscFeature, Alley, Fence, FireplaceQu, Garage*, Bsmt* ->")
print("  NA = the house has no pool/alley/fence/fireplace/garage/basement.")

# --- Correlations with target ----------------------------------------------
corr = train[list(num) + ["SalePrice"]].corr()["SalePrice"].drop("SalePrice")
top = corr.abs().sort_values(ascending=False).head(12)
print("\nTOP NUMERIC CORRELATIONS WITH SalePrice:")
print(corr[top.index].round(3).to_string())

# --- Figures ----------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(y, bins=40, color="#4C72B0"); ax[0].set_title("SalePrice (raw)")
ax[1].hist(np.log1p(y), bins=40, color="#55A868"); ax[1].set_title("log1p(SalePrice)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "01_target_distribution.png"), dpi=120); plt.close()

miss.head(20).iloc[::-1].plot.barh(figsize=(7, 6), color="#C44E52")
plt.title("Missingness by column (top 20)"); plt.xlabel("% missing")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "01_missingness.png"), dpi=120); plt.close()

# scatter of the single strongest predictor
best = corr.abs().idxmax()
plt.figure(figsize=(6, 4))
plt.scatter(train[best], y, s=10, alpha=0.5, color="#4C72B0")
plt.xlabel(best); plt.ylabel("SalePrice"); plt.title(f"SalePrice vs {best} (r={corr[best]:.2f})")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "01_top_predictor.png"), dpi=120); plt.close()

print("\nFIGURES SAVED: 01_target_distribution.png, 01_missingness.png, 01_top_predictor.png")
