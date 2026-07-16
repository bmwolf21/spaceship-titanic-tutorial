"""
04_blend_check.py  (Claude's pipeline)

The payoff test of the collaboration: does blending Claude's and Codex's
independent OOF predictions (on the shared folds) beat either alone? Weight is
tuned honestly out-of-fold (pick w on 4 folds, apply to the held-out fold).
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "shared"))
from metric import rmse  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
c = pd.read_csv(os.path.join(HERE, "oof.csv")).rename(columns={"tvt_pred": "claude"})
x = pd.read_csv(os.path.join(ROOT, "codex", "oof.csv")).rename(columns={"tvt_pred": "codex"})
df = c.merge(x, on=["well_id", "row_index"], how="inner")
print(f"merged OOF rows: {len(df):,}")

# truth from train TVT
truth = []
for wid, grp in df.groupby("well_id", sort=False):
    tvt = pd.read_csv(os.path.join(RAW, "train", f"{wid}__horizontal_well.csv"),
                      usecols=["TVT"])["TVT"].values
    truth.append(pd.Series(tvt[grp["row_index"].values], index=grp.index))
df["tvt_true"] = pd.concat(truth)
folds = pd.read_csv(os.path.join(ROOT, "shared", "folds.csv"))[["well_id", "fold"]]
df = df.merge(folds, on="well_id", how="left")

print(f"\n  Claude alone: {rmse(df['tvt_true'], df['claude']):.4f} ft")
print(f"  Codex  alone: {rmse(df['tvt_true'], df['codex']):.4f} ft")
print(f"  Equal blend : {rmse(df['tvt_true'], 0.5*df['claude']+0.5*df['codex']):.4f} ft")

# honest out-of-fold weight selection
ws = np.linspace(0, 1, 41)
oof_blend = np.zeros(len(df))
for fld in range(5):
    tr = df["fold"] != fld
    va = df["fold"] == fld
    errs = [rmse(df.loc[tr, "tvt_true"], w*df.loc[tr, "claude"]+(1-w)*df.loc[tr, "codex"]) for w in ws]
    w_best = ws[int(np.argmin(errs))]
    oof_blend[va.values] = w_best*df.loc[va, "claude"] + (1-w_best)*df.loc[va, "codex"]
print(f"  Blend (OOF-honest weight): {rmse(df['tvt_true'], oof_blend):.4f} ft")

# global best weight for reference
errs = [rmse(df["tvt_true"], w*df["claude"]+(1-w)*df["codex"]) for w in ws]
print(f"  Blend (global best w={ws[int(np.argmin(errs))]:.2f}): {min(errs):.4f} ft")
