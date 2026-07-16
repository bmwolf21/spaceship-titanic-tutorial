"""
02_explore_signals.py  (Claude's pipeline)

The geosteered wells stay near their PS TVT, so "flat" is a strong baseline.
What actually moves TVT past PS? Test simple physical hypotheses and measure which
signal (change in Z, MD, GR) predicts the change in TVT. This guides the model.

Reports RMSE for several closed-form baselines and the correlation of true dTVT
with candidate drivers, overall and per shared fold.
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

TRAIN = os.path.join(ROOT, "data", "raw", "train")
folds = pd.read_csv(os.path.join(ROOT, "shared", "folds.csv"))
fold_of = dict(zip(folds["well_id"], folds["fold"]))

acc = {name: {"t": [], "p": []} for name in ["flat", "dZ", "dZ_scaled"]}
dtvt_all, dz_all, dmd_all = [], [], []

for f in sorted(glob.glob(os.path.join(TRAIN, "*__horizontal_well.csv"))):
    df = pd.read_csv(f, usecols=["MD", "Z", "GR", "TVT", "TVT_input"])
    ps = np.where(df["TVT_input"].notna().values)[0].max()
    tvt_ps, z_ps, md_ps = df["TVT_input"].iloc[ps], df["Z"].iloc[ps], df["MD"].iloc[ps]
    toe = df.iloc[ps + 1:]
    t = toe["TVT"].values
    dz = toe["Z"].values - z_ps
    dmd = toe["MD"].values - md_ps
    acc["flat"]["t"].extend(t);       acc["flat"]["p"].extend([tvt_ps] * len(t))
    acc["dZ"]["t"].extend(t);         acc["dZ"]["p"].extend(tvt_ps + dz)      # dTVT = dZ
    acc["dZ_scaled"]["t"].extend(t);  acc["dZ_scaled"]["p"].extend(tvt_ps + 0.5 * dz)
    dtvt_all.extend(t - tvt_ps); dz_all.extend(dz); dmd_all.extend(dmd)

print("Closed-form baselines (overall RMSE, ft):")
for name in acc:
    print(f"  {name:10s} {rmse(acc[name]['t'], acc[name]['p']):.3f}")

dtvt = np.array(dtvt_all); dz = np.array(dz_all); dmd = np.array(dmd_all)
print("\nWhat drives dTVT (toe TVT minus PS TVT)?")
print(f"  corr(dTVT, dZ)  = {np.corrcoef(dtvt, dz)[0,1]:.3f}")
print(f"  corr(dTVT, dMD) = {np.corrcoef(dtvt, dmd)[0,1]:.3f}")
# best linear fit dTVT ~ dZ
b = np.polyfit(dz, dtvt, 1)
print(f"  best fit dTVT = {b[0]:.3f}*dZ + {b[1]:.3f}")
pred = b[0] * dz + b[1] + np.array([tv - dt for tv, dt in zip(acc['flat']['p'], dtvt)])  # placeholder
print(f"\n  dTVT magnitude: mean|dTVT| = {np.mean(np.abs(dtvt)):.2f} ft, "
      f"p95 = {np.percentile(np.abs(dtvt),95):.1f} ft")
print(f"  dZ magnitude:   mean|dZ|   = {np.mean(np.abs(dz)):.2f} ft")
