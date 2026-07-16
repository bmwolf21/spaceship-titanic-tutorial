"""
12_dtvt_magnitude_predictability.py  (Claude, supporting Codex's plan)

Codex plans "separate handling for high-dTVT wells". But dTVT is the target, so at
test time we must PREDICT which wells will be high-dTVT from inference-available
features. This script tests whether well-level dTVT magnitude is predictable at
all (group-CV). If yes -> Codex's adaptive-by-magnitude lever is feasible; if no
-> it can't be applied at test time and Codex should skip it.

Well-level target: mean(|TVT - TVT_PS|) over the toe.
Well-level inference features: recent-heel steering roughness, heel GR spread,
trajectory geometry, location, and a fold-aware neighbor prior on magnitude.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "shared"))
RAW = os.path.join(ROOT, "data", "raw")
folds = pd.read_csv(os.path.join(ROOT, "shared", "folds.csv"))
fold_of = dict(zip(folds["well_id"], folds["fold"]))


def well_row(wid):
    hz = pd.read_csv(os.path.join(RAW, "train", f"{wid}__horizontal_well.csv"))
    ps = np.where(hz["TVT_input"].notna().values)[0].max()
    tvt_ps = float(hz["TVT_input"].iloc[ps])
    ti = hz["TVT_input"].values[:ps + 1]
    gr = pd.Series(hz["GR"].values).interpolate(limit_direction="both").fillna(0).values
    z, md = hz["Z"].values, hz["MD"].values
    tail = slice(max(0, ps - 300), ps + 1)                 # recent heel
    incl = np.gradient(z) / (np.gradient(md) + 1e-6)
    toe = np.arange(ps + 1, len(hz))
    y = float(np.mean(np.abs(hz["TVT"].values[toe] - tvt_ps)))   # target: dTVT magnitude
    return {
        "well_id": wid, "fold": fold_of[wid], "target": y,
        "x": float(np.nanmean(hz["X"])), "y_": float(np.nanmean(hz["Y"])),
        "tvt_ps": tvt_ps, "toe_len": len(toe),
        "heel_step_std": float(np.nanstd(np.diff(ti[-300:]))),       # recent steering roughness
        "heel_tvt_range": float(np.nanmax(ti[-300:]) - np.nanmin(ti[-300:])),
        "heel_gr_std": float(np.nanstd(gr[tail])),
        "heel_gr_range": float(np.nanpercentile(gr[tail], 95) - np.nanpercentile(gr[tail], 5)),
        "incl_std": float(np.nanstd(incl[tail])),
        "incl_range": float(np.nanmax(incl[tail]) - np.nanmin(incl[tail])),
    }


print("Building well-level table...")
wids = [os.path.basename(f).replace("__horizontal_well.csv", "")
        for f in sorted(glob.glob(os.path.join(RAW, "train", "*__horizontal_well.csv")))]
W = pd.DataFrame([well_row(w) for w in wids])

# fold-aware neighbor prior on dTVT magnitude (neighbors from other folds only)
W["nbr_mag"] = np.nan
loc = W[["x", "y_"]].values
for k in range(5):
    tgt = W["fold"] == k
    pool = W[~tgt]
    tree = cKDTree(pool[["x", "y_"]].values)
    dist, idx = tree.query(loc[tgt.values], k=min(15, len(pool)))
    W.loc[tgt, "nbr_mag"] = pool["target"].values[idx].mean(axis=1)

FEATS = ["tvt_ps", "toe_len", "heel_step_std", "heel_tvt_range", "heel_gr_std",
         "heel_gr_range", "incl_std", "incl_range", "nbr_mag", "x", "y_"]
params = dict(objective="regression", n_estimators=300, learning_rate=0.03, num_leaves=15,
              min_child_samples=20, reg_lambda=3.0, subsample=0.8, subsample_freq=1,
              random_state=42, verbose=-1)
oof = np.zeros(len(W))
for k in range(5):
    tr, va = W["fold"] != k, W["fold"] == k
    m = lgb.LGBMRegressor(**params).fit(W.loc[tr, FEATS], W.loc[tr, "target"])
    oof[va.values] = m.predict(W.loc[va, FEATS])

r = np.corrcoef(oof, W["target"])[0, 1]
rho = spearmanr(oof, W["target"]).correlation
print(f"\nPredicting well-level dTVT magnitude (mean|dTVT|), out-of-fold:")
print(f"  Pearson r = {r:.3f} | Spearman rho = {rho:.3f}")
print(f"  corr(nbr_mag alone, target) = {np.corrcoef(W['nbr_mag'], W['target'])[0,1]:.3f}")
# can we flag the hard (top-quintile) high-dTVT wells in advance?
hi = W["target"] >= W["target"].quantile(0.8)
pred_hi = oof >= np.quantile(oof, 0.8)
prec = (hi & pred_hi).sum() / max(pred_hi.sum(), 1)
print(f"  top-quintile high-dTVT wells: precision@top-quintile-pred = {prec:.2f} "
      f"(0.20 = random)")
imp = pd.Series(lgb.LGBMRegressor(**params).fit(W[FEATS], W['target']).feature_importances_, index=FEATS)
print("  importance:", imp.sort_values(ascending=False).head(5).round(0).to_dict())
