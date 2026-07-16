"""
08_gr_offset_model.py  (Claude's pipeline)

Add an OFFSET-WELL PRIOR to my GR model (03). Neighboring wells share geological
dip, so the dTVT *shape* along the toe of nearby wells is a strong prior for the
target well - a signal my pure-GR model lacks, and the fold-3 hard wells (large
excursions) are exactly where a neighbor prior should help.

Leakage-safe: the prior for a validation-fold well is built ONLY from training-fold
neighbors. Feature = predicted dTVT at each toe point from the mean neighbor
dTVT-vs-toe_fraction profile. Combined with 03's GR-inversion features in LightGBM.

Writes claude/oof.csv, claude/test_pred.csv, submission.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "shared"))
from metric import rmse  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
folds = pd.read_csv(os.path.join(ROOT, "shared", "folds.csv"))
fold_of = dict(zip(folds["well_id"], folds["fold"]))
SHIFTS = np.arange(-40.0, 40.001, 0.5)
GRID = np.linspace(0.0, 1.0, 30)          # toe-fraction grid for neighbor profiles
K = 15                                     # neighbors


def roll(a, w, fn):
    return getattr(pd.Series(a).rolling(w, center=True, min_periods=1), fn)().values


def implied(gr_pt, ref_tvt, ref_gr, tvt_ps):
    if len(ref_tvt) < 3 or not np.isfinite(ref_gr).any():
        return np.zeros_like(gr_pt)
    o = np.argsort(ref_tvt)
    grid = np.interp(tvt_ps + SHIFTS, np.asarray(ref_tvt)[o], np.asarray(ref_gr)[o])
    return SHIFTS[np.argmin(np.abs(grid[None, :] - gr_pt[:, None]), axis=1)]


def well_features(hz, tw, is_train):
    ps = np.where(hz["TVT_input"].notna().values)[0].max()
    tvt_ps = float(hz["TVT_input"].iloc[ps])
    md_ps, z_ps = float(hz["MD"].iloc[ps]), float(hz["Z"].iloc[ps])
    gr = pd.Series(hz["GR"].values).interpolate(limit_direction="both").values
    gr_s = roll(gr, 5, "mean")
    grad = np.gradient(gr_s)
    self_tvt, self_gr = hz["TVT_input"].values[:ps + 1], gr_s[:ps + 1]
    toe = np.arange(ps + 1, len(hz))
    d_tw = implied(gr_s[toe], tw["TVT"].values, roll(tw["GR"].values, 5, "mean"), tvt_ps)
    d_self = implied(gr_s[toe], self_tvt, self_gr, tvt_ps)
    frac = (toe - ps) / (len(hz) - ps)
    feat = pd.DataFrame({
        "well_id": hz["well_id"].iloc[0], "row_index": toe, "tvt_ps": tvt_ps,
        "toe_frac": frac,
        "implied_tw": d_tw, "implied_self": d_self,
        "implied_tw_med": roll(d_tw, 21, "median"), "implied_self_med": roll(d_self, 21, "median"),
        "implied_mean": roll(0.5 * (d_tw + d_self), 41, "median"),
        "agree": np.abs(d_tw - d_self),
        "gr_s": gr_s[toe], "grad": grad[toe], "gr_vs_heel": gr_s[toe] - np.nanmean(self_gr),
        "dMD": hz["MD"].values[toe] - md_ps, "dZ": hz["Z"].values[toe] - z_ps,
        "incl": (hz["Z"].values[toe] - z_ps) / (hz["MD"].values[toe] - md_ps + 1e-6),
    })
    loc = (float(np.nanmean(hz["X"])), float(np.nanmean(hz["Y"])))
    if is_train:
        feat["dtvt"] = hz["TVT"].values[toe] - tvt_ps
        prof = np.interp(GRID, frac, feat["dtvt"].values)     # true dTVT profile
    else:
        prof = None
    return feat, loc, prof


def load(split, wid):
    hz = pd.read_csv(os.path.join(RAW, split, f"{wid}__horizontal_well.csv")); hz["well_id"] = wid
    tw = pd.read_csv(os.path.join(RAW, split, f"{wid}__typewell.csv"))
    return hz, tw


print("Building train features + profiles...")
wids = [os.path.basename(f).replace("__horizontal_well.csv", "")
        for f in sorted(glob.glob(os.path.join(RAW, "train", "*__horizontal_well.csv")))]
feats, loc, prof = {}, {}, {}
for w in wids:
    feats[w], loc[w], prof[w] = well_features(*load("train", w), True)
locs = np.array([loc[w] for w in wids])
train = pd.concat(feats.values(), ignore_index=True)
train["fold"] = train["well_id"].map(fold_of)


def neighbor_prior(target_wids, pool_wids):
    """mean neighbor dTVT profile for each target well, from pool wells only."""
    pool = np.array(pool_wids)
    tree = cKDTree(np.array([loc[w] for w in pool]))
    out = {}
    for w in target_wids:
        d, idx = tree.query(loc[w], k=min(K + 1, len(pool)))
        idx = np.atleast_1d(idx)
        nb = [pool[i] for i in idx if pool[i] != w][:K]
        out[w] = np.mean([prof[n] for n in nb], axis=0)
    return out


# fold-aware offset prior for OOF
train["offset_prior"] = 0.0
for k in range(5):
    tgt = [w for w in wids if fold_of[w] == k]
    poo = [w for w in wids if fold_of[w] != k]
    pr = neighbor_prior(tgt, poo)
    for w in tgt:
        m = train["well_id"] == w
        train.loc[m, "offset_prior"] = np.interp(train.loc[m, "toe_frac"], GRID, pr[w])
print(f"train rows {len(train):,}")

FEATS = [c for c in train.columns if c not in {"well_id", "row_index", "dtvt", "fold"}]
params = dict(objective="regression", n_estimators=1500, learning_rate=0.03, num_leaves=63,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=200,
              reg_lambda=2.0, random_state=42, verbose=-1, n_jobs=-1)
oof = np.zeros(len(train)); models = []
for k in range(5):
    tr, va = train["fold"] != k, train["fold"] == k
    m = lgb.LGBMRegressor(**params).fit(train.loc[tr, FEATS], train.loc[tr, "dtvt"]); models.append(m)
    oof[va.values] = m.predict(train.loc[va, FEATS])
    tp = train.loc[va, "tvt_ps"].values
    print(f"  fold {k}: RMSE {rmse(train.loc[va,'dtvt']+tp, oof[va.values]+tp):.3f} ft")
tpall = train["tvt_ps"].values
print(f"\nOOF RMSE: {rmse(train['dtvt']+tpall, oof+tpall):.3f} ft  (03 was 15.249)")
imp = pd.Series(models[0].feature_importances_, index=FEATS).sort_values(ascending=False)
print("top features:", list(imp.head(6).index))
pd.DataFrame({"well_id": train["well_id"], "row_index": train["row_index"],
              "tvt_pred": oof + tpall}).to_csv(os.path.join(HERE, "oof.csv"), index=False)

# test: neighbors = all train wells
print("Test wells...")
twids = [os.path.basename(f).replace("__horizontal_well.csv", "")
         for f in sorted(glob.glob(os.path.join(RAW, "test", "*__horizontal_well.csv")))]
tfeats = {}
for w in twids:
    tfeats[w], loc[w], _ = well_features(*load("test", w), False)
pr = neighbor_prior(twids, wids)
test = pd.concat(tfeats.values(), ignore_index=True)
test["offset_prior"] = 0.0
for w in twids:
    m = test["well_id"] == w
    test.loc[m, "offset_prior"] = np.interp(test.loc[m, "toe_frac"], GRID, pr[w])
tp = np.mean([mm.predict(test[FEATS]) for mm in models], axis=0) + test["tvt_ps"].values
pd.DataFrame({"well_id": test["well_id"], "row_index": test["row_index"],
              "tvt_pred": tp}).to_csv(os.path.join(HERE, "test_pred.csv"), index=False)
pd.DataFrame({"id": test["well_id"] + "_" + test["row_index"].astype(str), "tvt": tp}).to_csv(
    os.path.join(ROOT, "outputs", "submissions", "claude_gr_offset_20260716.csv"), index=False)
print("wrote oof.csv, test_pred.csv, submission")
