"""
05_gr_context_model.py  (Claude's pipeline)

Improve on 03: the per-point nearest-GR match was noisy (fold variance 13-18).
Two changes:
  1. Do the GR->TVT inversion on a MORE SMOOTHED GR (less ambiguous) and give the
     tree a GR-CONTEXT window (smoothed GR at several offsets) so it can pattern-
     match a local shape rather than a single value.
  2. Add a type-log vs self-log AGREEMENT signal (where the two references imply
     the same dTVT, trust it more) and multi-scale smoothing of the implied dTVT.

Same shared folds + LightGBM predicting dTVT. Writes claude/oof.csv,
claude/test_pred.csv, and a submission.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "shared"))
from metric import rmse  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
folds = pd.read_csv(os.path.join(ROOT, "shared", "folds.csv"))
fold_of = dict(zip(folds["well_id"], folds["fold"]))
SHIFTS = np.arange(-45.0, 45.001, 0.5)
CTX = [-15, -8, -4, 4, 8, 15]      # GR-context offsets (rows)


def roll(a, w, fn):
    return getattr(pd.Series(a).rolling(w, center=True, min_periods=1), fn)().values


def implied(gr_pt, ref_tvt, ref_gr, tvt_ps):
    if len(ref_tvt) < 3 or not np.isfinite(ref_gr).any():
        return np.zeros_like(gr_pt)
    o = np.argsort(ref_tvt)
    grid = np.interp(tvt_ps + SHIFTS, np.asarray(ref_tvt)[o], np.asarray(ref_gr)[o])
    return SHIFTS[np.argmin(np.abs(grid[None, :] - gr_pt[:, None]), axis=1)]


def features_for_well(hz, tw, is_train):
    ps = np.where(hz["TVT_input"].notna().values)[0].max()
    tvt_ps = float(hz["TVT_input"].iloc[ps])
    md_ps, z_ps = float(hz["MD"].iloc[ps]), float(hz["Z"].iloc[ps])
    gr = pd.Series(hz["GR"].values).interpolate(limit_direction="both").values
    gr15 = roll(gr, 15, "mean")
    gr31 = roll(gr, 31, "mean")
    grad = np.gradient(gr15)

    self_tvt = hz["TVT_input"].values[:ps + 1]
    self_gr = gr15[:ps + 1]
    toe = np.arange(ps + 1, len(hz))

    d_tw = implied(gr15[toe], tw["TVT"].values, roll(tw["GR"].values, 5, "mean"), tvt_ps)
    d_self = implied(gr15[toe], self_tvt, self_gr, tvt_ps)

    feat = {
        "well_id": hz["well_id"].iloc[0], "row_index": toe, "tvt_ps": tvt_ps,
        "implied_tw": d_tw, "implied_self": d_self,
        "implied_tw_med": roll(d_tw, 25, "median"),
        "implied_self_med": roll(d_self, 25, "median"),
        "implied_med_long": roll(0.5 * (d_tw + d_self), 61, "median"),
        "agree": np.abs(d_tw - d_self),                        # low = references agree
        "implied_std": roll(d_tw, 25, "std"),
        "gr15": gr15[toe], "gr31": gr31[toe], "grad": grad[toe],
        "gr_vs_heel": gr15[toe] - np.nanmean(self_gr),
        "dMD": hz["MD"].values[toe] - md_ps,
        "dZ": hz["Z"].values[toe] - z_ps,
        "incl": (hz["Z"].values[toe] - z_ps) / (hz["MD"].values[toe] - md_ps + 1e-6),
        "toe_frac": (toe - ps) / (len(hz) - ps),
    }
    for off in CTX:                       # GR-context window (shape, not a point)
        idx = np.clip(toe + off, 0, len(hz) - 1)
        feat[f"gr_ctx_{off}"] = gr15[idx]
    if is_train:
        feat["dtvt"] = hz["TVT"].values[toe] - tvt_ps
    return pd.DataFrame(feat)


def load(split, wid):
    hz = pd.read_csv(os.path.join(RAW, split, f"{wid}__horizontal_well.csv"))
    hz["well_id"] = wid
    tw = pd.read_csv(os.path.join(RAW, split, f"{wid}__typewell.csv"))
    return hz, tw


print("Building train features...")
wids = [os.path.basename(f).replace("__horizontal_well.csv", "")
        for f in sorted(glob.glob(os.path.join(RAW, "train", "*__horizontal_well.csv")))]
train = pd.concat([features_for_well(*load("train", w), True) for w in wids],
                  ignore_index=True)
train["fold"] = train["well_id"].map(fold_of)
FEATS = [c for c in train.columns if c not in {"well_id", "row_index", "dtvt", "fold"}]
print(f"train rows {len(train):,} | features {len(FEATS)}")

params = dict(objective="regression", n_estimators=1500, learning_rate=0.03,
              num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              min_child_samples=200, reg_lambda=2.0, random_state=42, verbose=-1, n_jobs=-1)
oof = np.zeros(len(train)); models = []
for fld in range(5):
    tr = train["fold"] != fld; va = train["fold"] == fld
    m = lgb.LGBMRegressor(**params).fit(train.loc[tr, FEATS], train.loc[tr, "dtvt"])
    oof[va.values] = m.predict(train.loc[va, FEATS]); models.append(m)
    tp = train.loc[va, "tvt_ps"].values
    print(f"  fold {fld}: RMSE {rmse(train.loc[va,'dtvt']+tp, oof[va.values]+tp):.3f} ft")

tp_all = train["tvt_ps"].values
print(f"\nOOF RMSE: {rmse(train['dtvt']+tp_all, oof+tp_all):.3f} ft  (03 was 15.249, flat 15.91)")
pd.DataFrame({"well_id": train["well_id"], "row_index": train["row_index"],
              "tvt_pred": oof + tp_all}).to_csv(os.path.join(HERE, "oof.csv"), index=False)

print("Predicting test wells...")
twids = [os.path.basename(f).replace("__horizontal_well.csv", "")
         for f in sorted(glob.glob(os.path.join(RAW, "test", "*__horizontal_well.csv")))]
test = pd.concat([features_for_well(*load("test", w), False) for w in twids], ignore_index=True)
tp = np.mean([m.predict(test[FEATS]) for m in models], axis=0) + test["tvt_ps"].values
pd.DataFrame({"well_id": test["well_id"], "row_index": test["row_index"],
              "tvt_pred": tp}).to_csv(os.path.join(HERE, "test_pred.csv"), index=False)
pd.DataFrame({"id": test["well_id"] + "_" + test["row_index"].astype(str), "tvt": tp}).to_csv(
    os.path.join(ROOT, "outputs", "submissions", "claude_gr_context_20260716.csv"), index=False)
print("wrote oof.csv, test_pred.csv, submission")
