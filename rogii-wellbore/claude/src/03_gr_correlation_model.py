"""
03_gr_correlation_model.py  (Claude's pipeline)

GR is the only real signal (see 02). This model inverts GR -> TVT position:
for each toe point, find the TVT shift (around the PS anchor) whose reference GR
best matches the point's GR. Two references, per the task deck:
  - the TYPE-LOG: type-well GR as a function of TVT.
  - the SELF-LOG: the lateral's OWN pre-PS GR-vs-TVT (higher resolution).
These GR-implied dTVT estimates (raw + sequence-smoothed) plus GR/geometry context
feed a LightGBM that predicts dTVT = TVT - TVT_PS. Trained with the shared group
folds; also predicts the 3 test wells.

Writes claude/oof.csv, claude/test_pred.csv (well_id,row_index,tvt_pred) and a
submission. Run:  python claude/src/03_gr_correlation_model.py
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

SHIFTS = np.arange(-40.0, 40.001, 0.5)     # candidate dTVT search grid (ft)


def roll(a, w, fn):
    s = pd.Series(a)
    return getattr(s.rolling(w, center=True, min_periods=1), fn)().values


def implied_dtvt(gr_smooth, ref_tvt, ref_gr, tvt_ps):
    """For each point's smoothed GR, the shift s whose reference GR(TVT_PS+s) is
    closest. Reference is (ref_tvt, ref_gr); grid is TVT_PS + SHIFTS."""
    if len(ref_tvt) < 3 or not np.isfinite(ref_gr).any():
        return np.zeros_like(gr_smooth)
    order = np.argsort(ref_tvt)
    grid_gr = np.interp(tvt_ps + SHIFTS, np.asarray(ref_tvt)[order],
                        np.asarray(ref_gr)[order])           # (S,)
    diff = np.abs(grid_gr[None, :] - gr_smooth[:, None])     # (N, S)
    return SHIFTS[np.argmin(diff, axis=1)]


def features_for_well(hz, tw, is_train):
    ps = np.where(hz["TVT_input"].notna().values)[0].max()
    tvt_ps = float(hz["TVT_input"].iloc[ps])
    md_ps, z_ps = float(hz["MD"].iloc[ps]), float(hz["Z"].iloc[ps])

    gr = pd.Series(hz["GR"].values).interpolate(limit_direction="both").values
    gr_s = roll(gr, 5, "mean")
    gr_grad = np.gradient(gr_s)

    # references
    tw_tvt, tw_gr = tw["TVT"].values, tw["GR"].values
    pre = slice(0, ps + 1)
    self_tvt = hz["TVT_input"].values[pre]
    self_gr = gr_s[pre]

    toe = np.arange(ps + 1, len(hz))
    d_tw = implied_dtvt(gr_s[toe], tw_tvt, tw_gr, tvt_ps)
    d_self = implied_dtvt(gr_s[toe], self_tvt, self_gr, tvt_ps)

    feat = pd.DataFrame({
        "well_id": hz["well_id"].iloc[0],
        "row_index": toe,
        "tvt_ps": tvt_ps,
        "implied_tw": d_tw,
        "implied_self": d_self,
        "implied_tw_med": roll(d_tw, 21, "median"),
        "implied_self_med": roll(d_self, 21, "median"),
        "implied_tw_mean": roll(d_tw, 21, "mean"),
        "gr": gr[toe],
        "gr_smooth": gr_s[toe],
        "gr_grad": gr_grad[toe],
        "gr_vs_heel": gr_s[toe] - np.nanmean(self_gr),
        "dMD": hz["MD"].values[toe] - md_ps,
        "dZ": hz["Z"].values[toe] - z_ps,
        "incl": (hz["Z"].values[toe] - z_ps) / (hz["MD"].values[toe] - md_ps + 1e-6),
        "toe_frac": (toe - ps) / (len(hz) - ps),
        "heel_gr_std": float(np.nanstd(self_gr)),
    })
    if is_train:
        feat["dtvt"] = hz["TVT"].values[toe] - tvt_ps
    return feat


def load(split, wid):
    hz = pd.read_csv(os.path.join(RAW, split, f"{wid}__horizontal_well.csv"))
    hz["well_id"] = wid
    tw = pd.read_csv(os.path.join(RAW, split, f"{wid}__typewell.csv"))
    return hz, tw


# --- Build train features ---------------------------------------------------
print("Building train features...")
train_wids = [os.path.basename(f).replace("__horizontal_well.csv", "")
              for f in sorted(glob.glob(os.path.join(RAW, "train", "*__horizontal_well.csv")))]
parts = []
for wid in train_wids:
    hz, tw = load("train", wid)
    parts.append(features_for_well(hz, tw, is_train=True))
train = pd.concat(parts, ignore_index=True)
train["fold"] = train["well_id"].map(fold_of)
FEATS = [c for c in train.columns if c not in
         {"well_id", "row_index", "dtvt", "fold"}]
print(f"train rows {len(train):,} | features {len(FEATS)}")

params = dict(objective="regression", n_estimators=1200, learning_rate=0.03,
              num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              min_child_samples=200, reg_lambda=2.0, random_state=42, verbose=-1, n_jobs=-1)

oof = np.zeros(len(train))
models = []
for fld in range(5):
    tr = train["fold"] != fld
    va = train["fold"] == fld
    m = lgb.LGBMRegressor(**params)
    m.fit(train.loc[tr, FEATS], train.loc[tr, "dtvt"])
    oof[va.values] = m.predict(train.loc[va, FEATS])
    models.append(m)
    tp = train.loc[va, "tvt_ps"].values
    print(f"  fold {fld}: RMSE {rmse(train.loc[va,'dtvt']+tp, oof[va.values]+tp):.3f} ft")

tvt_true = train["dtvt"] + train["tvt_ps"]
tvt_pred = oof + train["tvt_ps"]
print(f"\nOOF RMSE: {rmse(tvt_true, tvt_pred):.3f} ft  (flat baseline 15.91)")

# save OOF (absolute tvt, per interface spec)
pd.DataFrame({"well_id": train["well_id"], "row_index": train["row_index"],
              "tvt_pred": tvt_pred}).to_csv(os.path.join(HERE, "oof.csv"), index=False)

# --- Test wells -------------------------------------------------------------
print("\nPredicting test wells...")
test_wids = [os.path.basename(f).replace("__horizontal_well.csv", "")
             for f in sorted(glob.glob(os.path.join(RAW, "test", "*__horizontal_well.csv")))]
tparts = []
for wid in test_wids:
    hz, tw = load("test", wid)
    tparts.append(features_for_well(hz, tw, is_train=False))
test = pd.concat(tparts, ignore_index=True)
test_dtvt = np.mean([m.predict(test[FEATS]) for m in models], axis=0)
test_tvt = test_dtvt + test["tvt_ps"].values
pd.DataFrame({"well_id": test["well_id"], "row_index": test["row_index"],
              "tvt_pred": test_tvt}).to_csv(os.path.join(HERE, "test_pred.csv"), index=False)

sub = pd.DataFrame({"id": test["well_id"] + "_" + test["row_index"].astype(str),
                    "tvt": test_tvt})
sub_path = os.path.join(ROOT, "outputs", "submissions",
                        "claude_gr_corr_20260716.csv")
sub.to_csv(sub_path, index=False)
print(f"wrote oof.csv, test_pred.csv, and {os.path.basename(sub_path)}")
