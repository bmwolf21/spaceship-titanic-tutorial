"""
01_geometric_baseline.py  (Claude's pipeline)

First baseline: TVT changes smoothly along the well as it drills at some apparent
dip through the geology. The heel (rows with TVT_input) gives TVT vs MD; predict
the toe by extrapolating the local TVT-vs-MD slope from the end of the heel.

No cross-well training yet, so this is a per-well method; the shared folds still
let us report a fold-wise RMSE comparable to later ML models and to Codex.

Reports RMSE (shared metric) and writes claude/oof.csv for later blending.
Run:  python claude/src/01_geometric_baseline.py
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # claude/
ROOT = os.path.dirname(HERE)                                           # rogii-wellbore/
sys.path.insert(0, os.path.join(ROOT, "shared"))
from metric import rmse  # noqa: E402

TRAIN = os.path.join(ROOT, "data", "raw", "train")
folds = pd.read_csv(os.path.join(ROOT, "shared", "folds.csv"))
fold_of = dict(zip(folds["well_id"], folds["fold"]))

TAIL = 100   # heel points used to estimate the end slope


def predict_well(df):
    """Return (toe_index, tvt_true, tvt_pred) for one horizontal well."""
    ps_mask = df["TVT_input"].notna()
    ps = np.where(ps_mask.values)[0].max()            # last known (Prediction Start)
    heel = df.iloc[max(0, ps - TAIL):ps + 1]
    md = heel["MD"].values
    tvt = heel["TVT_input"].values
    # robust local slope of TVT vs MD near PS
    slope, intercept = np.polyfit(md, tvt, 1)
    tvt_ps = df["TVT_input"].iloc[ps]
    md_ps = df["MD"].iloc[ps]
    toe = df.iloc[ps + 1:]
    pred = tvt_ps + slope * (toe["MD"].values - md_ps)
    return toe.index.values, toe["TVT"].values, pred, slope


rows = []
per_fold = {f: {"t": [], "p": []} for f in range(5)}
naive_all_t, naive_all_p = [], []
for f in sorted(glob.glob(os.path.join(TRAIN, "*__horizontal_well.csv"))):
    wid = os.path.basename(f).replace("__horizontal_well.csv", "")
    df = pd.read_csv(f, usecols=["MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"])
    idx, t, p, slope = predict_well(df)
    fold = fold_of.get(wid, -1)
    for i, tv, pv in zip(idx, t, p):
        rows.append((wid, int(i), tv, pv))
    per_fold[fold]["t"].extend(t); per_fold[fold]["p"].extend(p)
    # naive reference: hold TVT flat at PS value
    tvt_ps = df["TVT_input"].dropna().iloc[-1]
    naive_all_t.extend(t); naive_all_p.extend([tvt_ps] * len(t))

oof = pd.DataFrame(rows, columns=["well_id", "row_index", "tvt_true", "tvt_pred"])
oof.to_csv(os.path.join(HERE, "oof.csv"), index=False)

print("Geometric extrapolation baseline (per-well end-slope)")
print(f"  overall RMSE: {rmse(oof['tvt_true'], oof['tvt_pred']):.3f} ft")
for fld in range(5):
    d = per_fold[fld]
    print(f"  fold {fld}: RMSE {rmse(d['t'], d['p']):.3f} ft  (n={len(d['t']):,})")
print(f"\n  naive (flat at PS) RMSE: {rmse(naive_all_t, naive_all_p):.3f} ft  (reference)")
print(f"\nwrote {os.path.join(HERE, 'oof.csv')}  ({len(oof):,} toe points)")
