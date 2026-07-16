"""
07_stack.py  (Claude's pipeline)

My standalone GR models (03=15.25; 05,06 failed) won't beat Codex solo (14.36).
Higher-value move, and mine to own: improve the BLEND. Instead of one global
weight, stack the two agents with a small meta-model that can weight them
adaptively by context (where they disagree, and where along the toe we are).
Also test the global dTVT-shrinkage trick.

Compares: weighted blend (current 14.086) vs shrunk blend vs GBM stack, all on the
shared group folds. Writes a stacked submission if it wins.
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
folds = pd.read_csv(os.path.join(ROOT, "shared", "folds.csv"))[["well_id", "fold"]]


def read(agent, name):
    return pd.read_csv(os.path.join(ROOT, agent, name)).rename(columns={"tvt_pred": agent})


df = read("claude", "oof.csv").merge(read("codex", "oof.csv"),
                                     on=["well_id", "row_index"], how="inner")

# per-row truth + context (TVT_PS, toe_frac) in one pass over wells
truth, tvtps, toefrac = [], [], []
for wid, g in df.groupby("well_id", sort=False):
    hz = pd.read_csv(os.path.join(RAW, "train", f"{wid}__horizontal_well.csv"),
                     usecols=["TVT", "TVT_input"])
    ps = np.where(hz["TVT_input"].notna().values)[0].max()
    tp = float(hz["TVT_input"].iloc[ps]); n = len(hz)
    idx = g["row_index"].values
    truth.append(pd.Series(hz["TVT"].values[idx], index=g.index))
    tvtps.append(pd.Series(np.full(len(g), tp), index=g.index))
    toefrac.append(pd.Series((idx - ps) / (n - ps), index=g.index))
df["y"] = pd.concat(truth); df["tvt_ps"] = pd.concat(tvtps); df["toe_frac"] = pd.concat(toefrac)
df = df.merge(folds, on="well_id", how="left")

# work in dTVT space (small target, easier + comparable)
for a in ["claude", "codex"]:
    df[a + "_d"] = df[a] - df["tvt_ps"]
df["y_d"] = df["y"] - df["tvt_ps"]
df["disagree"] = df["claude_d"] - df["codex_d"]

print(f"rows {len(df):,}")
print(f"  claude {rmse(df['y'], df['claude']):.4f} | codex {rmse(df['y'], df['codex']):.4f}")

# 1) global weighted blend (reference)
ws = np.linspace(0, 1, 41)
wb = min(ws, key=lambda w: rmse(df["y_d"], w*df["claude_d"]+(1-w)*df["codex_d"]))
print(f"  weighted blend (w={wb:.2f}): {rmse(df['y_d'], wb*df['claude_d']+(1-wb)*df['codex_d']):.4f}")

# 2) weighted blend + global dTVT shrink
best = None
for w in ws:
    b = w*df["claude_d"]+(1-w)*df["codex_d"]
    for sh in np.linspace(0.7, 1.05, 15):
        r = rmse(df["y_d"], sh*b)
        if best is None or r < best[0]:
            best = (r, w, sh)
print(f"  weighted+shrink (w={best[1]:.2f}, shrink={best[2]:.2f}): {best[0]:.4f}")

# 3) GBM stack: predict dTVT from the two base dTVTs + context, group-CV
FEATS = ["claude_d", "codex_d", "disagree", "toe_frac"]
params = dict(objective="regression", n_estimators=400, learning_rate=0.05,
              num_leaves=31, min_child_samples=500, reg_lambda=5.0,
              subsample=0.7, subsample_freq=1, random_state=42, verbose=-1, n_jobs=-1)
oof = np.zeros(len(df))
for fld in range(5):
    tr, va = df["fold"] != fld, df["fold"] == fld
    m = lgb.LGBMRegressor(**params).fit(df.loc[tr, FEATS], df.loc[tr, "y_d"])
    oof[va.values] = m.predict(df.loc[va, FEATS])
print(f"  GBM stack: {rmse(df['y_d'], oof):.4f}")
