"""
blend.py - shared out-of-fold weighted blend of the two agents' predictions.

Reads claude/ and codex/ oof.csv + test_pred.csv (columns well_id,row_index,
tvt_pred), tunes the blend weight honestly out-of-fold on the shared folds, prints
each model's and the blend's CV RMSE, and writes a blended submission.

Owned by Claude per the handoff log; either agent may run it.
Run:  python shared/blend.py
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from metric import rmse  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
folds = pd.read_csv(os.path.join(HERE, "folds.csv"))[["well_id", "fold"]]


def read(agent, name):
    p = os.path.join(ROOT, agent, name)
    return pd.read_csv(p).rename(columns={"tvt_pred": agent})


# --- OOF blend --------------------------------------------------------------
df = read("claude", "oof.csv").merge(read("codex", "oof.csv"),
                                     on=["well_id", "row_index"], how="inner")
truth = []
for wid, g in df.groupby("well_id", sort=False):
    tvt = pd.read_csv(os.path.join(RAW, "train", f"{wid}__horizontal_well.csv"),
                      usecols=["TVT"])["TVT"].values
    truth.append(pd.Series(tvt[g["row_index"].values], index=g.index))
df["y"] = pd.concat(truth)
df = df.merge(folds, on="well_id", how="left")

ws = np.linspace(0, 1, 41)
print(f"merged OOF rows: {len(df):,}")
print(f"  claude: {rmse(df['y'], df['claude']):.4f} ft")
print(f"  codex : {rmse(df['y'], df['codex']):.4f} ft")

oof_blend = np.zeros(len(df))
fold_w = []
for fld in range(5):
    tr, va = df["fold"] != fld, df["fold"] == fld
    errs = [rmse(df.loc[tr, "y"], w * df.loc[tr, "claude"] + (1 - w) * df.loc[tr, "codex"]) for w in ws]
    w = ws[int(np.argmin(errs))]
    fold_w.append(w)
    oof_blend[va.values] = w * df.loc[va, "claude"] + (1 - w) * df.loc[va, "codex"]
print(f"  blend (OOF-honest): {rmse(df['y'], oof_blend):.4f} ft   per-fold w={ [round(x,2) for x in fold_w] }")

w_final = float(np.mean(fold_w))
print(f"  final blend weight (mean of folds): claude {w_final:.2f} / codex {1-w_final:.2f}")

# --- Test submission --------------------------------------------------------
t = read("claude", "test_pred.csv").merge(read("codex", "test_pred.csv"),
                                          on=["well_id", "row_index"], how="inner")
t["tvt"] = w_final * t["claude"] + (1 - w_final) * t["codex"]
sub = pd.DataFrame({"id": t["well_id"] + "_" + t["row_index"].astype(str), "tvt": t["tvt"]})
out = os.path.join(ROOT, "outputs", "submissions", "blend_claude_codex_20260716.csv")
sub.to_csv(out, index=False)
print(f"\nwrote blended submission: {os.path.relpath(out, ROOT)} ({len(sub)} rows)")
