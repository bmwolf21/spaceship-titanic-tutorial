"""
06_dp_correlation.py  (Claude's pipeline)

Proper geosteering inversion via dynamic programming. Instead of matching GR
per-point (noisy, my 03/05 weakness), we find the whole TVT(MD) path along the toe
that minimizes GR mismatch against the reference log, subject to a smoothness
penalty on how fast TVT can change. This can TRACK large stratigraphic excursions
(exactly the fold-3 hard wells, per Codex's diagnostic).

DP states = TVT offsets from TVT_PS on a 1-ft grid. Emission = |ref_GR(state) -
lateral_GR|. Transition = quadratic penalty on state change (solved exactly with
the Felzenszwalb 1-D distance transform, O(states) per step). Reference GR(TVT) is
the lateral's own pre-PS self-log, filled with the type-log where the self-log
does not reach.

This script evaluates the DP prediction standalone on the shared folds. Use
N_WELLS to test on a subset first.
"""
import os
import sys
import glob
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "shared"))
from metric import rmse  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
folds = pd.read_csv(os.path.join(ROOT, "shared", "folds.csv"))
fold_of = dict(zip(folds["well_id"], folds["fold"]))

N_WELLS = int(os.environ.get("N_WELLS", "0"))     # 0 = all
OFF = np.arange(-45.0, 45.001, 1.0)               # TVT-offset states (ft)
S = len(OFF)
LAM = float(os.environ.get("LAM", "3.0"))         # smoothness penalty (per ft^2)
ANCHOR = 8.0                                       # cost pulling first toe pt to PS
DS = 3                                             # toe downsample step (ft)


def dt_1d(f, lam):
    """Felzenszwalb 1-D distance transform: d[q]=min_p f[p]+lam*(q-p)^2, + argmin."""
    n = len(f)
    d = np.empty(n)
    arg = np.empty(n, dtype=np.int32)
    v = np.zeros(n, dtype=np.int32)
    z = np.empty(n + 1)
    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf
    for q in range(1, n):
        s = ((f[q] + lam * q * q) - (f[v[k]] + lam * v[k] * v[k])) / (2 * lam * (q - v[k]))
        while s <= z[k]:
            k -= 1
            s = ((f[q] + lam * q * q) - (f[v[k]] + lam * v[k] * v[k])) / (2 * lam * (q - v[k]))
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = lam * (q - v[k]) ** 2 + f[v[k]]
        arg[q] = v[k]
    return d, arg


def reference_gr(hz, tw, ps, tvt_ps, gr_s):
    grid = tvt_ps + OFF
    self_tvt = hz["TVT_input"].values[:ps + 1]
    self_gr = gr_s[:ps + 1]
    o = np.argsort(self_tvt)
    ref = np.interp(grid, self_tvt[o], self_gr[o], left=np.nan, right=np.nan)
    # fill gaps the self-log doesn't cover with the type-log
    tw_tvt, tw_gr = tw["TVT"].values, pd.Series(tw["GR"].values).rolling(5, 1, center=True).mean().values
    ot = np.argsort(tw_tvt)
    ref_tw = np.interp(grid, tw_tvt[ot], tw_gr[ot])
    ref = np.where(np.isfinite(ref), ref, ref_tw)
    return ref


def dp_predict_well(hz, tw):
    ps = np.where(hz["TVT_input"].notna().values)[0].max()
    tvt_ps = float(hz["TVT_input"].iloc[ps])
    gr = pd.Series(hz["GR"].values).interpolate(limit_direction="both").values
    gr_s = pd.Series(gr).rolling(9, 1, center=True).mean().values
    ref = reference_gr(hz, tw, ps, tvt_ps, gr_s)          # (S,)

    toe = np.arange(ps + 1, len(hz))
    ds = toe[::DS]
    g = gr_s[ds]                                          # (M,)
    E = np.abs(ref[None, :] - g[:, None])                 # (M, S) emission
    M = len(ds)

    back = np.empty((M, S), dtype=np.int32)
    cost = E[0] + ANCHOR * ((OFF - 0.0) / 5.0) ** 2       # anchor first pt near PS
    for i in range(1, M):
        prop, arg = dt_1d(cost, LAM)
        cost = E[i] + prop
        back[i] = arg
    path = np.empty(M, dtype=np.int32)
    path[-1] = int(np.argmin(cost))
    for i in range(M - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    dtvt_ds = OFF[path]
    dtvt = np.interp(toe, ds, dtvt_ds)                    # back to full toe
    return toe, tvt_ps + dtvt, hz["TVT"].values[toe]


wids = [os.path.basename(f).replace("__horizontal_well.csv", "")
        for f in sorted(glob.glob(os.path.join(RAW, "train", "*__horizontal_well.csv")))]
if N_WELLS:
    wids = wids[:N_WELLS]

t0 = time.time()
per_fold = {f: {"t": [], "p": []} for f in range(5)}
allt, allp = [], []
for j, wid in enumerate(wids):
    hz = pd.read_csv(os.path.join(RAW, "train", f"{wid}__horizontal_well.csv"))
    tw = pd.read_csv(os.path.join(RAW, "train", f"{wid}__typewell.csv"))
    _, pred, true = dp_predict_well(hz, tw)
    fld = fold_of[wid]
    per_fold[fld]["t"].extend(true); per_fold[fld]["p"].extend(pred)
    allt.extend(true); allp.extend(pred)
    if (j + 1) % 100 == 0:
        print(f"  {j+1}/{len(wids)} wells ({time.time()-t0:.0f}s)")

print(f"\nDP standalone (LAM={LAM}, DS={DS}): overall RMSE {rmse(allt, allp):.3f} ft  "
      f"(03 was 15.249, flat 15.91)")
for f in range(5):
    d = per_fold[f]
    if d["t"]:
        print(f"  fold {f}: RMSE {rmse(d['t'], d['p']):.3f} ft")
print(f"elapsed {time.time()-t0:.0f}s for {len(wids)} wells")
