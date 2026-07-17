"""
ROGII Wellbore Geology Prediction - Claude+Codex blended submission notebook.

Self-contained: paste into a Kaggle Notebook (attach the competition data), run,
and it writes /kaggle/working/submission.csv. Also runs locally against
data/raw for validation. Blends two independent pipelines:
  - Claude: GR-correlation + LightGBM on dTVT
  - Codex : geometry anchor + ridge + fold-safe offset/residual priors
Both train on ALL train wells and predict whatever test wells are present, so it
works when Kaggle reruns it on the hidden test set.
"""
import os
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb

# ---- path switch: Kaggle vs local (robust: find the dir that holds the data) ----
def _find_input():
    if os.path.isdir("/kaggle/input"):
        print("kaggle input dirs:", sorted(os.listdir("/kaggle/input")))
        # data may be at /kaggle/input/<slug> or /kaggle/input/competitions/<slug>
        for pat in ("/kaggle/input/*", "/kaggle/input/*/*"):
            for d in sorted(glob.glob(pat)):
                if (os.path.isdir(d) and os.path.exists(os.path.join(d, "sample_submission.csv"))
                        and os.path.isdir(os.path.join(d, "train"))):
                    return d, "/kaggle/working"
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "data", "raw"), here


INPUT, OUT = _find_input()
print("INPUT =", INPUT)
TRAIN_DIR = os.path.join(INPUT, "train")
TEST_DIR = os.path.join(INPUT, "test")
SAMPLE = os.path.join(INPUT, "sample_submission.csv")
W_CLAUDE = 0.30                     # OOF-tuned blend weight (rest -> Codex)

# =====================================================================
# CLAUDE pipeline: GR-correlation + LightGBM (from claude/src/03)
# =====================================================================
SHIFTS = np.arange(-40.0, 40.001, 0.5)


def _roll(a, w, fn):
    return getattr(pd.Series(a).rolling(w, center=True, min_periods=1), fn)().values


def _implied(gr_pt, ref_tvt, ref_gr, tvt_ps):
    if len(ref_tvt) < 3 or not np.isfinite(ref_gr).any():
        return np.zeros_like(gr_pt)
    o = np.argsort(ref_tvt)
    grid = np.interp(tvt_ps + SHIFTS, np.asarray(ref_tvt)[o], np.asarray(ref_gr)[o])
    return SHIFTS[np.argmin(np.abs(grid[None, :] - gr_pt[:, None]), axis=1)]


def _claude_feats(hz, tw, is_train):
    ps = np.where(hz["TVT_input"].notna().values)[0].max()
    tvt_ps = float(hz["TVT_input"].iloc[ps])
    md_ps, z_ps = float(hz["MD"].iloc[ps]), float(hz["Z"].iloc[ps])
    gr = pd.Series(hz["GR"].values).interpolate(limit_direction="both").values
    gr_s = _roll(gr, 5, "mean")
    grad = np.gradient(gr_s)
    self_tvt, self_gr = hz["TVT_input"].values[:ps + 1], gr_s[:ps + 1]
    toe = np.arange(ps + 1, len(hz))
    d_tw = _implied(gr_s[toe], tw["TVT"].values, _roll(tw["GR"].values, 5, "mean"), tvt_ps)
    d_self = _implied(gr_s[toe], self_tvt, self_gr, tvt_ps)
    feat = pd.DataFrame({
        "well_id": hz["well_id"].iloc[0], "row_index": toe, "tvt_ps": tvt_ps,
        "implied_tw": d_tw, "implied_self": d_self,
        "implied_tw_med": _roll(d_tw, 21, "median"),
        "implied_self_med": _roll(d_self, 21, "median"),
        "implied_mean": _roll(0.5 * (d_tw + d_self), 41, "median"),
        "agree": np.abs(d_tw - d_self),
        "gr_s": gr_s[toe], "grad": grad[toe], "gr_vs_heel": gr_s[toe] - np.nanmean(self_gr),
        "dMD": hz["MD"].values[toe] - md_ps, "dZ": hz["Z"].values[toe] - z_ps,
        "incl": (hz["Z"].values[toe] - z_ps) / (hz["MD"].values[toe] - md_ps + 1e-6),
        "toe_frac": (toe - ps) / (len(hz) - ps),
    })
    if is_train:
        feat["dtvt"] = hz["TVT"].values[toe] - tvt_ps
    return feat


def _load(split_dir, wid):
    hz = pd.read_csv(os.path.join(split_dir, f"{wid}__horizontal_well.csv")); hz["well_id"] = wid
    tw = pd.read_csv(os.path.join(split_dir, f"{wid}__typewell.csv"))
    return hz, tw


def claude_predict(train_ids, test_ids):
    train = pd.concat([_claude_feats(*_load(TRAIN_DIR, w), True) for w in train_ids], ignore_index=True)
    feats = [c for c in train.columns if c not in {"well_id", "row_index", "dtvt"}]
    model = lgb.LGBMRegressor(objective="regression", n_estimators=1200, learning_rate=0.03,
                              num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              min_child_samples=200, reg_lambda=2.0, random_state=42, verbose=-1, n_jobs=-1)
    model.fit(train[feats], train["dtvt"])
    test = pd.concat([_claude_feats(*_load(TEST_DIR, w), False) for w in test_ids], ignore_index=True)
    pred = model.predict(test[feats]) + test["tvt_ps"].values
    return pd.DataFrame({"well_id": test["well_id"], "row_index": test["row_index"], "claude": pred})


# =====================================================================
# CODEX pipeline (verbatim from codex/train_predict.py, credit: Codex)
# =====================================================================
from dataclasses import dataclass


@dataclass
class WellMatrix:
    well_id: str
    row_index: np.ndarray
    x: np.ndarray
    base: np.ndarray
    tvt0: float
    ps_x: float
    ps_y: float
    azimuth: float
    toe_fraction: np.ndarray
    y: "np.ndarray | None" = None


def interp_nan(values):
    arr = np.asarray(values, dtype=float)
    if np.isfinite(arr).all():
        return arr
    idx = np.arange(len(arr)); mask = np.isfinite(arr)
    if not mask.any():
        return np.zeros(len(arr), dtype=float)
    return np.interp(idx, idx[mask], arr[mask])


def centered_mean(values, window):
    arr = interp_nan(values); radius = window // 2
    csum = np.concatenate([[0.0], np.cumsum(arr)]); out = np.empty(len(arr))
    for i in range(len(arr)):
        lo = max(0, i - radius); hi = min(len(arr), i + radius + 1)
        out[i] = (csum[hi] - csum[lo]) / (hi - lo)
    return out


def safe_polyfit_slope(x, y, default):
    if len(x) < 3 or np.nanstd(x) == 0:
        return default
    return float(np.polyfit(x, y, 1)[0])


def cx_build(well_id, split_dir, include_target):
    horiz = pd.read_csv(os.path.join(split_dir, f"{well_id}__horizontal_well.csv"))
    typewell = pd.read_csv(os.path.join(split_dir, f"{well_id}__typewell.csv"))
    ps = int(horiz["TVT_input"].notna().sum())
    row_index = np.arange(ps, len(horiz), dtype=int)
    md = horiz["MD"].to_numpy(float); x_c = horiz["X"].to_numpy(float)
    y_c = horiz["Y"].to_numpy(float); z_c = horiz["Z"].to_numpy(float)
    gr = interp_nan(horiz["GR"].to_numpy(float))
    g11, g51, g101, g501 = (centered_mean(gr, w) for w in (11, 51, 101, 501))
    tvt_input = horiz["TVT_input"].to_numpy(float)[:ps]
    tvt0 = float(tvt_input[-1]); md0 = float(md[ps - 1])
    x0 = float(x_c[ps - 1]); y0 = float(y_c[ps - 1]); z0 = float(z_c[ps - 1])
    base = tvt0 + (z0 - z_c[row_index])
    t_tvt = typewell["TVT"].to_numpy(float); t_gr = interp_nan(typewell["GR"].to_numpy(float))
    order = np.argsort(t_tvt)
    gr_at_base = np.interp(base, t_tvt[order], t_gr[order], left=t_gr[order][0], right=t_gr[order][-1])
    rel = (row_index - ps + 1).astype(float)
    md_d = md[row_index] - md0; x_d = x_c[row_index] - x0; y_d = y_c[row_index] - y0
    z_d = z_c[row_index] - z0; xy_d = np.sqrt(x_d * x_d + y_d * y_d)
    azimuth = float(np.arctan2(y_c[-1] - y0, x_c[-1] - x0))
    last_gr = gr[max(0, ps - 500):ps]; last_gr_50 = gr[max(0, ps - 50):ps]
    tn = min(200, ps)
    slope_md = safe_polyfit_slope(md[ps - tn:ps], tvt_input[-tn:], 0.0)
    slope_z = safe_polyfit_slope(z_c[ps - tn:ps], tvt_input[-tn:], -1.0)
    flat = []
    for off in [-60., -30., -15., 0., 15., 30., 60.]:
        v = np.interp(np.full(len(row_index), tvt0 + off), t_tvt[order], t_gr[order],
                      left=t_gr[order][0], right=t_gr[order][-1])
        flat.append(v / 100.0); flat.append((g101[row_index] - v) / 100.0)
    features = np.column_stack([
        np.ones(len(row_index)), rel / 1000., (rel / 1000.) ** 2, (rel / 1000.) ** 3,
        md_d / 1000., (md_d / 1000.) ** 2, z_d / 100., (z_d / 100.) ** 2,
        x_d / 1000., y_d / 1000., xy_d / 1000., gr[row_index] / 100., g51[row_index] / 100.,
        gr_at_base / 100., (g51[row_index] - gr_at_base) / 100.,
        np.full(len(row_index), tvt0 / 10000.), np.full(len(row_index), ps / 2000.),
        np.full(len(row_index), float(np.nanmean(last_gr)) / 100.),
        np.full(len(row_index), float(np.nanstd(last_gr)) / 50.),
        np.full(len(row_index), slope_md), np.full(len(row_index), slope_z + 1.),
        g11[row_index] / 100., g101[row_index] / 100., g501[row_index] / 100.,
        (g11[row_index] - g101[row_index]) / 50., (g101[row_index] - g501[row_index]) / 50.,
        (g101[row_index] - float(np.nanmean(last_gr))) / 100.,
        (g101[row_index] - float(np.nanmean(last_gr_50))) / 100., *flat])
    target = horiz["TVT"].to_numpy(float)[row_index] if include_target else None
    return WellMatrix(well_id, row_index, features.astype(float), base.astype(float),
                      tvt0, x0, y0, azimuth, rel / max(1.0, float(len(row_index))), target)


def offset_prior(query, pool, k):
    nb = [((o.ps_x - query.ps_x) ** 2 + (o.ps_y - query.ps_y) ** 2, o) for o in pool
          if o.well_id != query.well_id and o.y is not None]
    if not nb:
        return np.zeros(len(query.row_index))
    nb.sort(key=lambda i: i[0]); vals = []; wts = []
    for d2, o in nb[:k]:
        od = o.y - o.tvt0
        vals.append(np.interp(query.toe_fraction, o.toe_fraction, od, left=od[0], right=od[-1]))
        wts.append(1.0 / np.sqrt(d2 + 1e-6))
    w = np.asarray(wts); return (np.vstack(vals) * w[:, None]).sum(0) / w.sum()


def with_offset(well, pool, k):
    p = offset_prior(well, pool, k) / 20.0
    return WellMatrix(well.well_id, well.row_index, np.column_stack([well.x, p, p * p]),
                      well.base, well.tvt0, well.ps_x, well.ps_y, well.azimuth, well.toe_fraction, well.y)


def angle_distance(a, b):
    return float(abs(np.arctan2(np.sin(a - b), np.cos(a - b))))


def residual_prior(query, pool, curves, k, azw):
    nb = []
    for o in pool:
        if o.well_id == query.well_id or o.well_id not in curves:
            continue
        d = np.sqrt((o.ps_x - query.ps_x) ** 2 + (o.ps_y - query.ps_y) ** 2)
        if azw:
            d *= 1.0 + azw * angle_distance(o.azimuth, query.azimuth) / np.pi
        nb.append((d, o))
    if not nb:
        return np.zeros(len(query.row_index))
    nb.sort(key=lambda i: i[0]); vals = []; wts = []
    for d, o in nb[:k]:
        r = curves[o.well_id]
        vals.append(np.interp(query.toe_fraction, o.toe_fraction, r, left=r[0], right=r[-1]))
        wts.append(1.0 / np.sqrt(d * d + 1e-6))
    w = np.asarray(wts); return (np.vstack(vals) * w[:, None]).sum(0) / w.sum()


def fit_ridge(wells, ridge):
    x = np.vstack([w.x for w in wells])
    resid = np.concatenate([w.y - w.base for w in wells if w.y is not None])
    mu = x[:, 1:].mean(0); sd = x[:, 1:].std(0); sd[sd == 0] = 1.0
    xs = x.copy(); xs[:, 1:] = (xs[:, 1:] - mu) / sd
    pen = ridge * np.eye(xs.shape[1]); pen[0, 0] = 0.0
    return np.linalg.solve(xs.T @ xs + pen, xs.T @ resid), mu, sd


def cx_predict(well, coef, mu, sd):
    x = well.x.copy(); x[:, 1:] = (x[:, 1:] - mu) / sd
    return well.base + x @ coef


def scale_from_ps(well, pred, s):
    return well.tvt0 + s * (pred - well.tvt0)


def codex_predict(train_ids, test_ids):
    RIDGE, KS, WS, SCALE = 1000.0, [10, 60], [0.52, 0.48], 0.92
    RK, RA, RAZ = 10, 0.435, 2.0
    train = {w: cx_build(w, TRAIN_DIR, True) for w in train_ids}
    test = {w: cx_build(w, TEST_DIR, False) for w in test_ids}
    pool = list(train.values())
    tr_pred = {w: np.zeros(len(train[w].row_index)) for w in train_ids}
    te_pred = {w: np.zeros(len(test[w].row_index)) for w in test_ids}
    for k, wt in zip(KS, WS):
        tw = [with_offset(train[w], pool, k) for w in train_ids]
        coef, mu, sd = fit_ridge(tw, RIDGE)
        for well in tw:
            tr_pred[well.well_id] += wt * scale_from_ps(well, cx_predict(well, coef, mu, sd), SCALE)
        for w in test_ids:
            wp = with_offset(test[w], pool, k)
            te_pred[w] += wt * scale_from_ps(wp, cx_predict(wp, coef, mu, sd), SCALE)
    curves = {w: train[w].y - tr_pred[w] for w in train_ids}
    out = []
    for w in test_ids:
        pred = te_pred[w] + RA * residual_prior(test[w], pool, curves, RK, RAZ)
        out.append(pd.DataFrame({"well_id": w, "row_index": test[w].row_index, "codex": pred}))
    return pd.concat(out, ignore_index=True)


# =====================================================================
# Blend + write submission
# =====================================================================
def main():
    train_ids = sorted(os.path.basename(f).replace("__horizontal_well.csv", "")
                       for f in glob.glob(os.path.join(TRAIN_DIR, "*__horizontal_well.csv")))
    sample = pd.read_csv(SAMPLE)
    test_ids = sorted(sample["id"].str.rsplit("_", n=1).str[0].unique())
    print(f"train wells {len(train_ids)} | test wells {len(test_ids)}")

    c = claude_predict(train_ids, test_ids)
    x = codex_predict(train_ids, test_ids)
    b = c.merge(x, on=["well_id", "row_index"])
    b["tvt"] = W_CLAUDE * b["claude"] + (1 - W_CLAUDE) * b["codex"]
    b["id"] = b["well_id"] + "_" + b["row_index"].astype(str)
    sub = sample[["id"]].merge(b[["id", "tvt"]], on="id", how="left")
    assert sub["tvt"].notna().all(), "missing predictions"
    path = os.path.join(OUT, "submission.csv")
    sub.to_csv(path, index=False)
    print(f"wrote {path} ({len(sub)} rows), tvt range {sub.tvt.min():.1f}-{sub.tvt.max():.1f}")


if __name__ == "__main__":
    main()
