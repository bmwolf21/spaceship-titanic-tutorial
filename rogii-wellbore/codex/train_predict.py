#!/usr/bin/env python3
"""Codex independent pipeline for ROGII Wellbore Geology Prediction.

The model uses only inference-available horizontal-well columns, TVT_input in
the known heel interval, and the paired type-well GR-vs-TVT curve. It trains a
closed-form ridge residual model on top of a geometry anchor and evaluates on
the shared whole-well folds.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.metric import score  # noqa: E402


@dataclass
class WellMatrix:
    well_id: str
    row_index: np.ndarray
    x: np.ndarray
    base: np.ndarray
    y: np.ndarray | None = None


def interp_nan(values: np.ndarray) -> np.ndarray:
    """Linear fill for NaNs, with constant edge extension."""
    arr = np.asarray(values, dtype=float)
    if np.isfinite(arr).all():
        return arr
    idx = np.arange(len(arr))
    mask = np.isfinite(arr)
    if not mask.any():
        return np.zeros(len(arr), dtype=float)
    return np.interp(idx, idx[mask], arr[mask])


def centered_mean(values: np.ndarray, window: int) -> np.ndarray:
    arr = interp_nan(values)
    radius = window // 2
    csum = np.concatenate([[0.0], np.cumsum(arr)])
    out = np.empty(len(arr), dtype=float)
    for i in range(len(arr)):
        lo = max(0, i - radius)
        hi = min(len(arr), i + radius + 1)
        out[i] = (csum[hi] - csum[lo]) / (hi - lo)
    return out


def safe_polyfit_slope(x: np.ndarray, y: np.ndarray, default: float) -> float:
    if len(x) < 3 or np.nanstd(x) == 0:
        return default
    return float(np.polyfit(x, y, 1)[0])


def build_well_matrix(
    well_id: str,
    split_dir: Path,
    include_target: bool,
) -> WellMatrix:
    horiz = pd.read_csv(split_dir / f"{well_id}__horizontal_well.csv")
    typewell = pd.read_csv(split_dir / f"{well_id}__typewell.csv")

    ps = int(horiz["TVT_input"].notna().sum())
    if ps == 0:
        raise ValueError(f"{well_id}: no known TVT_input interval")

    row_index = np.arange(ps, len(horiz), dtype=int)
    if len(row_index) == 0:
        raise ValueError(f"{well_id}: no post-PS rows to predict")

    md = horiz["MD"].to_numpy(dtype=float)
    x_coord = horiz["X"].to_numpy(dtype=float)
    y_coord = horiz["Y"].to_numpy(dtype=float)
    z_coord = horiz["Z"].to_numpy(dtype=float)
    gr = interp_nan(horiz["GR"].to_numpy(dtype=float))
    gr_smooth = centered_mean(gr, 51)

    tvt_input = horiz["TVT_input"].to_numpy(dtype=float)[:ps]
    tvt0 = float(tvt_input[-1])
    md0 = float(md[ps - 1])
    x0 = float(x_coord[ps - 1])
    y0 = float(y_coord[ps - 1])
    z0 = float(z_coord[ps - 1])

    base = tvt0 + (z0 - z_coord[row_index])

    type_tvt = typewell["TVT"].to_numpy(dtype=float)
    type_gr = interp_nan(typewell["GR"].to_numpy(dtype=float))
    order = np.argsort(type_tvt)
    type_gr_at_base = np.interp(
        base,
        type_tvt[order],
        type_gr[order],
        left=type_gr[order][0],
        right=type_gr[order][-1],
    )

    rel = (row_index - ps + 1).astype(float)
    md_delta = md[row_index] - md0
    x_delta = x_coord[row_index] - x0
    y_delta = y_coord[row_index] - y0
    z_delta = z_coord[row_index] - z0
    xy_delta = np.sqrt(x_delta * x_delta + y_delta * y_delta)

    last_gr = gr[max(0, ps - 500) : ps]
    trend_n = min(200, ps)
    slope_md = safe_polyfit_slope(md[ps - trend_n : ps], tvt_input[-trend_n:], 0.0)
    slope_z = safe_polyfit_slope(z_coord[ps - trend_n : ps], tvt_input[-trend_n:], -1.0)

    features = np.column_stack(
        [
            np.ones(len(row_index)),
            rel / 1000.0,
            (rel / 1000.0) ** 2,
            (rel / 1000.0) ** 3,
            md_delta / 1000.0,
            (md_delta / 1000.0) ** 2,
            z_delta / 100.0,
            (z_delta / 100.0) ** 2,
            x_delta / 1000.0,
            y_delta / 1000.0,
            xy_delta / 1000.0,
            gr[row_index] / 100.0,
            gr_smooth[row_index] / 100.0,
            type_gr_at_base / 100.0,
            (gr_smooth[row_index] - type_gr_at_base) / 100.0,
            np.full(len(row_index), tvt0 / 10000.0),
            np.full(len(row_index), ps / 2000.0),
            np.full(len(row_index), float(np.nanmean(last_gr)) / 100.0),
            np.full(len(row_index), float(np.nanstd(last_gr)) / 50.0),
            np.full(len(row_index), slope_md),
            np.full(len(row_index), slope_z + 1.0),
        ]
    )

    target = None
    if include_target:
        target = horiz["TVT"].to_numpy(dtype=float)[row_index]

    return WellMatrix(
        well_id=well_id,
        row_index=row_index,
        x=features.astype(float),
        base=base.astype(float),
        y=target,
    )


def fit_ridge(train_wells: Iterable[WellMatrix], ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.vstack([well.x for well in train_wells])
    residual = np.concatenate([well.y - well.base for well in train_wells if well.y is not None])

    mu = x[:, 1:].mean(axis=0)
    sigma = x[:, 1:].std(axis=0)
    sigma[sigma == 0] = 1.0

    xs = x.copy()
    xs[:, 1:] = (xs[:, 1:] - mu) / sigma
    penalty = ridge * np.eye(xs.shape[1])
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(xs.T @ xs + penalty, xs.T @ residual)
    return coef, mu, sigma


def predict(well: WellMatrix, coef: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    x = well.x.copy()
    x[:, 1:] = (x[:, 1:] - mu) / sigma
    return well.base + x @ coef


def parse_submission_ids(sample: pd.DataFrame) -> pd.DataFrame:
    split = sample["id"].str.rsplit("_", n=1, expand=True)
    return pd.DataFrame({"id": sample["id"], "well_id": split[0], "row_index": split[1].astype(int)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--folds", type=Path, default=ROOT / "shared" / "folds.csv")
    parser.add_argument("--ridge", type=float, default=1000.0)
    parser.add_argument("--submission-date", default=datetime.now().strftime("%Y%m%d"))
    args = parser.parse_args()

    codex_dir = ROOT / "codex"
    sub_dir = ROOT / "outputs" / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)

    folds = pd.read_csv(args.folds)
    fold_by_well = dict(zip(folds["well_id"], folds["fold"]))

    train_dir = args.data_dir / "train"
    test_dir = args.data_dir / "test"
    train_wells = {
        well_id: build_well_matrix(well_id, train_dir, include_target=True)
        for well_id in folds["well_id"]
    }

    oof_parts = []
    fold_metrics = {}
    y_all = []
    pred_all = []
    for fold in sorted(folds["fold"].unique()):
        train_ids = [wid for wid in folds["well_id"] if fold_by_well[wid] != fold]
        valid_ids = [wid for wid in folds["well_id"] if fold_by_well[wid] == fold]
        coef, mu, sigma = fit_ridge([train_wells[wid] for wid in train_ids], args.ridge)

        fold_y = []
        fold_pred = []
        for wid in valid_ids:
            well = train_wells[wid]
            pred = predict(well, coef, mu, sigma)
            fold_y.append(well.y)
            fold_pred.append(pred)
            oof_parts.append(
                pd.DataFrame(
                    {
                        "well_id": wid,
                        "row_index": well.row_index,
                        "tvt_pred": pred,
                    }
                )
            )

        y = np.concatenate(fold_y)
        pred = np.concatenate(fold_pred)
        fold_metrics[str(int(fold))] = {
            "rmse": score(y, pred),
            "n_rows": int(len(y)),
            "n_wells": int(len(valid_ids)),
        }
        y_all.append(y)
        pred_all.append(pred)

    overall_rmse = score(np.concatenate(y_all), np.concatenate(pred_all))
    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(codex_dir / "oof.csv", index=False)

    coef, mu, sigma = fit_ridge(train_wells.values(), args.ridge)

    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    sample_ids = parse_submission_ids(sample)
    test_well_ids = sorted(sample_ids["well_id"].unique())
    test_wells = {
        well_id: build_well_matrix(well_id, test_dir, include_target=False)
        for well_id in test_well_ids
    }

    test_parts = []
    pred_lookup = {}
    for wid, well in test_wells.items():
        pred = predict(well, coef, mu, sigma)
        test_parts.append(
            pd.DataFrame(
                {
                    "well_id": wid,
                    "row_index": well.row_index,
                    "tvt_pred": pred,
                }
            )
        )
        pred_lookup.update({f"{wid}_{idx}": val for idx, val in zip(well.row_index, pred)})

    test_pred = pd.concat(test_parts, ignore_index=True)
    test_pred.to_csv(codex_dir / "test_pred.csv", index=False)

    submission = sample[["id"]].copy()
    submission["tvt"] = submission["id"].map(pred_lookup)
    if submission["tvt"].isna().any():
        missing = submission.loc[submission["tvt"].isna(), "id"].head().tolist()
        raise ValueError(f"Missing predictions for sample ids: {missing}")

    submission_path = sub_dir / f"codex_ridge_residual_{args.submission_date}.csv"
    submission.to_csv(submission_path, index=False)

    metrics = {
        "model": "geometry_anchor_plus_ridge_residual",
        "ridge": args.ridge,
        "overall_rmse": overall_rmse,
        "folds": fold_metrics,
        "n_oof_rows": int(len(oof)),
        "n_test_rows": int(len(test_pred)),
        "submission": str(submission_path.relative_to(ROOT)),
        "features": [
            "relative row and MD terms",
            "relative X/Y/Z trajectory terms",
            "current and smoothed GR",
            "type-well GR interpolated at geometry-anchor TVT",
            "heel GR summary stats",
            "last-heel TVT trend slopes",
        ],
    }
    with open(codex_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

