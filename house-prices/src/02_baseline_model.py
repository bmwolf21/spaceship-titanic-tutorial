"""
02_baseline_model.py - cross-validated LightGBM baseline + submission.

Regression on log1p(SalePrice); the competition metric is RMSE on the log scale,
so CV RMSE here is directly comparable to the leaderboard.

Run:  python src/02_baseline_model.py
"""
import os
import sys
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from features import build_features  # noqa: E402

RAW = os.path.join(HERE, "data", "raw")
SUB = os.path.join(HERE, "outputs", "submissions")
os.makedirs(SUB, exist_ok=True)

train = pd.read_csv(os.path.join(RAW, "train.csv"))
test = pd.read_csv(os.path.join(RAW, "test.csv"))
X, y, X_test, cols = build_features(train, test)
print(f"Features: {len(cols)}")

params = dict(objective="regression", n_estimators=2000, learning_rate=0.02,
              num_leaves=15, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.6, reg_lambda=1.0, min_child_samples=10,
              random_state=42, verbose=-1)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))
scores = []
for fold, (tr, va) in enumerate(cv.split(X), 1):
    m = lgb.LGBMRegressor(**params)
    m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[va] = m.predict(X.iloc[va])
    test_pred += m.predict(X_test) / cv.n_splits
    rmse = np.sqrt(mean_squared_error(y.iloc[va], oof[va]))
    scores.append(rmse)
    print(f"fold {fold}: RMSE(log) = {rmse:.5f}  (best_iter={m.best_iteration_})")

print(f"\nCV mean RMSE(log): {np.mean(scores):.5f} +/- {np.std(scores):.5f}")

# --- Submission (back-transform log1p -> price) -----------------------------
tag = dt.datetime.now().strftime("%Y%m%d_%H%M")
sub = pd.DataFrame({"Id": test["Id"], "SalePrice": np.expm1(test_pred)})
path = os.path.join(SUB, f"submission_lgbm_{tag}.csv")
sub.to_csv(path, index=False)
print(f"Wrote {path}  (median predicted price ${sub['SalePrice'].median():,.0f})")
