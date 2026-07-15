"""
03_models_ensemble.py — v2 features, LightGBM + XGBoost, and a blend.

Angle 1 (compete): train two gradient-boosting models under identical 5-fold
stratified CV, then average their out-of-fold probabilities to see if blending
helps. Refit and predict the test set for a submission.
Angle 2 (document): prints per-model and blended CV accuracy for TUTORIAL.md.

Run:  python src/03_models_ensemble.py
"""
import os
import sys
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from features import build_features  # noqa: E402

RAW = os.path.join(HERE, "data", "raw")
SUB = os.path.join(HERE, "outputs", "submissions")
os.makedirs(SUB, exist_ok=True)

train = pd.read_csv(os.path.join(RAW, "train.csv"))
test = pd.read_csv(os.path.join(RAW, "test.csv"))
X, y, X_test, cols = build_features(train, test)
print(f"v2 features: {len(cols)}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def run_cv(make_model, name):
    """Train `make_model()` across folds; return (oof_proba, test_proba, mean_acc)."""
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    scores = []
    for fold, (tr, va) in enumerate(cv.split(X, y), 1):
        model = make_model()
        model.fit(
            X.iloc[tr], y.iloc[tr],
            eval_set=[(X.iloc[va], y.iloc[va])],
            **fit_kwargs(model),
        )
        p = model.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        test_pred += model.predict_proba(X_test)[:, 1] / cv.n_splits
        scores.append(accuracy_score(y.iloc[va], (p > 0.5).astype(int)))
    acc = accuracy_score(y, (oof > 0.5).astype(int))
    print(f"{name:10s}  CV acc {np.mean(scores):.4f} +/- {np.std(scores):.4f}  | OOF {acc:.4f}")
    return oof, test_pred, acc


def fit_kwargs(model):
    """Early-stopping kwargs differ slightly between the two libraries."""
    if isinstance(model, lgb.LGBMClassifier):
        return dict(callbacks=[lgb.early_stopping(50, verbose=False)])
    return dict(verbose=False)  # xgboost: early stopping set on the constructor


def make_lgb():
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=800, learning_rate=0.02,
        num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=42, verbose=-1,
    )


def make_xgb():
    return xgb.XGBClassifier(
        n_estimators=800, learning_rate=0.02, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        eval_metric="logloss", early_stopping_rounds=50,
        random_state=42, verbosity=0,
    )


oof_lgb, test_lgb, acc_lgb = run_cv(make_lgb, "LightGBM")
oof_xgb, test_xgb, acc_xgb = run_cv(make_xgb, "XGBoost")

# --- Feature importance (fit one LightGBM on all data) ----------------------
imp_model = make_lgb()
imp_model.fit(X, y)
imp = (pd.Series(imp_model.feature_importances_, index=cols)
       .sort_values(ascending=False))
print("\nTop 12 features (LightGBM gain-based importance):")
print(imp.head(12).to_string())

# --- Blend: search a simple weight on the LightGBM probability --------------
best_w, best_acc = 0.5, 0.0
for w in np.linspace(0, 1, 21):
    blend = w * oof_lgb + (1 - w) * oof_xgb
    a = accuracy_score(y, (blend > 0.5).astype(int))
    if a > best_acc:
        best_acc, best_w = a, w
print(f"\nBlend       best OOF acc {best_acc:.4f}  at w_lgb={best_w:.2f}")

test_blend = best_w * test_lgb + (1 - best_w) * test_xgb

# --- Submission from the best of {lgb, xgb, blend} --------------------------
options = {"lgbm": (acc_lgb, test_lgb), "xgb": (acc_xgb, test_xgb),
           "blend": (best_acc, test_blend)}
best_name = max(options, key=lambda k: options[k][0])
best_score, best_test = options[best_name]
print(f"Best model for submission: {best_name} (OOF {best_score:.4f})")

tag = dt.datetime.now().strftime("%Y%m%d_%H%M")
sub = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Transported": (best_test > 0.5).astype(bool),
})
path = os.path.join(SUB, f"submission_{best_name}_{tag}.csv")
sub.to_csv(path, index=False)
print(f"Wrote {path}")
