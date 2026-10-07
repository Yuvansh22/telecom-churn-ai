"""Train, tune, evaluate and save Logistic Regression, Random Forest and XGBoost.

Run:  python src/train_models.py
All tuning uses 3-fold stratified CV (search) and 5-fold OOF (selection) on the TRAINING split only. The test set is touched
once, for final reporting. The best model is chosen by cross-validated ROC-AUC (not by test score).
"""
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocessing import (SEED, add_engineered_features, build_preprocessor, get_feature_lists,
                           get_split, leakage_audit, load_data, select_features, ID_COL)
from evaluate import (MET_DIR, recall_target_threshold, compute_metrics, plot_calibration, plot_confusion_matrices,
                      plot_model_comparison, plot_pr, plot_risk_levels, plot_roc)
from retention import assign_risk_level
import explain

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


def build_models(numeric, categorical):
    lr = Pipeline([("prep", build_preprocessor(numeric, categorical, scale=True)),
                   ("clf", LogisticRegression(max_iter=3000, random_state=SEED))])
    rf = Pipeline([("prep", build_preprocessor(numeric, categorical, scale=False)),
                   ("clf", RandomForestClassifier(random_state=SEED, n_jobs=1))])
    xgb = Pipeline([("prep", build_preprocessor(numeric, categorical, scale=False)),
                    ("clf", XGBClassifier(random_state=SEED, n_jobs=1, eval_metric="logloss", tree_method="hist"))])
    return lr, rf, xgb


RF_PARAMS = {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [6, 8, 10, 14],
             "clf__min_samples_leaf": [5, 10, 20, 40], "clf__max_features": ["sqrt", 0.3, 0.5]}
XGB_PARAMS = {"clf__n_estimators": [200, 300, 500], "clf__max_depth": [2, 3, 4, 5],
              "clf__learning_rate": [0.02, 0.05, 0.1], "clf__subsample": [0.7, 0.85, 1.0],
              "clf__colsample_bytree": [0.6, 0.8, 1.0], "clf__min_child_weight": [1, 5, 10]}


def main():
    t0 = time.time()
    MODEL_DIR.mkdir(exist_ok=True); MET_DIR.mkdir(parents=True, exist_ok=True)
    df = add_engineered_features(load_data())
    X_train, X_test, y_train, y_test = get_split(df)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}  | churn rate train={y_train.mean():.3f} test={y_test.mean():.3f}")

    # ---- split bookkeeping (used by predict/dashboard to flag in-sample customers)
    pd.concat([pd.DataFrame({"customer_id": X_train.index, "split": "train"}),
               pd.DataFrame({"customer_id": X_test.index, "split": "test"})]).to_csv(MET_DIR / "split.csv", index=False)

    # ---- leakage audit
    audit, bad_names = leakage_audit(X_train, y_train)
    audit.to_csv(MET_DIR / "leakage_audit.csv", index=False)
    print(f"Leakage audit: forbidden-name columns={bad_names or 'none'}; "
          f"features with single-feature AUC>=0.90: {int(audit['leakage_flag'].sum())}; "
          f"strongest single feature = {audit.iloc[0]['feature']} (AUC {audit.iloc[0]['single_feature_auc']:.3f})")

    # ---- feature selection (train only)
    numeric, categorical = get_feature_lists(X_train)
    numeric, dropped = select_features(X_train, numeric, 0.95)
    features = numeric + categorical
    print(f"Feature selection: dropped {dropped or 'nothing'} (|r|>0.95); using {len(features)} features")
    X_train, X_test = X_train[features], X_test[features]

    lr, rf, xgb = build_models(numeric, categorical)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    fitted, best_params = {}, {}
    print("Fitting Logistic Regression ...")
    fitted["Logistic Regression"] = lr.fit(X_train, y_train)
    for name, est, grid, n_iter in [("Random Forest", rf, RF_PARAMS, 8), ("XGBoost", xgb, XGB_PARAMS, 15)]:
        print(f"Tuning {name} (RandomizedSearchCV, {n_iter} candidates x 3 folds) ...")
        s = RandomizedSearchCV(est, grid, n_iter=n_iter, scoring="roc_auc", cv=StratifiedKFold(3, shuffle=True, random_state=SEED), random_state=SEED,
                               n_jobs=1, refit=True)
        s.fit(X_train, y_train)
        fitted[name] = s.best_estimator_
        best_params[name] = {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v) for k, v in s.best_params_.items()}
        pd.DataFrame(s.cv_results_).sort_values("rank_test_score").drop(columns=["params"]).head(10).to_csv(
            MET_DIR / f"tuning_{name.lower().replace(' ', '_')}.csv", index=False)
        print(f"   best CV ROC-AUC = {s.best_score_:.4f}  params = {best_params[name]}")
    json.dump(best_params, open(MET_DIR / "best_params.json", "w"), indent=2)

    # ---- out-of-fold predictions on TRAIN: CV AUC + recall-oriented threshold (no test leakage)
    rows, rows_t, probas, thresholds, cv_auc = [], [], {}, {}, {}
    for name, model in fitted.items():
        oof = cross_val_predict(clone(model), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        cv_auc[name] = roc_auc_score(y_train, oof)
        thresholds[name] = recall_target_threshold(y_train, oof)
        probas[name] = model.predict_proba(X_test)[:, 1]
        rows.append({"model": name, "cv_roc_auc_train": cv_auc[name], **compute_metrics(y_test, probas[name], 0.5)})
        rows_t.append({"model": name, **compute_metrics(y_test, probas[name], thresholds[name])})
    table = pd.DataFrame(rows).set_index("model")
    table_t = pd.DataFrame(rows_t).set_index("model")
    table.to_csv(MET_DIR / "model_comparison.csv"); table_t.to_csv(MET_DIR / "model_comparison_tuned_threshold.csv")

    best_name = max(cv_auc, key=cv_auc.get)
    print("\n=== TEST-SET RESULTS (threshold = 0.5) ===")
    print(table.round(4).to_string())
    print("\n=== TEST-SET RESULTS (recall >= 70% threshold chosen on training OOF predictions) ===")
    print(table_t.round(4).to_string())
    print(f"\nBest model by cross-validated ROC-AUC: {best_name}")

    # ---- plots
    plot_confusion_matrices(y_test, probas, {k: 0.5 for k in probas})
    plot_confusion_matrices(y_test, probas, thresholds, "confusion_matrices_tuned_threshold.png",
                            "Confusion Matrices - recall >= 70% thresholds (chosen on training OOF)")
    plot_roc(y_test, probas); plot_pr(y_test, probas); plot_model_comparison(table); plot_calibration(y_test, probas)
    plot_risk_levels(pd.Series(assign_risk_level(probas[best_name])))

    # ---- save models + config
    for name, model in fitted.items():
        joblib.dump(model, MODEL_DIR / f"{name.lower().replace(' ', '_')}.pkl")
    joblib.dump(fitted[best_name], MODEL_DIR / "best_model.pkl")
    json.dump({"best_model": best_name, "features": features, "numeric": numeric, "categorical": categorical,
               "dropped_by_correlation": dropped, "decision_threshold": thresholds[best_name],
               "thresholds": thresholds, "cv_roc_auc": cv_auc, "seed": SEED},
              open(MODEL_DIR / "model_config.json", "w"), indent=2)
    json.dump({"test_metrics_0.5": table.to_dict("index"), "test_metrics_tuned": table_t.to_dict("index"),
               "best_model": best_name, "n_train": len(X_train), "n_test": len(X_test)},
              open(MET_DIR / "metrics.json", "w"), indent=2, default=float)

    # ---- explainability
    explain.run_explainability(fitted[best_name])
    print(f"\nDone in {time.time() - t0:.0f}s. Models -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
