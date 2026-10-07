"""Builds notebooks/telecom_churn_analysis.ipynb (29 sections) with nbformat.
Execute it afterwards:
    jupyter nbconvert --to notebook --execute --inplace notebooks/telecom_churn_analysis.ipynb
Prerequisite: run `python src/train_models.py` and `python src/predict.py` once (the notebook reuses their saved outputs)."""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = nb.cells
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t.strip()))
code = lambda t: cells.append(nbf.v4.new_code_cell(t.strip()))
n = [0]


def h(title):
    n[0] += 1
    return f"## {n[0]}. {title}\n\n"


md("# Telecom Customer Churn Prediction & Retention Intelligence\n*Classification - Explainable AI - Customer Risk Scoring - Retention Strategy*")

md(h("Introduction") + """Customer churn - customers leaving for a competitor - is one of the costliest problems in telecom, because
acquiring a new subscriber is typically several times more expensive than retaining an existing one. This notebook builds an
end-to-end system that answers three questions:

* **WHO** is likely to churn? (classification + calibrated churn probability + risk level)
* **WHY** are they likely to churn? (SHAP explainability)
* **WHAT** should the company do about it? (rule-based retention layer + customer health score)""")

md(h("Problem Statement") + """> Build a classification-based machine learning model to predict whether a customer is likely to churn based on their
demographic details, usage patterns, billing information and service feedback.""")

md(h("Objectives") + """1. Generate a realistic dataset (20,000 customers, 37 predictors, ~30% churn).
2. Clean, impute and preprocess without data leakage (scikit-learn `Pipeline` + `ColumnTransformer`).
3. Explore the data and engineer behavioural-trend features.
4. Train and tune Logistic Regression, Random Forest and XGBoost; compare them with imbalance-aware metrics.
5. Output churn probability and LOW / MEDIUM / HIGH / CRITICAL risk, explain it with SHAP, and attach rule-based retention actions.""")

md(h("Dataset Description") + """The dataset is **synthetic** (seed 42) because no public dataset has all required feature groups. Churn is generated from a
latent logistic risk mechanism (contract, tenure, satisfaction, complaints, bill shock, network quality, usage decline ...) plus
unobserved noise, so the relationships are realistic but the problem is **not** perfectly separable. The latent risk is never saved.

| Group | Features |
|---|---|
| Demographics | age, gender, senior_citizen, city_tier |
| Account | tenure_months, contract_type, plan_type, number_of_lines, device_type |
| Usage | data_usage_gb, call_minutes, sms_count, international_minutes, roaming_usage, five_g_usage_percentage |
| Billing | monthly_charges, total_charges, average_bill_last_3_months, late_payment_count, payment_method |
| Network | network_drops, outage_minutes, average_download/upload_speed, call_drop_rate |
| Support | support_calls, complaints, average_resolution_time_hours, tickets_reopened |
| Feedback | satisfaction_score (1-10), nps_score (-100..100), negative_feedback_count |
| Behavioural change (3 months) | data_usage_change_3m, call_usage_change_3m, bill_change_3m, complaint_change_3m, satisfaction_change_3m |""")

md(h("Import Libraries"))
code("""import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt, seaborn as sns
from IPython.display import Image, display, Markdown
pd.set_option("display.width", 200, "display.max_columns", 60)
sns.set_theme(style="whitegrid")
import preprocessing as prep, evaluate as ev, explain, retention, eda, generate_data
FIG = ROOT / "outputs" / "figures"; MET = ROOT / "outputs" / "metrics"
show = lambda name, w=800: display(Image(filename=str(FIG / name), width=w))""")

md(h("Data Generation / Loading") + "`generate_data.py` writes `data/telecom_churn.csv`. We generate it if it does not exist yet.")
code("""if not prep.DATA_PATH.exists():
    generate_data.main()
df = prep.load_data()
df.shape""")

md(h("Data Inspection"))
code("""display(df.head())
display(df.dtypes.value_counts())
display(df.describe().T.round(2))""")
code("""print("Duplicate customer IDs:", df["customer_id"].duplicated().sum())
print("Churn rate: %.2f%%" % (100 * df["churn"].mean()))""")

md(h("Data Cleaning") + """Checks: duplicate IDs, impossible values (negative usage/charges), ranges of scores. The dataset is generated clean, but a real project must
verify this, so the checks are executed explicitly.""")
code("""checks = {
 "negative charges": (df["monthly_charges"] < 0).sum(),
 "satisfaction outside 1-10": ((df["satisfaction_score"] < 1) | (df["satisfaction_score"] > 10)).sum(),
 "NPS outside -100..100": ((df["nps_score"] < -100) | (df["nps_score"] > 100)).sum(),
 "negative tenure": (df["tenure_months"] <= 0).sum(),
}
pd.Series(checks, name="violations")""")

md(h("Missing Value Analysis") + """Missing values were injected into survey (satisfaction, NPS), international/roaming usage and speed columns. We do **not** fill them with zero:
zero satisfaction would be an impossible value and zero speed would mean "no service". The pipeline instead uses **median imputation plus a
missing-indicator flag** (non-response itself can carry signal) - fitted on training data only.""")
code("""miss = df.isna().sum(); miss = miss[miss > 0]
display(pd.DataFrame({"missing": miss, "pct": (100 * miss / len(df)).round(2)}))
show("eda_11_missing_values.png", 600)""")

md(h("Exploratory Data Analysis") + "All plots are produced by `src/eda.py`. Below we show them with brief interpretation.")
code("""eda.make_all(df)
show("eda_01_churn_distribution.png", 450)""")
md("**Interpretation:** roughly 3 in 10 customers churn - an imbalanced (but not extreme) problem, so accuracy alone is misleading (see section 20).")

md(h("Visualization"))
code("""for f in ["eda_02_churn_by_contract.png", "eda_03_churn_by_plan.png", "eda_04_churn_by_tenure.png"]:
    show(f, 560)""")
md("**Interpretation:** churn falls steeply with longer contracts and longer tenure - new, month-to-month customers are the most exposed. Plan type has a smaller effect.")
code("""for f in ["eda_05_satisfaction_vs_churn.png", "eda_07_complaints_vs_churn.png", "eda_08_network_drops_vs_churn.png"]:
    show(f, 560)""")
md("**Interpretation:** churn rises as satisfaction falls and as complaints / network drops increase. These service-experience variables are actionable for the telecom operator.")
code("""for f in ["eda_06_monthly_charges_vs_churn.png", "eda_09_usage_change_vs_churn.png"]:
    show(f, 800)""")
md("**Interpretation:** declining data/call usage over the last three months is a leading churn signal - a customer who is already disengaging. A static snapshot would miss this, which is why trend features are included.")
code("""show("eda_10_correlation_heatmap.png", 900); show("eda_10b_correlation_with_churn.png", 520)""")
md("**Interpretation:** no single numeric feature is highly correlated with churn (no leakage), while several feature groups are correlated with each other (e.g. satisfaction vs NPS, network metrics vs complaints), which motivates regularised/ensemble models and a correlation filter.")

md(h("Feature Engineering") + """Static attributes alone are not enough: *Customer A* (satisfaction 8, 1 complaint, stable usage) and *Customer B* (satisfaction fell 8 -> 4, complaints rising, usage declining, bill up)
may look similar in a snapshot but have very different risk. Beyond the five provided 3-month change features, we add fixed-formula features (no learned statistics, hence no leakage):
`spend_per_line`, `network_issue_index`, `support_intensity`, `is_new_customer`, `bill_shock`, and `n_negative_trends` (number of simultaneous deterioration signals).""")
code("""dfe = prep.add_engineered_features(df)
dfe[prep.ENGINEERED].describe().T.round(2)""")

md(h("Feature Selection") + """Two safeguards, both computed on the **training split only**: (1) a **leakage audit** - single-feature ROC-AUC; any feature with AUC >= 0.90 (or a forbidden name such as cancellation/termination/retention-offer) would be flagged;
(2) a **correlation filter** dropping one feature of any pair with |r| > 0.95.""")
code("""X_train, X_test, y_train, y_test = prep.get_split(dfe)
audit, bad = prep.leakage_audit(X_train, y_train)
print("Forbidden-name columns:", bad or "none", "| flagged features:", int(audit.leakage_flag.sum()))
display(audit.head(8).round(3))
numeric, categorical = prep.get_feature_lists(X_train)
numeric, dropped = prep.select_features(X_train, numeric)
print("Dropped by correlation filter:", dropped)
features = numeric + categorical; len(features)""")
md("**Interpretation:** the strongest single feature reaches only a modest AUC, confirming that no predictor leaks the target. Only the redundant `average_bill_last_3_months` (almost identical to `monthly_charges`) is removed.")

md(h("Preprocessing") + """A `ColumnTransformer` handles numeric columns (median imputation + missing indicators [+ StandardScaler for the linear model]) and categorical columns
(most-frequent imputation + one-hot encoding with `handle_unknown='ignore'`). It sits **inside** each model `Pipeline`, so it is fitted only on training folds.
Tree models do not need scaling, so scaling is applied to Logistic Regression only.""")
code("""pre = prep.build_preprocessor(numeric, categorical, scale=True)
pre.fit(X_train[features])
print("Transformed feature count:", len(pre.get_feature_names_out()))""")

md(h("Train-Test Split") + """Stratified 80/20 split (`random_state=42`) keeps the 30% churn rate identical in both parts.
*Production note:* in a live telecom setting a **temporal split** (train on older months, test on the most recent month) is preferable because it mimics real deployment and exposes concept drift; a random split can be optimistic.""")
code("""X_train, X_test = X_train[features], X_test[features]
print(X_train.shape, X_test.shape, "| churn:", round(y_train.mean(), 3), round(y_test.mean(), 3))""")

md(h("Logistic Regression") + "Interpretable baseline: coefficients show direction and strength of every effect on the log-odds of churn.")
code("""from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
lr = Pipeline([("prep", prep.build_preprocessor(numeric, categorical, True)), ("clf", LogisticRegression(max_iter=3000, random_state=42))]).fit(X_train, y_train)
print("Quick test ROC-AUC:", round(ev.compute_metrics(y_test, lr.predict_proba(X_test)[:, 1])["roc_auc"], 4))""")

md(h("Random Forest") + "Captures non-linearities and interactions (untuned quick fit here; the tuned version is in section 19).")
code("""from sklearn.ensemble import RandomForestClassifier
rf = Pipeline([("prep", prep.build_preprocessor(numeric, categorical, False)), ("clf", RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1))]).fit(X_train, y_train)
print("Quick test ROC-AUC:", round(ev.compute_metrics(y_test, rf.predict_proba(X_test)[:, 1])["roc_auc"], 4))""")

md(h("XGBoost") + "Gradient-boosted trees - the advanced model (untuned quick fit here).")
code("""from xgboost import XGBClassifier
xgb = Pipeline([("prep", prep.build_preprocessor(numeric, categorical, False)), ("clf", XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42, eval_metric="logloss", n_jobs=1))]).fit(X_train, y_train)
print("Quick test ROC-AUC:", round(ev.compute_metrics(y_test, xgb.predict_proba(X_test)[:, 1])["roc_auc"], 4))""")

md(h("Hyperparameter Tuning") + """`RandomizedSearchCV` on the **training set only** (3-fold stratified CV, scoring = ROC-AUC) for Random Forest (`n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`)
and XGBoost (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`). The search was run by `train_models.py`; results are loaded here.""")
code("""print(open(MET / "best_params.json").read())
display(pd.read_csv(MET / "tuning_xgboost.csv").head(5).filter(regex="param_|mean_test|std_test|rank").round(4))""")

md(h("Model Evaluation") + """**Why accuracy is not enough.** If 90% of customers stay, predicting "stay" for everyone gives 90% accuracy yet catches zero churners. Even at 30% churn, accuracy hides the trade-off between
**precision** (of the customers we flag, how many really churn - wasted offers) and **recall** (of the real churners, how many we catch - lost revenue). **F1** balances them; **ROC-AUC** measures ranking quality across all thresholds;
**PR-AUC** is more informative when the positive class is rare. Because missing a churner is usually costlier than a wasted offer, we give recall special attention.""")
code("""cfg = json.load(open(ROOT / "models" / "model_config.json"))
models = {n: joblib.load(ROOT / "models" / f"{f}.pkl") for n, f in [("Logistic Regression", "logistic_regression"), ("Random Forest", "random_forest"), ("XGBoost", "xgboost")]}
probas = {n: m.predict_proba(X_test[cfg["features"]])[:, 1] for n, m in models.items()}
table = pd.DataFrame({n: ev.compute_metrics(y_test, p) for n, p in probas.items()}).T
display(table.round(4))""")
code("""show("confusion_matrices.png", 1000); show("roc_curves.png", 520); show("precision_recall_curves.png", 520)""")
md("**Interpretation:** the confusion matrices at the default 0.5 threshold show high precision but modest recall - many churners are missed. Section 25 shows how a business-driven threshold changes this.")

md(h("Model Comparison") + """The best model is selected using **cross-validated ROC-AUC on the training data**, not the test score, to avoid selecting on the test set.""")
code("""display(pd.read_csv(MET / "model_comparison.csv", index_col=0).round(4))
print("Selected best model:", cfg["best_model"], "| CV AUC:", {k: round(v, 4) for k, v in cfg["cv_roc_auc"].items()})
show("model_comparison.png", 800); show("calibration_curve.png", 480)""")
md("**Interpretation:** the three models score very close to one another. The data contain mostly additive effects with a few interactions, so a well-regularised linear model is highly competitive - an honest finding, not a flaw. Calibration matters because probabilities are mapped to risk levels.")

md(h("Feature Importance") + "Global importance = mean absolute SHAP value of the best model, aggregated from one-hot columns back to the original feature.")
code("""imp = pd.read_csv(MET / "feature_importance.csv", index_col=0).iloc[:, 0]
for f, v in imp.head(12).items():
    print(f"{f:28s} {'#' * int(40 * v / imp.iloc[0])} {v:.3f}")
show("feature_importance.png", 650)""")

md(h("SHAP Explainability") + """SHAP assigns each feature a contribution to a customer's churn log-odds. The summary plot shows the global picture (colour = feature value, position = push towards churn);
the individual plot explains one customer. The explanations are computed from the trained model - nothing is hand-written.""")
code("""show("shap_summary.png", 750); show("shap_individual_example.png", 650)
ex = json.load(open(MET / "shap_individual_example.json"))
print(f"Customer {ex['customer_id']} - churn probability {ex['churn_probability']:.0%}\\nMain contributing factors:")
for i, f in enumerate(ex["risk_factors"], 1): print(f"  {i}. {f['label']} (value: {f['value']}, SHAP {f['shap']:+.2f})")""")

md(h("Churn Probability") + """Final output per customer: `customer_id`, `churn_probability`, `risk_level` (plus factors and action). Produced by `python src/predict.py`.
Note: customers in the *training* split are in-sample; use rows with `split == "test"` for honest evaluation.""")
code("""pred = pd.read_csv(ROOT / "outputs" / "predictions" / "churn_predictions.csv")
display(pred.sort_values("churn_probability", ascending=False).head(5)[["customer_id", "churn_probability", "risk_level", "customer_health_score", "top_risk_factors"]])""")

md(h("Customer Risk Segmentation") + """Business thresholds (adjustable in `retention.py`): **0-30% LOW, 30-60% MEDIUM, 60-80% HIGH, 80-100% CRITICAL**.""")
code("""t = pred[pred.split == "test"]
seg = t.groupby("risk_level").agg(customers=("customer_id", "size"), avg_prob=("churn_probability", "mean"), actual_churn_rate=("actual_churn", "mean")).reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"])
display(seg.round(3)); show("risk_level_distribution.png", 450)
print("Operating threshold giving >= 70% recall (chosen on training OOF):", cfg["decision_threshold"])
show("confusion_matrices_tuned_threshold.png", 1000)
display(pd.read_csv(MET / "model_comparison_tuned_threshold.csv", index_col=0).round(4))""")
md("**Interpretation:** the observed churn rate rises monotonically across the risk levels, which shows the probabilities are meaningful, not just rankings. Lowering the decision threshold trades precision for recall - the right point depends on the cost of a retention offer versus a lost customer.")

md(h("Retention Recommendation") + """The retention layer in `retention.py` is **rule-based** (e.g. many network problems -> network investigation; bill rise + high spend -> loyalty plan; low satisfaction + many support calls -> priority service;
late payments -> flexible payment). The ML model supplies only the probability and the SHAP factors used to *prioritise* the rules - it does not generate the action text. We also compute a
**customer_health_score** (0-100): a weighted, fixed-bound business index of satisfaction, network quality, complaint burden, payment stability and engagement trend (never uses the churn label).""")
code("""display(pred[pred.risk_level.isin(["HIGH", "CRITICAL"])].head(5)[["customer_id", "churn_probability", "risk_level", "top_risk_factors", "recommended_action"]])
print(pred.groupby("risk_level")["customer_health_score"].mean().round(1))""")

md(h("Business Interpretation") + """* Churn probability lets the retention team focus a limited budget on the customers where it matters most (rank by probability x customer value).
* SHAP factors tell agents *what to talk about* with each customer; aggregated factors show systemic issues (e.g. a network problem in an area).
* Actionable drivers (satisfaction, complaints, bill shock, network quality) point to operational fixes - not just marketing offers.
* Month-to-month, short-tenure customers are structurally riskier: converting them to longer contracts is a strategic lever.""")

md(h("Conclusion") + """We built a leakage-free, reproducible pipeline that predicts churn, explains it, and links it to retention actions. Model performance on this synthetic dataset is moderate and realistic (see metrics above) and the three
models are close, which is a credible outcome because the generator contains irreducible noise. The system answers WHO, WHY and WHAT.""")

md(h("Future Scope") + """* **Uplift / treatment-effect modelling**: ordinary churn prediction answers *who is likely to churn*; an advanced system should answer *who is likely to churn AND can be saved by an intervention*.
  A customer with 90% churn risk who will leave regardless is a worse target than a 70%-risk customer who responds strongly to an offer.
* Temporal / sequence models on monthly histories, real-time scoring, customer-lifetime-value weighting, causal inference, personalised offers, CRM integration, validation on real telecom data, fairness testing and drift monitoring.""")

nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = Path(__file__).resolve().parent / "telecom_churn_analysis.ipynb"
nbf.write(nb, out)
print("Wrote", out, "with", len(cells), "cells")
