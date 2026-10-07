"""Generates PROJECT_REPORT.md and README.md from the ACTUAL saved results (no hand-typed metrics).
Run after train_models.py and predict.py:   python src/build_report.py"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "outputs" / "metrics"


def table(df, floatfmt=3):
    df = df.copy()
    cols = [df.index.name or "model"] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, r in df.iterrows():
        vals = [f"{v:.{floatfmt}f}" if isinstance(v, float) else str(v) for v in r.values]
        lines.append("| " + " | ".join([str(idx)] + vals) + " |")
    return "\n".join(lines)


def fill(text, d):
    for k, v in d.items():
        text = text.replace("{{" + k + "}}", str(v))
    return text


def main():
    data = pd.read_csv(ROOT / "data" / "telecom_churn.csv")
    cfg = json.load(open(ROOT / "models" / "model_config.json"))
    comp = pd.read_csv(MET / "model_comparison.csv", index_col=0)
    tuned = pd.read_csv(MET / "model_comparison_tuned_threshold.csv", index_col=0)
    imp = pd.read_csv(MET / "feature_importance.csv", index_col=0).iloc[:, 0]
    audit = pd.read_csv(MET / "leakage_audit.csv")
    params = json.load(open(MET / "best_params.json"))
    ex = json.load(open(MET / "shap_individual_example.json"))
    pred = pd.read_csv(ROOT / "outputs" / "predictions" / "churn_predictions.csv")
    test = pred[pred["split"] == "test"]
    seg = (test.groupby("risk_level").agg(customers=("customer_id", "size"), mean_predicted_probability=("churn_probability", "mean"),
                                          observed_churn_rate=("actual_churn", "mean"))
           .reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"]))
    seg.index.name = "risk_level"
    best = cfg["best_model"]
    miss = (data.isna().mean() * 100)
    miss = miss[miss > 0].round(1)
    ct = data.groupby("contract_type")["churn"].mean()
    pt = data.groupby("plan_type")["churn"].mean()
    imp_rank = {f: i + 1 for i, f in enumerate(imp.index)}
    top = imp.head(8)
    top_txt = ", ".join(f"`{f}` ({v:.3f})" for f, v in top.items())
    maxbar = 30
    bars = "\n".join(f"{f:26s} {'█' * max(1, int(maxbar * v / imp.iloc[0]))} {v:.3f}" for f, v in imp.head(12).items())
    ex_factors = "\n".join(f"{i}. {f['label']} (value: {f['value']}, SHAP {f['shap']:+.2f})" for i, f in enumerate(ex["risk_factors"], 1))
    r = comp.loc[best]; rt = tuned.loc[best]
    d = {
        "N": f"{len(data):,}", "CHURN": f"{data['churn'].mean():.1%}", "NFEAT": len(data.columns) - 2,
        "BEST": best, "THR": f"{cfg['decision_threshold']:.2f}",
        "MISS": ", ".join(f"`{k}` {v}%" for k, v in miss.items()),
        "COMP": table(comp.round(4)), "TUNED": table(tuned.round(4)), "SEG": table(seg.round(3)),
        "TOPTXT": top_txt, "BARS": bars, "EXCID": ex["customer_id"], "EXPROB": f"{ex['churn_probability']:.0%}", "EXFACT": ex_factors,
        "CT": ", ".join(f"{k}: {v:.1%}" for k, v in ct.items()), "PT": ", ".join(f"{k}: {v:.1%}" for k, v in pt.items()),
        "AUC": f"{r['roc_auc']:.3f}", "REC": f"{r['recall']:.3f}", "PRE": f"{r['precision']:.3f}", "F1": f"{r['f1']:.3f}", "ACC": f"{r['accuracy']:.3f}",
        "REC_T": f"{rt['recall']:.3f}", "PRE_T": f"{rt['precision']:.3f}", "ACC_T": f"{rt['accuracy']:.3f}", "F1_T": f"{rt['f1']:.3f}",
        "PRAUC": f"{r['pr_auc']:.3f}", "MAXAUC": f"{audit.iloc[0]['strength']:.3f}", "MAXFEAT": audit.iloc[0]["feature"],
        "DROPPED": ", ".join(cfg["dropped_by_correlation"]) or "none",
        "PARAMS": "\n".join(f"- **{k}**: `{v}`" for k, v in params.items()),
        "GENDER_RANK": imp_rank.get("gender", "n/a"), "AGE_RANK": imp_rank.get("age", "n/a"), "CITY_RANK": imp_rank.get("city_tier", "n/a"),
        "SENIOR_RANK": imp_rank.get("senior_citizen", "n/a"), "NIMP": len(imp),
        "CVAUC": ", ".join(f"{k}: {v:.4f}" for k, v in cfg["cv_roc_auc"].items()),
    }
    report = fill(REPORT, d)
    readme = fill(README, d)
    (ROOT / "PROJECT_REPORT.md").write_text(report, encoding="utf-8")
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    print("Wrote PROJECT_REPORT.md and README.md")


REPORT = r"""
# Telecom Customer Churn Prediction & Retention Intelligence

*A classification-based machine-learning system that answers: **who** will churn, **why**, and **what to do about it**.*

---

## Abstract

Customer churn is a major revenue risk for telecom operators. This project builds an end-to-end, leakage-free machine-learning system on a synthetic telecom dataset of {{N}} customers and {{NFEAT}} predictive features (demographics, account, usage, billing, network quality, support and feedback, plus 3-month behavioural-change features). Three classifiers - Logistic Regression, Random Forest and XGBoost - are trained inside scikit-learn pipelines, tuned with cross-validation on the training set only, and compared with imbalance-aware metrics. The selected model ({{BEST}}) reaches a test ROC-AUC of {{AUC}}, PR-AUC {{PRAUC}} and, at a business-chosen decision threshold, recall {{REC_T}}. Predictions are converted into churn probabilities and LOW / MEDIUM / HIGH / CRITICAL risk levels, explained with SHAP, and linked to a transparent rule-based retention layer and a customer health score. A Streamlit dashboard exposes the results. All numbers in this report are produced by the code (`src/build_report.py` reads the saved metrics).

## Introduction

Telecom is a saturated, highly competitive market: customers can switch operators easily and acquiring a new subscriber costs substantially more than retaining an existing one. Churn prediction lets an operator spend limited retention budget on the customers most likely to leave. This project goes beyond a binary "churn / no churn" label: it produces calibrated probabilities, explanations and recommended actions.

## Problem Statement

> Customer churn is a critical challenge in the telecom industry, where customers discontinue their services and move to competitors. Reducing churn can significantly improve revenue and customer loyalty. Build a classification-based machine learning model to predict whether a customer is likely to churn based on their demographic details, usage patterns, billing information, and service feedback.

## Objectives

1. Generate a realistic dataset with realistic feature relationships and a ~30% churn rate.
2. Clean the data, handle missing values properly and build leakage-free preprocessing pipelines.
3. Perform exploratory analysis and engineer behavioural-trend features.
4. Train, tune and compare Logistic Regression, Random Forest and XGBoost.
5. Output churn probability and risk level for each customer.
6. Explain predictions with SHAP and attach rule-based retention actions.
7. Provide a customer health score, a dashboard and full documentation.

## Existing Problem

Basic churn notebooks usually (a) use a static snapshot of customer attributes, (b) report accuracy only, (c) output a hard 0/1 label and (d) stop there. They ignore behavioural *change* (falling satisfaction, rising complaints, declining usage), ignore class imbalance, give no explanation of the prediction and no guidance on what to do next. Imputation is often done on the whole dataset before splitting, which leaks information.

## Proposed Solution

* **Behavioural features** that capture deterioration over the last 3 months.
* **Pipeline-based preprocessing** fitted on training data only.
* **Model comparison** with precision / recall / F1 / ROC-AUC / PR-AUC, calibration and threshold analysis.
* **SHAP explanations** aggregated to readable feature names.
* A **separate rule-based retention layer** and an interpretable **health score**, clearly distinguished from the ML prediction.

## Dataset

The dataset is **synthetic** (`src/generate_data.py`, seed 42) because no public dataset has all required feature groups. Churn is drawn from a latent logistic risk mechanism combining many features plus unobserved noise, and the intercept is calibrated so the churn rate is about 30%. The latent risk is never saved. Final dataset: **{{N}} customers, churn rate {{CHURN}}**.

| Group | Features | Notes |
|---|---|---|
| Demographics | age, gender, senior_citizen, city_tier | Possible fairness concern (see Ethics) |
| Account | tenure_months, contract_type, plan_type, number_of_lines, device_type | Longer contracts -> longer tenure |
| Usage | data_usage_gb, call_minutes, sms_count, international_minutes, roaming_usage, five_g_usage_percentage | Scaled by plan and number of lines |
| Billing | monthly_charges, total_charges, average_bill_last_3_months, late_payment_count, payment_method | Premium plans cost more; cash payers pay late more often |
| Network | network_drops, outage_minutes, average_download_speed, average_upload_speed, call_drop_rate | Driven by a hidden network-quality factor and city tier |
| Support | support_calls, complaints, average_resolution_time_hours, tickets_reopened | More frequent with poor network |
| Feedback | satisfaction_score (1-10), nps_score (-100..100), negative_feedback_count | Lower with poor network, complaints, late bills |
| Behavioural change (3 months) | data_usage_change_3m, call_usage_change_3m, bill_change_3m, complaint_change_3m, satisfaction_change_3m | Share a hidden "deterioration" factor |
| Target | `churn` (0 stayed, 1 churned) | Generated from latent risk + noise |

**Missing values** were injected intentionally: {{MISS}}.

**Realism checks** (computed from the data): churn by contract type - {{CT}}; churn by plan type - {{PT}}.

## Methodology

### Preprocessing
* Numeric: `SimpleImputer(strategy="median", add_indicator=True)` (the indicator keeps the information that a value was missing), then `StandardScaler` for the linear model only.
* Categorical: most-frequent imputation, then one-hot encoding (`handle_unknown="ignore"`).
* Implemented with `ColumnTransformer` + `Pipeline`; the preprocessor is part of every model pipeline, so it is fitted **only on training data** (and only on training folds inside cross-validation).
* Missing values are *not* replaced with zero because zero satisfaction or zero speed would be impossible/misleading values.

### Exploratory Data Analysis
Twelve+ figures (`outputs/figures/eda_*.png`): churn distribution, churn by contract, plan, tenure, satisfaction, monthly charges, complaints, network drops, usage change, correlation heatmap, correlation with churn, missing values. Key patterns: churn falls strongly with contract length ({{CT}}) and tenure; plan type has only a small effect ({{PT}}); churn rises with complaints, network drops and declining usage.

### Feature Engineering
Besides the five provided change features, fixed-formula features were added (no statistics learned from the data): `spend_per_line`, `network_issue_index`, `support_intensity`, `is_new_customer`, `bill_shock`, `n_negative_trends`. Static attributes alone are not enough: a customer with satisfaction 8, one complaint and stable usage is low risk, while a customer whose satisfaction fell from 8 to 4 with rising complaints, falling usage and a higher bill is high risk even if the snapshot looks similar.

### Feature Selection
A correlation filter fitted on the training split dropped: {{DROPPED}} (|r| > 0.95). A leakage audit (below) was run as well.

### Train / Test Split
Stratified 80/20 split with `random_state=42` (16,000 training, 4,000 test customers; churn rate preserved). In a **real production setting a temporal split is preferable**: train on earlier months and test on the most recent month, which matches how the model will be used and exposes drift; random splits can be optimistic.

### Model Selection and Tuning
* **Logistic Regression** - interpretable baseline.
* **Random Forest** - non-linear model, tuned with `RandomizedSearchCV` (n_estimators, max_depth, min_samples_leaf, max_features).
* **XGBoost** - gradient boosting, tuned with `RandomizedSearchCV` (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight).
* Tuning: 3-fold stratified CV on the training set only, scoring ROC-AUC. Best parameters found:

{{PARAMS}}

* **The best model is chosen by cross-validated ROC-AUC on the training data** (out-of-fold predictions, 5 folds), never by test performance. CV ROC-AUC: {{CVAUC}}.

### Evaluation
Accuracy, precision, recall, F1, ROC-AUC, PR-AUC and Brier score on the untouched test set, plus confusion matrices, ROC and precision-recall curves and a calibration curve.

**Why accuracy alone is insufficient.** Churn is an imbalanced problem. If 90% of customers stay, a model that predicts "stay" for everybody has 90% accuracy and zero business value. Precision shows how many flagged customers really churn (wasted offers otherwise), recall shows how many churners are caught (lost revenue otherwise), F1 balances both, ROC-AUC measures ranking quality across thresholds and PR-AUC focuses on the positive class. Recall receives special attention because missing a real churner is usually costlier than a wasted retention offer.

### Data Leakage Inspection
* Forbidden names (cancellation request, termination date, retention-offer outcome, latent risk) are absent from the predictors; the audit checks for them automatically.
* All predictors represent information available **before** the churn event.
* The latent churn score is not in the dataset.
* Single-feature ROC-AUC audit on the training set: the strongest single feature is `{{MAXFEAT}}` with a separation strength of only {{MAXAUC}} (0.5 = none, 1.0 = perfect); no feature reaches the 0.90 alarm level.
* Imputation, scaling, encoding, correlation filtering and threshold selection all use training data only.
* The model reaches a realistic ROC-AUC of {{AUC}} instead of a suspicious ~1.0.

## System Architecture

![Pipeline flow diagram](outputs/figures/flow_diagram.png)

Raw data -> inspection -> missing-value handling -> encoding -> scaling -> EDA -> feature engineering -> feature selection -> stratified split -> model training (LR, RF, XGBoost) -> hyperparameter tuning -> evaluation -> best model -> churn probability -> risk classification -> SHAP explainability -> retention recommendation. (In the code, imputation/encoding/scaling are fitted *after* the split inside the pipeline to avoid leakage.)

## Results

All values below were calculated by the program on the held-out test set (4,000 customers).

### Model Comparison (decision threshold = 0.5)

{{COMP}}

![Model comparison](outputs/figures/model_comparison.png)
![ROC curves](outputs/figures/roc_curves.png)
![Precision-recall curves](outputs/figures/precision_recall_curves.png)
![Confusion matrices](outputs/figures/confusion_matrices.png)

**Reading the table.** The three models are very close in ranking quality (ROC-AUC). This is an honest result: the synthetic data are dominated by additive effects (plus a few interactions) and contain irreducible noise, so a well-regularised linear model is competitive with boosted trees. Random Forest is the most conservative (highest precision, lowest recall at 0.5).

### Recall-oriented operating point

At 0.5 the selected model ({{BEST}}) has accuracy {{ACC}}, precision {{PRE}}, recall {{REC}} and F1 {{F1}} - it misses many churners. A business that values recall can lower the threshold. Below, each threshold was chosen on **training out-of-fold predictions** as the highest cut-off that still catches at least 70% of churners (no test data used):

{{TUNED}}

For {{BEST}} the chosen threshold is {{THR}}: recall rises to {{REC_T}} at precision {{PRE_T}} (accuracy {{ACC_T}}, F1 {{F1_T}}). Accuracy falls, which is exactly the trade-off an accuracy-only analysis would hide.

![Confusion matrices at tuned thresholds](outputs/figures/confusion_matrices_tuned_threshold.png)
![Calibration](outputs/figures/calibration_curve.png)

### Churn probability and risk levels

Thresholds (business-oriented and adjustable in `src/retention.py`): 0-30% **LOW**, 30-60% **MEDIUM**, 60-80% **HIGH**, 80-100% **CRITICAL**. Test-set segmentation:

{{SEG}}

The observed churn rate rises across the levels and tracks the mean predicted probability closely, i.e. the probabilities are reasonably calibrated and the segments are meaningful. Output format (`outputs/predictions/churn_predictions.csv`): `customer_id, churn_probability, risk_level` (plus predicted label, health score, SHAP risk factors and recommended action). Rows with `split = train` are in-sample; use `split = test` for honest evaluation.

## Feature Importance

Mean absolute SHAP value of the selected model, aggregated back to the original feature (top 12 shown):

```
{{BARS}}
```

The most influential features are {{TOPTXT}}. Contract type, tenure and satisfaction dominate, followed by behavioural-trend features (change in data usage, change in satisfaction), NPS and the bill change - supporting the argument that trends add information beyond static attributes.

![Feature importance](outputs/figures/feature_importance.png)

## Explainability

SHAP values attribute each prediction to features. The summary plot shows the global direction and size of effects; the individual plot explains one customer. Explanations are computed by SHAP from the trained model, not written by hand. Example (highest-risk customer in the explained sample):

```
Customer {{EXCID}}   Churn probability: {{EXPROB}}
Main contributing factors:
{{EXFACT}}
```

![SHAP summary](outputs/figures/shap_summary.png)
![SHAP individual](outputs/figures/shap_individual_example.png)

## Business Interpretation

* Rank customers by churn probability (ideally multiplied by customer value) so the retention team contacts the most valuable at-risk customers first.
* Use the SHAP factors as talking points for the agent and aggregate them to find systemic problems (for example a network problem in a region).
* Month-to-month, short-tenure customers are structurally riskier; migrating them to longer contracts is a strategic lever.
* Falling satisfaction, rising complaints and declining usage are early-warning signs that can trigger proactive outreach before the customer decides to leave.

## Retention Strategy

The retention layer (`src/retention.py`) is **rule-based and separate from the ML prediction**; the model only supplies the probability and the SHAP factors used to *rank* the rules. Examples: many network drops / high call-drop rate -> priority network support; large bill increase with high spend -> lower-cost or loyalty plan; low satisfaction with many support calls -> priority customer-service call-back; late payments -> flexible payment options; month-to-month with short tenure -> discounted 12-month contract; reopened tickets -> escalation; sharply declining usage -> re-engagement campaign.

**Customer health score** (`customer_health_score`, 0-100): a weighted business index (satisfaction 30%, network quality 20%, complaint burden 20%, payment stability 15%, engagement trend 15%) using fixed, documented normalisation bounds. It is a business analytics score, **not** the ML prediction, and it does not use the churn label.

## Ethics and Fairness

Demographic variables (gender, age, location/city tier, senior-citizen status) can create unfair treatment even when they improve accuracy, and some may be restricted by regulation. They are included here for completeness; their SHAP importance ranks are: gender #{{GENDER_RANK}}, age #{{AGE_RANK}}, city tier #{{CITY_RANK}}, senior citizen #{{SENIOR_RANK}} out of {{NIMP}} features. Note that in the generator gender has no true effect, so any weight the model gives it is noise. In particular the small "Other" gender group (~2%) can receive a spurious coefficient - an example of why fairness testing is needed. A real operator should not assume these variables are appropriate for production and should perform **fairness testing, bias analysis, a privacy review and a regulatory review** before deployment, and should consider removing or constraining sensitive attributes and offering retention incentives consistently.

## Limitations

* **Synthetic data**: relationships are designed by us, so performance does not transfer to a real operator.
* Behavioural trends are simulated; real trend features need monthly history tables.
* No real network telemetry (cell-level congestion, coverage maps), competitor offers or customer value data.
* Random (not temporal) split; real deployments suffer concept drift and need monitoring/retraining.
* Predicting churn does not tell us whom a retention offer will actually save (see Future Scope).
* Demographic bias risk; probabilities are calibrated only on this data distribution.

## Future Scope

* **Uplift / treatment-effect modelling.** Ordinary churn prediction answers *"Who is likely to churn?"*. An advanced system should answer *"Who is likely to churn AND who can actually be saved by an intervention?"* Example: Customer A has 90% churn probability but a retention offer is unlikely to help (they are leaving regardless); Customer B has 70% churn probability and an offer would very likely prevent churn. Customer B may be the better target. This needs data from randomised retention experiments and methods such as two-model (T-learner), class-transformation or causal-forest uplift models; it is presented as future work because it cannot be done correctly with purely observational, synthetic data.
* Real-time churn scoring and streaming features; temporal/sequence models on monthly histories.
* Customer lifetime value to prioritise by expected revenue saved.
* Causal inference and A/B testing of interventions; personalised retention offers; automated CRM integration.
* Validation on real telecom data; fairness audits; drift monitoring.

## Conclusion

The project delivers a complete, reproducible and explainable churn system. The chosen model ({{BEST}}) achieves ROC-AUC {{AUC}} on unseen customers with well-calibrated probabilities; lowering the decision threshold to {{THR}} catches {{REC_T}} of churners at a precision of {{PRE_T}}. SHAP shows that contract, tenure, satisfaction and behavioural trends drive the risk, and the retention layer turns each prediction into a concrete next step. Results on synthetic data are indicative only, but the workflow - leakage-free preprocessing, imbalance-aware evaluation, explainability, and a clear separation between prediction and business rules - is what would be reused on real data.
"""

README = r"""
# Telecom Churn AI - Prediction & Retention Intelligence

An end-to-end machine-learning project that predicts **who** will churn, explains **why** (SHAP) and suggests **what to do** (rule-based retention layer), with a Streamlit dashboard.

## Features
* Synthetic, realistic dataset: {{N}} customers, {{NFEAT}} predictors, churn rate {{CHURN}}, intentional missing values
* Leakage-free scikit-learn pipelines (`ColumnTransformer` + `Pipeline`), median imputation with missing-indicators
* Behavioural-trend features (3-month changes in usage, bill, complaints, satisfaction)
* Logistic Regression, Random Forest, XGBoost with `RandomizedSearchCV`
* Imbalance-aware evaluation, calibration, recall-oriented threshold
* Churn probability + LOW/MEDIUM/HIGH/CRITICAL risk levels
* SHAP global and per-customer explanations
* Rule-based retention actions and a `customer_health_score` (business score, not ML)
* Streamlit dashboard, Jupyter notebook, full report

## Dataset
Synthetic (seed 42), generated by `src/generate_data.py` -> `data/telecom_churn.csv`. Groups: demographics, account, usage, billing, network quality, support, feedback and 3-month behavioural change. Churn is generated from a hidden latent risk plus noise (not exposed as a feature).

## ML Models
Logistic Regression (baseline), Random Forest, XGBoost. Best model by cross-validated ROC-AUC: **{{BEST}}**.

## Architecture
![Flow diagram](outputs/figures/flow_diagram.png)

```
telecom-churn-ai/
├── data/telecom_churn.csv
├── notebooks/telecom_churn_analysis.ipynb   (built by notebooks/build_notebook.py)
├── src/  generate_data.py  preprocessing.py  eda.py  train_models.py  evaluate.py
│         explain.py  predict.py  retention.py  flow_diagram.py  build_report.py
├── models/  logistic_regression.pkl  random_forest.pkl  xgboost.pkl  best_model.pkl  model_config.json
├── outputs/ figures/  predictions/  metrics/
├── app/app.py
├── requirements.txt  README.md  PROJECT_REPORT.md
```

## Installation
```bash
cd telecom-churn-ai
python3 -m venv .venv && source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
```

## Usage
Run from the project root, in this order:
```bash
python src/generate_data.py      # create data/telecom_churn.csv
python src/eda.py                # EDA figures
python src/flow_diagram.py       # flow diagram image
python src/train_models.py       # preprocessing, tuning, evaluation, models, SHAP (takes a few minutes)
python src/predict.py            # churn_predictions.csv (probability, risk level, factors, action)
python src/build_report.py       # regenerate README/PROJECT_REPORT from the saved metrics
streamlit run app/app.py         # dashboard
```
Score a single customer explanation: `python src/predict.py --customer C100123`; score a new CSV: `python src/predict.py --input new_customers.csv`.

### Running the notebook
```bash
python notebooks/build_notebook.py      # (re)creates the .ipynb
jupyter notebook notebooks/telecom_churn_analysis.ipynb
# or execute headless:
jupyter nbconvert --to notebook --execute --inplace notebooks/telecom_churn_analysis.ipynb
```
(Run `train_models.py` and `predict.py` first; the notebook reuses their saved outputs.)

## Results (test set, 4,000 customers, threshold 0.5)

{{COMP}}

At the recall-oriented thresholds (chosen on training out-of-fold predictions):

{{TUNED}}

Risk segmentation (test set):

{{SEG}}

Top features (mean |SHAP|): {{TOPTXT}}.

## Notes
* Thresholds for risk levels are business-oriented and adjustable in `src/retention.py`.
* Retention actions and the health score are rule-based business logic, not ML output.
* Rows with `split = train` in the predictions file are in-sample; evaluate on `split = test`.
* Data are synthetic: do not interpret the numbers as real-world telecom performance.

## Future scope
Uplift modelling ("who can be saved?"), temporal models on monthly histories, real-time scoring, customer lifetime value, causal inference, personalised offers, CRM integration, real telecom data, fairness testing and drift monitoring.
"""

if __name__ == "__main__":
    main()
