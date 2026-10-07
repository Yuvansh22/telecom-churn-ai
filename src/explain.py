"""Explainability: SHAP global + local explanations with a feature-importance fallback.

All explanations are computed from the trained best model. SHAP values are aggregated from
the transformed (one-hot / missing-indicator) columns back to the ORIGINAL feature names so
they are readable ("contract_type", not "contract_type_Month-to-month").
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing import (get_split, add_engineered_features, load_data, original_feature_name, CATEGORICAL)

try:
    import shap
    HAS_SHAP = True
except Exception:                       # pragma: no cover
    HAS_SHAP = False

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
MET_DIR = ROOT / "outputs" / "metrics"
MODEL_DIR = ROOT / "models"

PRETTY = {
    "satisfaction_score": "Low satisfaction score", "nps_score": "Low NPS", "complaints": "High complaints",
    "complaint_change_3m": "Rising complaints", "satisfaction_change_3m": "Falling satisfaction",
    "bill_change_3m": "Bill increase", "network_drops": "Frequent network drops",
    "call_drop_rate": "High call-drop rate", "outage_minutes": "Long outages",
    "contract_type": "Contract type", "tenure_months": "Short tenure", "data_usage_change_3m": "Declining data usage",
    "call_usage_change_3m": "Declining call usage", "late_payment_count": "Late payments",
    "payment_method": "Payment method", "support_calls": "Many support calls",
    "tickets_reopened": "Reopened tickets", "negative_feedback_count": "Negative feedback",
    "monthly_charges": "High monthly charges", "city_tier": "City tier", "age": "Age",
    "network_issue_index": "Poor network quality", "support_intensity": "High support burden",
    "n_negative_trends": "Multiple negative trends", "bill_shock": "Bill shock (>15% rise)",
    "is_new_customer": "New customer", "spend_per_line": "Spend per line",
    "average_download_speed": "Slow download speed", "average_resolution_time_hours": "Slow ticket resolution",
}


def pretty(name):
    return PRETTY.get(name, name.replace("_", " ").capitalize())


def _split_pipeline(pipe):
    return pipe.named_steps["prep"], pipe.named_steps["clf"]


def _positive_class_values(sv):
    """Normalise the SHAP output shape to (n_samples, n_features) for the churn class."""
    if isinstance(sv, list):
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def transformed_frame(pipe, X):
    prep, _ = _split_pipeline(pipe)
    Xt = prep.transform(X)
    names = list(prep.get_feature_names_out())
    return pd.DataFrame(Xt, columns=names, index=X.index)


def compute_shap(pipe, X, background=None, seed=42):
    """Return (shap_values_original_features DataFrame, Xt DataFrame, raw sv ndarray, names)."""
    prep, clf = _split_pipeline(pipe)
    Xt = transformed_frame(pipe, X)
    if hasattr(clf, "coef_"):                                   # linear model
        bg = transformed_frame(pipe, background if background is not None else X.sample(100, random_state=seed))
        sv = shap.LinearExplainer(clf, bg).shap_values(Xt)
    else:                                                       # tree models
        sv = shap.TreeExplainer(clf).shap_values(Xt, check_additivity=False)
    sv = _positive_class_values(sv)
    names = list(Xt.columns)
    orig = pd.Series([original_feature_name(n, CATEGORICAL) for n in names])
    sv_orig = pd.DataFrame(sv, columns=names, index=X.index).T.groupby(orig.values).sum().T
    return sv_orig, Xt, sv, names


def top_risk_factors(sv_row, x_row, k=5):
    """Top-k features that PUSH the churn probability up for one customer (from SHAP)."""
    pos = sv_row[sv_row > 0].sort_values(ascending=False).head(k)
    out = []
    for f, v in pos.items():
        val = x_row.get(f, np.nan)
        val_txt = "missing" if pd.isna(val) else (f"{val:.1f}" if isinstance(val, (float, np.floating)) else str(val))
        label = pretty(f) if not pd.isna(val) else f.replace("_", " ").capitalize()   # directional label only if value known
        out.append({"feature": f, "label": label, "value": val_txt, "shap": float(v)})
    return out


def global_importance(sv_orig):
    return sv_orig.abs().mean().sort_values(ascending=False)


def plot_importance(imp, title, fname, top=15):
    top_imp = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh([pretty(i) for i in top_imp.index], top_imp.values, color="#1f77b4")
    for b in bars:
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {b.get_width():.3f}", va="center", fontsize=8)
    ax.set_xlabel("Importance (mean |SHAP value|)" if "SHAP" in title else "Importance")
    ax.set_ylabel("Feature"); ax.set_title(title); ax.grid(axis="x", alpha=0.3)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight"); plt.close(fig)


def fallback_importance(pipe):
    """Used when SHAP is unavailable: native model importance mapped to original features."""
    prep, clf = _split_pipeline(pipe)
    names = list(prep.get_feature_names_out())
    vals = np.abs(clf.coef_[0]) if hasattr(clf, "coef_") else clf.feature_importances_
    orig = [original_feature_name(n, CATEGORICAL) for n in names]
    return pd.Series(vals, index=orig).groupby(level=0).sum().sort_values(ascending=False)


def run_explainability(pipe=None, n_explain=2000, seed=42):
    """Global SHAP plots + one worked individual example. Saves figures and CSV/JSON."""
    cfg = json.load(open(MODEL_DIR / "model_config.json"))
    if pipe is None:
        pipe = joblib.load(MODEL_DIR / "best_model.pkl")
    X_train, X_test, y_train, y_test = get_split()
    X_test = X_test[cfg["features"]]
    X_train = X_train[cfg["features"]]
    MET_DIR.mkdir(parents=True, exist_ok=True)

    if not HAS_SHAP:
        imp = fallback_importance(pipe)
        imp.to_csv(MET_DIR / "feature_importance.csv", header=["importance"])
        plot_importance(imp, "Feature Importance (model-native fallback)", "feature_importance.png")
        print("SHAP not available -> used model-native feature importance.")
        return imp

    sample = X_test.sample(min(n_explain, len(X_test)), random_state=seed)
    sv_orig, Xt, sv_raw, names = compute_shap(pipe, sample, background=X_train.sample(100, random_state=seed))
    imp = global_importance(sv_orig)
    imp.to_csv(MET_DIR / "feature_importance.csv", header=["mean_abs_shap"])
    plot_importance(imp, "Global Feature Importance (mean |SHAP|, best model)", "feature_importance.png")

    # --- SHAP summary (beeswarm) on original-feature level: requires per-feature values
    # We plot on the transformed matrix for the beeswarm (standard SHAP plot), then labels are readable enough.
    plt.figure(figsize=(9, 7))
    shap.summary_plot(sv_raw, Xt, feature_names=[n.split("__", 1)[1] for n in names], max_display=15, show=False)
    plt.title("SHAP Summary Plot (impact on churn probability)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()

    # --- individual explanation: highest-risk customer in the sample
    pred = pipe.predict_proba(sample)[:, 1]
    cid = sample.index[int(np.argmax(pred))]
    sv_row = sv_orig.loc[cid]
    factors = top_risk_factors(sv_row, sample.loc[cid])
    protect = sv_row[sv_row < 0].sort_values().head(3)
    fig, ax = plt.subplots(figsize=(8, 5))
    show = pd.concat([sv_row.sort_values(ascending=False).head(6), sv_row.sort_values().head(4)]).sort_values()
    ax.barh([pretty(i) for i in show.index], show.values, color=["#d62728" if v > 0 else "#2ca02c" for v in show.values])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("SHAP value (log-odds contribution to churn)"); ax.set_ylabel("Feature")
    ax.set_title(f"Why is customer {cid} at risk? (churn probability {pred.max():.0%})")
    fig.tight_layout(); fig.savefig(FIG_DIR / "shap_individual_example.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    json.dump({"customer_id": cid, "churn_probability": float(pred.max()),
               "risk_factors": factors, "protective_factors": {k: float(v) for k, v in protect.items()}},
              open(MET_DIR / "shap_individual_example.json", "w"), indent=2)
    print(f"SHAP done. Top global features: {', '.join(imp.index[:6])}")
    return imp


if __name__ == "__main__":
    run_explainability()
