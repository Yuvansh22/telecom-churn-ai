"""Metrics, threshold tuning and evaluation plots. Saves figures to outputs/figures."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
MET_DIR = ROOT / "outputs" / "metrics"
COLORS = {"Logistic Regression": "#1f77b4", "Random Forest": "#2ca02c", "XGBoost": "#d62728"}


def compute_metrics(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
    }


def recall_target_threshold(y_true, proba, min_recall=0.70):
    """Business threshold: the HIGHEST cut-off (best precision) that still catches at least
    `min_recall` of churners. Must be called with OUT-OF-FOLD training predictions, never test data."""
    grid = np.linspace(0.05, 0.90, 86)
    ok = [t for t in grid if recall_score(y_true, proba >= t) >= min_recall]
    return float(max(ok)) if ok else float(grid[0])


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(y_true, probas, thresholds, fname="confusion_matrices.png", title="Confusion Matrices on Held-out Test Set"):
    fig, axes = plt.subplots(1, len(probas), figsize=(5 * len(probas), 4.4))
    for ax, (name, p) in zip(np.atleast_1d(axes), probas.items()):
        cm = confusion_matrix(y_true, (p >= thresholds[name]).astype(int))
        ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Churned"]).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{name}\n(threshold = {thresholds[name]:.2f})")
        ax.set_xlabel("Predicted label"); ax.set_ylabel("Actual label")
    fig.suptitle(title, fontsize=13, y=1.03)
    _save(fig, fname)


def plot_roc(y_true, probas):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, color=COLORS.get(name), lw=2, label=f"{name} (AUC = {roc_auc_score(y_true, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curves - Test Set"); ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    _save(fig, "roc_curves.png")


def plot_pr(y_true, probas):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, p in probas.items():
        pr, rc, _ = precision_recall_curve(y_true, p)
        ax.plot(rc, pr, color=COLORS.get(name), lw=2, label=f"{name} (PR-AUC = {average_precision_score(y_true, p):.3f})")
    ax.axhline(np.mean(y_true), color="k", ls="--", lw=1, label=f"Baseline (churn rate = {np.mean(y_true):.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves - Test Set"); ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    _save(fig, "precision_recall_curves.png")


def plot_model_comparison(table):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.8 / len(table)
    x = np.arange(len(metrics))
    for i, (name, row) in enumerate(table.iterrows()):
        bars = ax.bar(x + i * w, row[metrics].values.astype(float), w, label=name, color=COLORS.get(name))
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, f"{b.get_height():.2f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x + w * (len(table) - 1) / 2); ax.set_xticklabels([m.upper() if "auc" in m else m.title() for m in metrics])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.set_title("Model Comparison (Test Set, threshold = 0.5)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _save(fig, "model_comparison.png")


def plot_calibration(y_true, probas, n_bins=10):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, p in probas.items():
        bins = pd.qcut(p, n_bins, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": np.asarray(y_true), "b": bins}).groupby("b", observed=True).mean()
        ax.plot(g["p"], g["y"], "o-", color=COLORS.get(name), label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted churn probability"); ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration Curve - Test Set"); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, "calibration_curve.png")


def plot_risk_levels(risk_series):
    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    counts = risk_series.value_counts().reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(order, counts.values, color=["#2ca02c", "#f2c230", "#ff7f0e", "#d62728"])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(b.get_height()):,}", ha="center", va="bottom")
    ax.set_xlabel("Risk level"); ax.set_ylabel("Number of customers"); ax.set_title("Customer Risk Segmentation (Test Set)")
    _save(fig, "risk_level_distribution.png")
