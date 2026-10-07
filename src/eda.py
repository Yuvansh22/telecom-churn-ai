"""Exploratory data analysis. Saves all EDA figures to outputs/figures.  Run: python src/eda.py"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_data

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
sns.set_theme(style="whitegrid", context="notebook")
BASE = 0.0   # overall churn rate, set in main()


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight"); plt.close(fig)


def rate_bar(df, col, title, xlabel, fname, order=None, rotate=0):
    g = df.groupby(col, observed=True)["churn"].agg(["mean", "size"]).reindex(order) if order else \
        df.groupby(col, observed=True)["churn"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(g.index.astype(str), g["mean"] * 100, color=sns.color_palette("Blues_d", len(g)))
    for b, n in zip(bars, g["size"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}%\n(n={n:,})", ha="center", fontsize=8)
    ax.axhline(BASE * 100, color="red", ls="--", lw=1, label=f"Overall churn {BASE:.1%}")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Churn rate (%)"); ax.legend()
    ax.set_ylim(0, g["mean"].max() * 100 * 1.3); plt.setp(ax.get_xticklabels(), rotation=rotate)
    _save(fig, fname)


def binned_rate(df, col, bins, title, xlabel, fname, labels=None):
    b = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    tmp = df.assign(_b=b)
    rate_bar(tmp, "_b", title, xlabel, fname, rotate=20)


def make_all(df):
    global BASE
    BASE = df["churn"].mean()

    # 1 churn distribution
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    c = df["churn"].value_counts().sort_index()
    bars = ax.bar(["Stayed (0)", "Churned (1)"], c.values, color=["#2ca02c", "#d62728"])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(b.get_height()):,}\n({b.get_height() / len(df):.1%})", ha="center", va="bottom")
    ax.set_title("Churn Distribution"); ax.set_xlabel("Customer outcome"); ax.set_ylabel("Number of customers"); ax.set_ylim(0, c.max() * 1.15)
    _save(fig, "eda_01_churn_distribution.png")

    # 2-3 contract, plan
    rate_bar(df, "contract_type", "Churn Rate by Contract Type", "Contract type", "eda_02_churn_by_contract.png",
             order=["Month-to-month", "One year", "Two year"])
    rate_bar(df, "plan_type", "Churn Rate by Plan Type", "Plan type", "eda_03_churn_by_plan.png",
             order=["Basic", "Standard", "Premium", "Unlimited"])

    # 4 tenure
    binned_rate(df, "tenure_months", [0, 6, 12, 24, 36, 48, 72], "Churn Rate by Tenure", "Tenure (months)",
                "eda_04_churn_by_tenure.png", ["0-6", "7-12", "13-24", "25-36", "37-48", "49-72"])

    # 5 satisfaction
    rate_bar(df.dropna(subset=["satisfaction_score"]).assign(sat=lambda d: d["satisfaction_score"].astype(int)),
             "sat", "Churn Rate by Satisfaction Score", "Satisfaction score (1-10)", "eda_05_satisfaction_vs_churn.png")

    # 6 monthly charges
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.boxplot(data=df, x="churn", y="monthly_charges", ax=axes[0], palette=["#2ca02c", "#d62728"], hue="churn", legend=False)
    axes[0].set_xticks([0, 1], ["Stayed", "Churned"]); axes[0].set_title("Monthly Charges by Churn Status")
    axes[0].set_xlabel("Customer outcome"); axes[0].set_ylabel("Monthly charges (INR)")
    sns.kdeplot(data=df, x="monthly_charges", hue="churn", fill=True, common_norm=False, ax=axes[1], palette=["#2ca02c", "#d62728"])
    axes[1].set_title("Monthly Charges Density"); axes[1].set_xlabel("Monthly charges (INR)"); axes[1].set_ylabel("Density")
    _save(fig, "eda_06_monthly_charges_vs_churn.png")

    # 7 complaints, 8 network drops
    rate_bar(df.assign(c=df["complaints"].clip(upper=5).astype(int).astype(str).replace("5", "5+")),
             "c", "Churn Rate by Number of Complaints", "Complaints (last period)", "eda_07_complaints_vs_churn.png",
             order=["0", "1", "2", "3", "4", "5+"])
    binned_rate(df, "network_drops", [-1, 1, 3, 6, 9, 14, 100], "Churn Rate by Network Drops", "Network drops",
                "eda_08_network_drops_vs_churn.png", ["0-1", "2-3", "4-6", "7-9", "10-14", "15+"])

    # 9 usage change
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col, lab in zip(axes, ["data_usage_change_3m", "call_usage_change_3m"], ["Data usage", "Call usage"]):
        b = pd.cut(df[col], [-100, -40, -20, -5, 5, 20, 200], labels=["< -40%", "-40 to -20", "-20 to -5", "-5 to 5", "5 to 20", "> 20%"])
        g = df.groupby(b, observed=True)["churn"].mean() * 100
        ax.bar(g.index.astype(str), g.values, color=sns.color_palette("Reds_d", len(g))[::-1])
        ax.axhline(BASE * 100, color="k", ls="--", lw=1)
        ax.set_title(f"Churn Rate by {lab} Change (3 months)"); ax.set_xlabel("Change vs previous 3 months (%)"); ax.set_ylabel("Churn rate (%)")
        plt.setp(ax.get_xticklabels(), rotation=20)
    _save(fig, "eda_09_usage_change_vs_churn.png")

    # 10 correlation heatmap
    num = df.drop(columns=["customer_id"]).select_dtypes("number")
    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(num.corr(), cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Pearson correlation"},
                xticklabels=True, yticklabels=True)
    ax.set_title("Correlation Heatmap (numeric features + churn)")
    ax.tick_params(labelsize=7)
    _save(fig, "eda_10_correlation_heatmap.png")

    # extra: correlation of each numeric feature with churn
    cc = num.corr()["churn"].drop("churn").sort_values()
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.barh(cc.index, cc.values, color=["#d62728" if v > 0 else "#1f77b4" for v in cc.values])
    ax.set_title("Correlation of Each Numeric Feature with Churn"); ax.set_xlabel("Pearson correlation with churn"); ax.set_ylabel("Feature")
    ax.tick_params(axis="y", labelsize=7)
    _save(fig, "eda_10b_correlation_with_churn.png")

    # extra: missing values
    miss = (df.isna().mean() * 100)
    miss = miss[miss > 0].sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(miss.index, miss.values, color="#ff7f0e")
    for i, v in enumerate(miss.values):
        ax.text(v, i, f" {v:.1f}%", va="center")
    ax.set_title("Missing Values by Column"); ax.set_xlabel("Missing (%)"); ax.set_ylabel("Column")
    _save(fig, "eda_11_missing_values.png")
    print(f"Saved EDA figures to {FIG_DIR}")


def main():
    df = load_data()
    print(df.shape, f"| churn rate {df['churn'].mean():.2%}")
    make_all(df)


if __name__ == "__main__":
    main()
