"""Draws the project pipeline flow diagram -> outputs/figures/flow_diagram.png"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]

STEPS = [  # (label, phase colour)
    ("Dataset", "data"), ("Data Inspection", "data"), ("Missing Value\nHandling", "prep"),
    ("Encoding", "prep"), ("Feature Scaling", "prep"),
    ("EDA &\nVisualization", "eda"), ("Feature\nEngineering", "eda"), ("Feature\nSelection", "eda"),
    ("Train / Test Split\n(80/20 stratified)", "model"), ("Model Training\nLogistic Regression\nRandom Forest\nXGBoost", "model"),
    ("Hyperparameter\nTuning (CV)", "model"), ("Model\nEvaluation", "eval"), ("Best Model", "eval"),
    ("Churn\nProbability", "out"), ("Risk\nClassification", "out"), ("SHAP\nExplainability", "out"),
    ("Retention\nRecommendation", "out"),
]
COLORS = {"data": "#dbe9f6", "prep": "#fde9c9", "eda": "#e3f2d5", "model": "#e8dcf3", "eval": "#f9d9d9", "out": "#d2f0ee"}
LEGEND = {"data": "Data", "prep": "Preprocessing", "eda": "Analysis & features", "model": "Modelling", "eval": "Evaluation", "out": "Prediction & action"}


def main():
    per_row, w, h, gx, gy = 5, 3.6, 1.9, 0.8, 1.2
    rows = (len(STEPS) + per_row - 1) // per_row
    fig, ax = plt.subplots(figsize=(per_row * (w + gx) / 1.9, rows * (h + gy) / 1.9))
    pos = []
    for i, (label, ph) in enumerate(STEPS):
        r, c = divmod(i, per_row)
        if r % 2 == 1:
            c = per_row - 1 - c                       # snake layout
        x, y = c * (w + gx), -r * (h + gy)
        pos.append((x, y))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.18",
                                    fc=COLORS[ph], ec="#34495e", lw=1.6))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#2c3e50")
        ax.text(x + 0.12, y + h - 0.12, str(i + 1), ha="left", va="top", fontsize=8, color="#7f8c8d")
    for i in range(len(STEPS) - 1):
        (x1, y1), (x2, y2) = pos[i], pos[i + 1]
        if y1 == y2:
            s, e = ((x1 + w, y1 + h / 2), (x2, y2 + h / 2)) if x2 > x1 else ((x1, y1 + h / 2), (x2 + w, y2 + h / 2))
        else:
            s, e = (x1 + w / 2, y1), (x2 + w / 2, y2 + h)
        ax.add_patch(FancyArrowPatch(s, e, arrowstyle="-|>", mutation_scale=18, lw=2, color="#34495e", shrinkA=2, shrinkB=2))
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[k], ec="#34495e") for k in LEGEND]
    ax.legend(handles, LEGEND.values(), loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=9)
    ax.set_xlim(-0.3, per_row * (w + gx) - gx + 0.3); ax.set_ylim(-(rows - 1) * (h + gy) - 0.3, h + 0.5)
    ax.set_title("Telecom Churn Prediction & Retention Intelligence - Pipeline", fontsize=15, fontweight="bold", pad=10)
    ax.axis("off")
    out = ROOT / "outputs" / "figures" / "flow_diagram.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print("Saved", out)


if __name__ == "__main__":
    main()
