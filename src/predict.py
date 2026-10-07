"""Score customers: churn probability, risk level, SHAP risk factors, health score, retention action.

Usage:
    python src/predict.py                       # scores data/telecom_churn.csv
    python src/predict.py --input new.csv       # scores any CSV with the same raw columns
    python src/predict.py --customer C100123    # also prints a full explanation for one customer
Output: outputs/predictions/churn_predictions.csv
"""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd

from preprocessing import add_engineered_features, get_split, load_data, ID_COL, TARGET
from retention import assign_risk_level, customer_health_score, recommend_actions
import explain

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUT = ROOT / "outputs" / "predictions" / "churn_predictions.csv"


def score(df_raw, chunk=2000):
    cfg = json.load(open(MODEL_DIR / "model_config.json"))
    pipe = joblib.load(MODEL_DIR / "best_model.pkl")
    df = add_engineered_features(df_raw).set_index(ID_COL, drop=False)
    X = df[cfg["features"]]
    proba = pipe.predict_proba(X)[:, 1]
    risk = assign_risk_level(proba)
    health = customer_health_score(df)

    bg = get_split()[0][cfg["features"]].sample(100, random_state=42)       # SHAP background = training data
    factors_txt, actions_txt, factor_names = [], [], []
    for start in range(0, len(X), chunk):
        Xc = X.iloc[start:start + chunk]
        sv, _, _, _ = explain.compute_shap(pipe, Xc, background=bg)
        for i, cid in enumerate(Xc.index):
            f = explain.top_risk_factors(sv.loc[cid], Xc.loc[cid], k=5)
            names = [x["feature"] for x in f]
            r = df.loc[cid]
            lvl = risk[start + i]
            factor_names.append(names)
            factors_txt.append("; ".join(f"{x['label']} ({x['value']})" for x in f))
            actions_txt.append(" | ".join(recommend_actions(r, names, lvl)))
    out = pd.DataFrame({
        "customer_id": df[ID_COL].values, "churn_probability": proba.round(4), "risk_level": risk,
        "predicted_churn": (proba >= cfg["decision_threshold"]).astype(int),
        "customer_health_score": health.values, "top_risk_factors": factors_txt, "recommended_action": actions_txt,
    })
    return out, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--customer", default=None)
    a = ap.parse_args()
    raw = pd.read_csv(a.input) if a.input else load_data()
    out, cfg = score(raw)
    if TARGET in raw.columns:
        out["actual_churn"] = raw[TARGET].values
    split_path = ROOT / "outputs" / "metrics" / "split.csv"
    if split_path.exists() and not a.input:
        out = out.merge(pd.read_csv(split_path), on="customer_id", how="left")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Best model: {cfg['best_model']} | scored {len(out):,} customers -> {OUT}")
    print(out["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).to_string())
    cid = a.customer or out.sort_values("churn_probability", ascending=False).iloc[0]["customer_id"]
    r = out[out["customer_id"] == cid].iloc[0]
    print(f"\nCustomer {cid}\nChurn probability: {r['churn_probability']:.0%}   Risk: {r['risk_level']}   "
          f"Health score: {r['customer_health_score']}\nMain contributing factors (from SHAP):")
    for i, f in enumerate(r["top_risk_factors"].split("; "), 1):
        print(f"  {i}. {f}")
    print("Recommended action(s) (rule-based, not ML):")
    for x in r["recommended_action"].split(" | "):
        print("  -", x)


if __name__ == "__main__":
    main()
