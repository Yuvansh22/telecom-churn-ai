"""Business layer: risk levels, customer health score and RULE-BASED retention actions.

IMPORTANT: Everything in this file is deterministic business logic. It is NOT produced by the
ML model. The ML model only supplies churn probability and the SHAP-based risk factors;
the actions below are hand-written rules that a retention team could edit.
"""
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ risk levels
RISK_BINS = [0.0, 0.30, 0.60, 0.80, 1.0001]          # business thresholds - adjustable
RISK_LABELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def assign_risk_level(proba):
    """0-30% LOW, 30-60% MEDIUM, 60-80% HIGH, 80-100% CRITICAL (lower bound inclusive)."""
    return pd.cut(pd.Series(np.asarray(proba)), bins=RISK_BINS, labels=RISK_LABELS, right=False).astype(str).values


# ------------------------------------------------------------------ health score
HEALTH_WEIGHTS = {"satisfaction": 0.30, "network": 0.20, "complaints": 0.20, "payment": 0.15, "engagement": 0.15}


def customer_health_score(df):
    """Interpretable 0-100 business score (100 = very healthy relationship).

    Each indicator is mapped to [0, 1] with FIXED, documented bounds (not learned from data and
    NOT derived from the churn label -> no leakage):
      satisfaction : (score-1)/9, missing -> 0.5 (neutral)
      network      : mean of 1-drops/15, 1-call_drop_rate/10, 1-outage/300 (each clipped to 0-1)
      complaints   : 1 - (complaints + 0.5*support_calls + tickets_reopened)/10
      payment      : 1 - late_payment_count/5
      engagement   : 0.5 + mean(data_change, call_change)/100, shifted by satisfaction_change/10
    """
    clip = lambda s: np.clip(s, 0, 1)
    sat = (df["satisfaction_score"].fillna(5.5) - 1) / 9
    net = (clip(1 - df["network_drops"] / 15) + clip(1 - df["call_drop_rate"] / 10)
           + clip(1 - df["outage_minutes"] / 300)) / 3
    comp = clip(1 - (df["complaints"] + 0.5 * df["support_calls"] + df["tickets_reopened"]) / 10)
    pay = clip(1 - df["late_payment_count"] / 5)
    eng = clip(0.5 + (df["data_usage_change_3m"] + df["call_usage_change_3m"]) / 200
               + df["satisfaction_change_3m"] / 20)
    w = HEALTH_WEIGHTS
    score = w["satisfaction"] * clip(sat) + w["network"] * net + w["complaints"] * comp \
        + w["payment"] * pay + w["engagement"] * eng
    return (100 * score).round(1)


# ------------------------------------------------------------------ rule-based actions
# (rule name, condition(row) -> bool, action text, related model features used for prioritisation)
RULES = [
    ("network",
     lambda r: r["network_drops"] >= 8 or r["call_drop_rate"] >= 5 or r["outage_minutes"] >= 150
     or (pd.notna(r["average_download_speed"]) and r["average_download_speed"] < 15),
     "Priority network support / network quality investigation for this customer's area.",
     {"network_drops", "call_drop_rate", "outage_minutes", "network_issue_index", "average_download_speed"}),
    ("pricing",
     lambda r: r["bill_change_3m"] >= 15 and r["spend_per_line"] >= 500,
     "Offer a lower-cost or loyalty plan to offset the recent bill increase.",
     {"bill_change_3m", "bill_shock", "monthly_charges", "spend_per_line"}),
    ("service",
     lambda r: pd.notna(r["satisfaction_score"]) and r["satisfaction_score"] <= 4
     and (r["support_calls"] >= 4 or r["complaints"] >= 3),
     "Priority customer-service intervention: senior agent call-back within 48 hours.",
     {"satisfaction_score", "support_calls", "complaints", "support_intensity", "nps_score"}),
    ("tickets",
     lambda r: r["tickets_reopened"] >= 2 or r["average_resolution_time_hours"] >= 48,
     "Escalate unresolved tickets to a dedicated resolution manager.",
     {"tickets_reopened", "average_resolution_time_hours", "complaint_change_3m"}),
    ("payment",
     lambda r: r["late_payment_count"] >= 2,
     "Offer flexible payment options (due-date change, auto-pay incentive).",
     {"late_payment_count", "payment_method"}),
    ("contract",
     lambda r: r["contract_type"] == "Month-to-month" and r["tenure_months"] <= 12,
     "Offer a discounted 12-month contract upgrade with a loyalty benefit.",
     {"contract_type", "tenure_months", "is_new_customer"}),
    ("engagement",
     lambda r: r["data_usage_change_3m"] <= -30 or r["n_negative_trends"] >= 3,
     "Run a re-engagement campaign: personalised data/call pack and usage tips.",
     {"data_usage_change_3m", "call_usage_change_3m", "n_negative_trends", "satisfaction_change_3m"}),
]


def recommend_actions(row, top_factors=(), risk_level="MEDIUM", max_actions=3):
    """Return list of suggested actions. Rules whose related features also appear among the
    model's top SHAP risk factors are ranked first (so the ML explanation orders the list,
    but the action text itself is rule-based)."""
    if risk_level == "LOW":
        return ["No intervention needed - continue standard engagement and monitor."]
    top = set(top_factors)
    hits = [(len(top & feats), text) for _, cond, text, feats in RULES if cond(row)]
    hits.sort(key=lambda t: -t[0])
    actions = [t for _, t in hits][:max_actions]
    return actions or ["General loyalty outreach: personalised thank-you offer and satisfaction check-in call."]
