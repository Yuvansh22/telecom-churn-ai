"""Synthetic telecom customer dataset generator.

Churn is produced from a *latent* logistic risk mechanism that combines many
customer attributes plus unobserved noise (so the problem is not trivially
separable). The latent risk itself is never saved to the CSV.
Run:  python src/generate_data.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telecom_churn.csv"
TARGET_CHURN_RATE = 0.30


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n=20000, seed=SEED):
    rng = np.random.default_rng(seed)

    # ---------------- Demographics ----------------
    age = np.clip(rng.normal(41, 14, n), 18, 85).astype(int)
    gender = rng.choice(["Male", "Female", "Other"], n, p=[0.49, 0.49, 0.02])
    senior = (age >= 60).astype(int)
    city_tier = rng.choice([1, 2, 3], n, p=[0.30, 0.40, 0.30])

    # ---------------- Account ----------------
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    plan = rng.choice(["Basic", "Standard", "Premium", "Unlimited"], n, p=[0.30, 0.30, 0.25, 0.15])
    lines = rng.choice([1, 2, 3, 4, 5], n, p=[0.35, 0.30, 0.18, 0.10, 0.07])
    device = rng.choice(["Android", "iPhone", "Feature Phone"], n, p=[0.60, 0.32, 0.08])
    tenure_mult = pd.Series(contract).map({"Month-to-month": 0.8, "One year": 1.3, "Two year": 1.8}).values
    tenure = np.clip(rng.gamma(2.0, 12.0, n) * tenure_mult, 1, 72).astype(int)

    # ---------------- Hidden drivers (NOT saved) ----------------
    net_bad = rng.normal(0, 1, n) + 0.25 * (city_tier - 2)      # network badness
    decline = rng.normal(0, 1, n)                                # recent deterioration of relationship

    # ---------------- Network quality ----------------
    network_drops = rng.poisson(np.exp(0.9 + 0.45 * net_bad))
    outage_minutes = np.round(rng.gamma(1.5, 25 * np.exp(0.35 * net_bad)), 1)
    download = np.clip(rng.lognormal(np.log(45) - 0.30 * net_bad, 0.40), 2, 300)
    five_g_share = np.where(device == "Feature Phone", 0.0,
                            np.clip(rng.normal(35 - 8 * (city_tier - 1), 22, n), 0, 100))
    download = np.round(download * (1 + 0.004 * five_g_share), 1)
    upload = np.round(download * rng.uniform(0.10, 0.40, n), 1)
    call_drop_rate = np.round(np.clip(1.5 * np.exp(0.40 * net_bad + 0.3 * rng.normal(0, 1, n)), 0.1, 25), 2)

    # ---------------- Support ----------------
    support_calls = rng.poisson(np.exp(0.25 + 0.40 * net_bad + 0.15 * decline))
    complaints_prev = rng.poisson(np.exp(-0.65 + 0.35 * net_bad))
    complaints = rng.poisson(np.exp(-0.65 + 0.40 * net_bad + 0.45 * decline))
    complaint_change = complaints - complaints_prev
    tickets_reopened = rng.binomial(complaints, 0.25)
    resolution_hours = np.round(np.clip(rng.gamma(3.0, 6.0 * np.exp(0.2 * net_bad)), 1, 120), 1)

    # ---------------- Usage ----------------
    base_gb = pd.Series(plan).map({"Basic": 4, "Standard": 10, "Premium": 20, "Unlimited": 35}).values
    data_usage = np.round(base_gb * lines ** 0.3 * rng.lognormal(0, 0.45, n), 2)
    call_minutes = np.round(rng.gamma(3.0, 110, n) * lines ** 0.3, 0)
    sms = rng.poisson(np.where(age < 35, 40, 90))
    intl = np.round(rng.gamma(1.2, 25, n) * rng.binomial(1, 0.22, n), 1)
    roaming = np.round(rng.gamma(1.0, 150, n) * rng.binomial(1, 0.15, n), 1)   # MB
    data_change = np.round(np.clip(rng.normal(0, 14, n) - 9 * decline, -90, 150), 1)
    call_change = np.round(np.clip(rng.normal(0, 14, n) - 6 * decline + 0.25 * data_change, -90, 150), 1)

    # ---------------- Billing ----------------
    base_price = pd.Series(plan).map({"Basic": 299, "Standard": 499, "Premium": 799, "Unlimited": 1099}).values
    monthly = np.round(base_price * lines ** 0.85 * (1 + rng.normal(0, 0.06, n)) + 0.8 * intl + 0.05 * roaming, 2)
    total = np.round(monthly * tenure * rng.uniform(0.90, 1.05, n), 2)
    price_hike = rng.binomial(1, 0.20, n) * rng.exponential(14, n)
    bill_change = np.round(rng.normal(1.5, 6, n) + price_hike + 2.0 * np.clip(decline, 0, None), 1)
    avg_bill_3m = np.round(monthly / (1 + bill_change / 100), 2)
    payment = rng.choice(["Auto-pay", "Credit Card", "Debit Card", "UPI", "Cash"], n, p=[0.30, 0.20, 0.20, 0.20, 0.10])
    late_rate = pd.Series(payment).map({"Auto-pay": 0.08, "Credit Card": 0.25, "Debit Card": 0.30,
                                        "UPI": 0.30, "Cash": 0.80}).values
    late_payments = rng.poisson(late_rate * (1 + 0.3 * np.clip(decline, 0, None)))

    # ---------------- Feedback ----------------
    sat_latent = (7.6 - 0.85 * net_bad - 0.35 * complaints - 0.10 * np.minimum(support_calls, 8)
                  - 0.15 * late_payments + 0.004 * (download - 45) - 0.4 * np.clip(bill_change - 10, 0, None) / 10
                  + rng.normal(0, 1.1, n))
    satisfaction = np.clip(np.round(sat_latent), 1, 10).astype(int)
    sat_change = np.round(np.clip(rng.normal(0, 0.8, n) - 0.7 * decline - 0.2 * complaint_change, -7, 4), 1)
    nps = np.clip(np.round((satisfaction - 5.5) * 22 + rng.normal(0, 25, n)), -100, 100).astype(int)
    neg_feedback = rng.poisson(0.15 + 0.22 * (10 - satisfaction) + 0.2 * complaints)

    # ---------------- Latent churn risk ----------------
    z = (
        0.75 * (contract == "Month-to-month") - 0.95 * (contract == "Two year")
        - 0.025 * tenure + 0.80 * (tenure <= 6)
        - 0.32 * (satisfaction - 6) - 0.006 * nps
        + 0.16 * complaints + 0.18 * complaint_change - 0.22 * sat_change
        + 0.022 * bill_change + 0.70 * ((bill_change > 15) & (satisfaction <= 5))     # interaction
        - 0.012 * data_change - 0.008 * call_change + 0.5 * (data_change < -30)       # threshold
        + 0.05 * network_drops + 0.05 * call_drop_rate + 0.0015 * outage_minutes
        + 0.18 * late_payments - 0.35 * (payment == "Auto-pay")
        + 0.0004 * (monthly / lines - 500) + 0.14 * (city_tier == 3)
        + 0.008 * (40 - age) + 0.15 * (device == "iPhone") * (plan == "Basic")
        + rng.normal(0, 1.1, n)                                                         # unobserved factors
    )
    lo, hi = -10.0, 10.0
    for _ in range(60):                       # bisection: calibrate intercept to target churn rate
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() > TARGET_CHURN_RATE:
            hi = mid
        else:
            lo = mid
    churn = rng.binomial(1, _sigmoid(z + (lo + hi) / 2))

    df = pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(n)],
        "age": age, "gender": gender, "senior_citizen": senior, "city_tier": city_tier,
        "tenure_months": tenure, "contract_type": contract, "plan_type": plan,
        "number_of_lines": lines, "device_type": device,
        "data_usage_gb": data_usage, "call_minutes": call_minutes, "sms_count": sms,
        "international_minutes": intl, "roaming_usage": roaming, "five_g_usage_percentage": np.round(five_g_share, 1),
        "data_usage_change_3m": data_change, "call_usage_change_3m": call_change,
        "monthly_charges": monthly, "total_charges": total, "average_bill_last_3_months": avg_bill_3m,
        "bill_change_3m": bill_change, "late_payment_count": late_payments, "payment_method": payment,
        "network_drops": network_drops, "outage_minutes": outage_minutes,
        "average_download_speed": download, "average_upload_speed": upload, "call_drop_rate": call_drop_rate,
        "support_calls": support_calls, "complaints": complaints,
        "average_resolution_time_hours": resolution_hours, "tickets_reopened": tickets_reopened,
        "complaint_change_3m": complaint_change, "satisfaction_change_3m": sat_change,
        "satisfaction_score": satisfaction.astype(float), "nps_score": nps.astype(float),
        "negative_feedback_count": neg_feedback,
        "churn": churn,
    })

    # ---------------- Intentional missing values ----------------
    miss = {"satisfaction_score": 0.05, "nps_score": 0.06, "international_minutes": 0.03,
            "roaming_usage": 0.03, "average_download_speed": 0.04, "average_upload_speed": 0.04}
    for col, p in miss.items():
        df.loc[rng.random(n) < p, col] = np.nan
    # survey non-response is a bit more common among silent, stable customers (MAR-style)
    quiet = (df["complaints"] == 0) & (df["support_calls"] <= 1)
    df.loc[quiet & (rng.random(n) < 0.04), "satisfaction_score"] = np.nan
    return df


def main():
    df = generate()
    DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df):,} rows x {df.shape[1]} cols -> {DATA_PATH}")
    print(f"Churn rate: {df['churn'].mean():.2%}   |   predictive features: {df.shape[1] - 2}")
    miss = df.isna().sum()
    print("Missing values per column:\n", miss[miss > 0].to_string())


if __name__ == "__main__":
    main()
