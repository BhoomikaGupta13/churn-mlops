"""Builds small, realistic, fake Telco-style customer data for tests."""

import numpy as np
import pandas as pd

YES_NO = ["Yes", "No"]
THREE_WAY = ["Yes", "No", "No internet service"]
PAYMENT = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


def make_customers(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 73, n)
    monthly = rng.uniform(20, 115, n).round(2)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20]
    )

    df = pd.DataFrame(
        {
            "customerID": [f"C{i:05d}" for i in range(n)],
            "gender": rng.choice(["Female", "Male"], n),
            "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
            "Partner": rng.choice(YES_NO, n),
            "Dependents": rng.choice(YES_NO, n, p=[0.3, 0.7]),
            "tenure": tenure,
            "PhoneService": rng.choice(YES_NO, n, p=[0.9, 0.1]),
            "MultipleLines": rng.choice(["No phone service", "No", "Yes"], n),
            "InternetService": rng.choice(
                ["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.20]
            ),
            "OnlineSecurity": rng.choice(THREE_WAY, n),
            "OnlineBackup": rng.choice(THREE_WAY, n),
            "DeviceProtection": rng.choice(THREE_WAY, n),
            "TechSupport": rng.choice(THREE_WAY, n),
            "StreamingTV": rng.choice(THREE_WAY, n),
            "StreamingMovies": rng.choice(THREE_WAY, n),
            "Contract": contract,
            "PaperlessBilling": rng.choice(YES_NO, n),
            "PaymentMethod": rng.choice(PAYMENT, n),
            "MonthlyCharges": monthly,
        }
    )
    df["TotalCharges"] = pd.Series((tenure * monthly).round(2), dtype=object)

    # The real Telco file has blank TotalCharges for brand-new customers. Reproduce that quirk.
    df.loc[[0, 1], "tenure"] = 0
    df.loc[[0, 1], "TotalCharges"] = " "

    # Give the fake data a real, learnable signal: short contracts and low tenure churn more.
    logit = (
        -1.0
        + 1.5 * (contract == "Month-to-month")
        - 0.03 * df["tenure"].to_numpy()
        + 0.01 * monthly
    )
    p = 1 / (1 + np.exp(-logit))
    df["Churn"] = np.where(rng.random(n) < p, "Yes", "No")
    return df
