from src.monitoring.drift import detect_drift
from src.monitoring.simulate_drift import inject_drift
from tests.data_factory import make_customers

THRESHOLD = 0.2


def test_similar_data_is_not_flagged():
    out = detect_drift(
        make_customers(1500, seed=0), make_customers(500, seed=1), THRESHOLD
    )
    assert out["drift_detected"] is False
    assert out["share_drifted"] < THRESHOLD


def test_shifted_data_is_flagged():
    shifted = inject_drift(make_customers(500, seed=2))
    out = detect_drift(make_customers(1500, seed=0), shifted, THRESHOLD)
    assert out["drift_detected"] is True
    assert "MonthlyCharges" in out["drifted_features"]
    assert "tenure" in out["drifted_features"]
