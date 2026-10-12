from src.monitoring.simulate_drift import inject_drift


def test_drift_shifts_distributions_but_keeps_the_schema(customers):
    drifted = inject_drift(customers)
    assert list(drifted.columns) == list(customers.columns)
    assert len(drifted) == len(customers)
    assert drifted["MonthlyCharges"].mean() > customers["MonthlyCharges"].mean() * 1.2
    assert drifted["tenure"].mean() < customers["tenure"].mean() * 0.6
    base = (customers["Contract"] == "Month-to-month").mean()
    assert (drifted["Contract"] == "Month-to-month").mean() > base


def test_drift_does_not_modify_the_input(customers):
    before = customers.copy()
    inject_drift(customers)
    assert customers.equals(before)
