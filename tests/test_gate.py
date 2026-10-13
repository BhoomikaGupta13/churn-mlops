from src.models.gate import should_promote

CFG = {"primary_metric": "f1", "min_improvement": 0.01, "min_absolute_f1": 0.55}


def test_promotes_when_clearly_better():
    ok, _ = should_promote({"f1": 0.60}, {"f1": 0.65}, CFG)
    assert ok


def test_rejects_when_the_gain_is_too_small():
    ok, reason = should_promote({"f1": 0.60}, {"f1": 0.605}, CFG)
    assert not ok
    assert "below the required" in reason


def test_rejects_when_worse():
    ok, _ = should_promote({"f1": 0.65}, {"f1": 0.60}, CFG)
    assert not ok


def test_gain_exactly_equal_to_the_margin_is_accepted():
    # 0.6654 - 0.6554 is 0.00999999... in raw floating point; the gate must still accept it
    ok, _ = should_promote({"f1": 0.6554}, {"f1": 0.6654}, CFG)
    assert ok


def test_better_but_below_the_floor_is_rejected():
    ok, reason = should_promote({"f1": 0.40}, {"f1": 0.50}, CFG)
    assert not ok
    assert "floor" in reason


def test_first_model_is_promoted_if_it_clears_the_floor():
    ok, _ = should_promote(None, {"f1": 0.60}, CFG)
    assert ok


def test_first_model_is_rejected_below_the_floor():
    ok, _ = should_promote(None, {"f1": 0.30}, CFG)
    assert not ok
