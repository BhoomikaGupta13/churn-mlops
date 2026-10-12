from src.data.validate import validate_dataframe


def test_valid_data_passes(customers):
    assert validate_dataframe(customers) == []


def test_missing_column_is_reported(customers):
    errors = validate_dataframe(customers.drop(columns=["tenure"]))
    assert any("Missing columns" in e for e in errors)


def test_negative_tenure_is_reported(customers):
    df = customers.copy()
    df.loc[5, "tenure"] = -3
    assert any("tenure" in e for e in validate_dataframe(df))


def test_bad_churn_label_is_reported(customers):
    df = customers.copy()
    df.loc[5, "Churn"] = "Maybe"
    assert any("Churn" in e for e in validate_dataframe(df))


def test_duplicate_ids_are_reported(customers):
    df = customers.copy()
    df.loc[1, "customerID"] = df.loc[0, "customerID"]
    assert any("Duplicate" in e for e in validate_dataframe(df))


def test_too_many_nulls_is_reported(customers):
    df = customers.copy()
    df.loc[df.index[:100], "Contract"] = None
    assert any("null threshold" in e for e in validate_dataframe(df))


def test_empty_data_is_reported(customers):
    assert validate_dataframe(customers.iloc[0:0]) == ["Dataset is empty"]
