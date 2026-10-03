import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from model.classifier import FEATURES, load_model, predict, train_model, training_data


@pytest.fixture(scope="session")
def model(tmp_path_factory):
    """Train a fresh model once per test session in pytest's temporary directory."""
    path = tmp_path_factory.mktemp("model") / "classifier.joblib"
    train_model(path)
    return load_model(path)


def valid_input():
    return pd.DataFrame(
        [
            [29, 45.0, 2, "standard", "west"],
            [50, 55.0, 6, "basic", "east"],
        ],
        columns=FEATURES,
    )


def test_valid_input_returns_predictions(model):
    assert len(predict(model, valid_input())) == 2


def test_predictions_are_binary(model):
    assert set(predict(model, valid_input())).issubset({0, 1})


@pytest.mark.parametrize("row_index", [0, 1])
def test_prediction_shape_matches_rows(model, row_index):
    """Each individual input row produces one prediction."""
    assert predict(model, valid_input().iloc[[row_index]]).shape == (1,)


@pytest.mark.bad_input
@pytest.mark.parametrize(
    ("data", "error", "message"),
    [
        (pd.DataFrame(columns=FEATURES), ValueError, "must not be empty"),
        ({"age": 30}, TypeError, "pandas DataFrame"),
    ],
)
def test_invalid_input_contracts_are_rejected(model, data, error, message):
    """Malformed input is rejected with a clear validation error."""
    with pytest.raises(error, match=message):
        predict(model, data)


@pytest.mark.missing_data
@pytest.mark.parametrize(
    "data",
    [
        valid_input().drop(columns=["region"]),
        valid_input().assign(
            age=np.nan, monthly_spend=np.nan, support_tickets=np.nan
        ),
    ],
)
def test_missing_required_data_is_rejected(model, data):
    """Missing required columns or all numeric values are rejected."""
    message = (
        "Missing required columns"
        if "region" not in data.columns
        else "cannot all be missing"
    )
    with pytest.raises(ValueError, match=message):
        predict(model, data)


@pytest.mark.bad_input
def test_unexpected_column_rejected(model):
    """Unexpected columns are rejected rather than silently ignored."""
    with pytest.raises(ValueError, match="Unexpected columns"):
        predict(model, valid_input().assign(unexpected=123))


@pytest.mark.missing_data
def test_partial_missing_values_are_imputed(model):
    """A partially missing numeric row is accepted through median imputation."""
    data = valid_input()
    data.loc[0, "monthly_spend"] = np.nan
    assert len(predict(model, data)) == 2


@pytest.mark.dtype
@pytest.mark.parametrize(
    ("column", "value"),
    [("age", "thirty"), ("monthly_spend", "unknown")],
)
def test_wrong_numeric_types_are_rejected(model, column, value):
    """Non-numeric values in numeric features are rejected."""
    data = valid_input()
    data[column] = value
    with pytest.raises(TypeError, match="must be numeric"):
        predict(model, data)


@pytest.mark.unseen_category
@pytest.mark.parametrize(
    "updates",
    [{"plan": "enterprise"}, {"plan": "enterprise", "region": "central"}],
)
def test_unseen_categories_do_not_crash(model, updates):
    """Unseen categorical values are accepted because unknown categories are ignored."""
    data = valid_input()
    for column, value in updates.items():
        data.loc[0, column] = value
    assert predict(model, data).shape == (2,)


@pytest.mark.missing_data
def test_none_and_nan_numeric_values_are_imputed(model):
    """None and NaN in numeric columns are both treated as missing values."""
    none_data = valid_input()
    nan_data = valid_input()
    none_data.loc[0, "monthly_spend"] = None
    nan_data.loc[0, "monthly_spend"] = np.nan
    np.testing.assert_array_equal(
        predict(model, none_data), predict(model, nan_data)
    )


@pytest.mark.unseen_category
def test_category_casing_and_whitespace_are_distinct(model):
    """Case and whitespace changes create unseen but accepted categories."""
    data = valid_input()
    data.loc[0, "plan"] = "Basic"
    basic = predict(model, data)
    data.loc[0, "plan"] = " basic "
    whitespace = predict(model, data)
    assert basic.shape == whitespace.shape == (2,)


@pytest.mark.quality
def test_shuffled_column_order_gives_identical_predictions(model):
    """Shuffled columns produce the same predictions because features are named."""
    data = valid_input()
    shuffled = data[
        ["region", "support_tickets", "plan", "age", "monthly_spend"]
    ]
    np.testing.assert_array_equal(predict(model, data), predict(model, shuffled))


@pytest.mark.quality
def test_predict_does_not_mutate_input_dataframe(model):
    """Prediction leaves the caller's DataFrame unchanged."""
    data = valid_input()
    before = data.copy(deep=True)
    predict(model, data)
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.quality
def test_single_row_matches_same_row_in_batch(model):
    """A row predicts the same way alone as it does inside a batch."""
    data = valid_input()
    single = predict(model, data.iloc[[0]])[0]
    batch = predict(model, data)[0]
    assert single == batch


@pytest.mark.quality
def test_model_is_reproducible(model):
    """Repeated predictions from the same model and data are identical."""
    np.testing.assert_array_equal(
        predict(model, valid_input()), predict(model, valid_input())
    )


@pytest.mark.quality
@pytest.mark.parametrize("label", [0, 1])
def test_predictions_are_integer_binary_labels(model, label):
    """Predictions use integer binary labels."""
    predictions = predict(model, valid_input())
    assert np.issubdtype(predictions.dtype, np.integer)
    assert set(predictions).issubset({0, 1})
    assert label in (0, 1)


def test_model_can_be_persisted_and_loaded(tmp_path):
    """A trained model can be saved and loaded without changing predictions."""
    path = tmp_path / "classifier.joblib"
    trained = train_model(path)
    assert predict(load_model(path), valid_input()).shape == (2,)
    np.testing.assert_array_equal(
        predict(trained, valid_input()), predict(load_model(path), valid_input())
    )


@pytest.mark.quality
def test_single_valid_record(model):
    """A valid single record returns a binary prediction."""
    assert predict(model, valid_input().iloc[[0]])[0] in (0, 1)


@pytest.mark.quality
def test_model_quality_floor():
    """A held-out split must meet a minimum accuracy threshold to catch bad retrains."""
    data = training_data()
    train, test = train_test_split(
        data, test_size=0.3, random_state=42, stratify=data["churn"]
    )
    model = train_model()
    model.fit(train[FEATURES], train["churn"])
    predictions = predict(model, test[FEATURES])
    assert accuracy_score(test["churn"], predictions) >= 0.67


@pytest.mark.quality
@pytest.mark.parametrize(
    ("value", "expected_error"),
    [(np.inf, True), (-np.inf, True), (-100.0, False), (1e12, False)],
)
def test_extreme_numeric_values_follow_validation_contract(
    model, value, expected_error
):
    """Infinities are rejected; negative and very large finite values are accepted."""
    data = valid_input()
    data.loc[0, "monthly_spend"] = value
    if expected_error:
        with pytest.raises(ValueError, match="must contain finite values"):
            predict(model, data)
    else:
        assert predict(model, data).shape == (2,)
