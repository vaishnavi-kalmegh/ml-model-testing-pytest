import numpy as np
import pandas as pd
import pytest
from model.classifier import FEATURES,load_model,predict,train_model

@pytest.fixture(scope="module")
def model(): return train_model()

def valid_input():
    return pd.DataFrame([[29,45.0,2,"standard","west"],[50,55.0,6,"basic","east"]],columns=FEATURES)

def test_valid_input_returns_predictions(model): assert len(predict(model,valid_input()))==2
def test_predictions_are_binary(model): assert set(predict(model,valid_input())).issubset({0,1})
def test_prediction_shape_matches_rows(model): assert predict(model,valid_input().iloc[[0]]).shape==(1,)
def test_empty_dataframe_rejected(model):
    with pytest.raises(ValueError,match="must not be empty"): predict(model,pd.DataFrame(columns=FEATURES))
def test_non_dataframe_rejected(model):
    with pytest.raises(TypeError,match="pandas DataFrame"): predict(model,{"age":30})
def test_missing_required_column_rejected(model):
    with pytest.raises(ValueError,match="Missing required columns"): predict(model,valid_input().drop(columns=["region"]))
def test_unexpected_column_rejected(model):
    with pytest.raises(ValueError,match="Unexpected columns"): predict(model,valid_input().assign(unexpected=123))
def test_all_numeric_values_missing_rejected(model):
    data=valid_input(); data[["age","monthly_spend","support_tickets"]]=np.nan
    with pytest.raises(ValueError,match="cannot all be missing"): predict(model,data)
def test_partial_missing_values_are_imputed(model):
    data=valid_input(); data.loc[0,"monthly_spend"]=np.nan; assert len(predict(model,data))==2
def test_wrong_numeric_type_rejected(model):
    data=valid_input(); data["age"]="thirty"
    with pytest.raises(TypeError,match="must be numeric"): predict(model,data)
def test_multiple_wrong_numeric_types_rejected(model):
    data=valid_input(); data["monthly_spend"]="unknown"
    with pytest.raises(TypeError,match="must be numeric"): predict(model,data)
def test_unseen_category_does_not_crash(model):
    data=valid_input(); data.loc[0,"plan"]="enterprise"; assert len(predict(model,data))==2
def test_multiple_unseen_categories_do_not_crash(model):
    data=valid_input(); data.loc[0,"plan"]="enterprise"; data.loc[0,"region"]="central"; assert predict(model,data).shape==(2,)
def test_none_categorical_value_is_imputed(model):
    data=valid_input(); data.loc[0,"plan"]=None; assert len(predict(model,data))==2
def test_model_is_reproducible(model): np.testing.assert_array_equal(predict(model,valid_input()),predict(model,valid_input()))
def test_predictions_are_integer_labels(model): assert np.issubdtype(predict(model,valid_input()).dtype,np.integer)
def test_model_can_be_persisted_and_loaded(tmp_path):
    path=tmp_path/"classifier.joblib"; train_model(path); assert predict(load_model(path),valid_input()).shape==(2,)
def test_single_valid_record(model): assert predict(model,valid_input().iloc[[0]])[0] in (0,1)
