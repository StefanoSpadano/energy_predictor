import pandas as pd
import numpy as np 
from model import train_model 
import pytest
from sklearn.ensemble import RandomForestRegressor

def test_train_model_success():
    """
    GIVEN: a dataframe sufficiently big with valid values
    WHEN: the train_model function is called
    THEN: two elements should be returned, an error and a model name
    """

    df_input = pd.DataFrame({
        "timestamps": pd.date_range("2025-01-01", periods=20, freq="h"),
        "temperature_measured": np.random.uniform(low=-20.0, high=59.0, size=20),
        "consumes": np.linspace(50, 200, 20) + np.random.normal(0, 1, 20)
    })

    error, model = train_model(df_input)

    assert error > 0
    assert model
    assert isinstance(error, float)
    assert isinstance(model, RandomForestRegressor)

def test_train_model_dataset_too_small():
    """
    GIVEN: a dataframe which is too small to be splitted in order to train the model
    WHEN: the dataframe is passed to train_model
    THEN: the correct error should be displayed
    """

    n=4

    df_input = pd.DataFrame({
        "timestamps": pd.date_range("2025-01-01", periods=n, freq="h"),
        "temperature_measured": np.random.uniform(low=-49.0, high=59.0, size=n),
        "consumes": np.linspace(50, 200, n) + np.random.normal(0, 1, n)
    })

    with pytest.raises(ValueError):
        train_model(df=df_input)
    
def test_train_model_determinism():
    """
    GIVEN: two identical sets of data 
    WHEN: they are passed to the same training model
    THEN: their error should be identical
    """
    
    n=10

    df_input = pd.DataFrame({
        "timestamps": pd.date_range("2025-01-01", periods=n, freq="h"),
        "temperature_measured": np.linspace(-50.0, 59.0, n),
        "consumes": np.random.uniform(low=50.0, high=200.0, size=n)
    })

    error_1, model_1 = train_model(df_input)
    error_2, model_2 = train_model(df_input)

    assert error_1 == error_2
