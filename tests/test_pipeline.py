import pandas as pd
import numpy as np
import pytest
from pipeline import clean_data

def test_clean_data_happy_path():
    """
    GIVEN: a dataframe with some input errors
    WHEN: the function to clean the data is called
    THEN: the dataframe gets cleaned from invalid values inside of it
    """
    df_input = pd.DataFrame({
        "timestamps": pd.date_range("2025-01-02", periods=3, freq="h"),
        "temperature_measured": [38.0, 999.0, -4.0],
        "consumes": [55.0, 60.0, None]
    })

    df_clean = clean_data(df_input)
    
    assert len(df_clean) == 1
    assert list(df_clean.columns) == ["timestamps", "temperature_measured", "consumes"]
    assert df_clean.loc[0, "temperature_measured"] == 38.0
    assert list(df_clean.index) == [0]

def test_clean_data_all_entries_are_wrong():
    """
    GIVEN: a dataframe containing invalid values for our analysis
    WHEN: the function clean_data is called
    THEN: the resulting dataframe should be empty
    """

    df_input = pd.DataFrame({
        "timestamps": pd.date_range("2025-01-01", periods=4, freq="h"),
        "temperature_measured": [-184.0, -np.exp(9), 999999.0, "no data"],
        "consumes": [None, np.pi, "nan", -20.0]
    })

    df_clean = clean_data(df_input)

    assert df_clean.empty
    assert list(df_clean.columns) == ["timestamps", "temperature_measured", "consumes"]

def test_clean_data_already_clean():
    """
    GIVEN: a database with all correct entries
    WHEN: the function clean_data is called
    THEN: an identical database to the one passed to the function should be returned
    """
    df_input = pd.DataFrame({
        "timestamps": pd.date_range("2025-01-01", periods=3, freq="h"),
        "temperature_measured": [-49.99999999, 0.0, 25.0],
        "consumes": [120.0, 70.0, 50.0]
    })

    df_clean = clean_data(df_input)

    assert len(df_clean) == len(df_input)
    assert list(df_clean.index) == [0, 1, 2]
    assert df_clean["temperature_measured"].dtype == float
    assert pd.api.types.is_numeric_dtype(df_clean["consumes"])