import pandas as pd
import numpy as np
import pytest
from database import gen_data

def test_gen_data():
    """
    GIVEN: the user inputs a dataframe with 500 rows to be generated,
    WHEN: the function gen_data is called,
    THEN: the resulting dataframe should be the one expected (500 rows with correct column names). 
    """
    df_test = gen_data(500, "1500-01-01")

    assert len(df_test) == 500
    assert list(df_test.columns) == ["timestamps", "temperature_measured", "consumes"]
    assert df_test["temperature_measured"].dtype == float
    assert df_test["consumes"].dtype == float
    assert pd.api.types.is_datetime64_any_dtype(df_test["timestamps"])

def test_thermodynamic_limits():
    """
    GIVEN: a dataframe generated with 200 entries and starting date as 2025-01-01
    WHEN: its rows are being filtered for invalid parameters (misread by the device)
    THEN: the remaining temperatures have to be in the range [-10, +39]
    """

    df_test = gen_data(200, "2025-01-01")
    clean_df_test_data = df_test[df_test["temperature_measured"] != 999.0]
    
    assert clean_df_test_data["temperature_measured"].between(-10, 39).all()

def test_fault_injection():
    """
    GIVEN: a dataframe generated with at least 100 entries so that we still have faulting entries with our gen_data
    WHEN: the columns consumes and temperature_measured are being inspected
    THEN: we expect a value different from 1 for both the number of nan entries in cunsumes and 999.0 in temperature_measured
    """

    df_test = gen_data(500, "2025-01-01")

    n_nan_in_consumes = df_test["consumes"].isna().sum()
    n_999_0_in_temperature_measured = (df_test["temperature_measured"] == 999.0).sum()

    assert n_nan_in_consumes > 0
    assert n_999_0_in_temperature_measured > 0