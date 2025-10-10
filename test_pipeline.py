import numpy as np
import pandas as pd
from pipeline import load_data, prepare_data
import pytest
import pytest_html as pytest_html

def test_no_missing_after_transform():
    df = load_data('data.csv')
    X_train, X_test, y_train, y_test, preproc = prepare_data(df)

    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()


def test_dimensions_after_split():
    df = load_data('data.csv')
    X_train, X_test, y_train, y_test, preproc = prepare_data(df)

    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)


def test_numeric_output():
    df = load_data('data.csv')
    X_train, X_test, y_train, y_test, preproc = prepare_data(df)

    assert np.issubdtype(X_train.dtype, np.number)
    assert np.issubdtype(X_test.dtype, np.number)
