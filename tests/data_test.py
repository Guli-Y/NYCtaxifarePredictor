# -*- coding: UTF-8 -*-

# Import from standard library
from NYCtaxifarePredictor.data import get_data, clean_data
import pytest

def test_clean_data():
    df = get_data(n=1000)
    assert df.shape == (1000, 8)
    df_clean = clean_data(df)
    assert df_clean.shape == (976, 9)
