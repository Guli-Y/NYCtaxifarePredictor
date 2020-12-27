# -*- coding: UTF-8 -*-

# Import from standard library
from NYCtaxifarePredictor.data import get_data, clean_train
import pytest


def test_get_data():
    df = get_data(n=1000)
    assert df.shape == (1000, 8)

def test_clean_data():
    df = get_data(n=1000)
    df_clean = clean_train(df)
    assert df_clean.shape == (976, 9)
