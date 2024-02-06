import pandas as pd
import numpy as np
import pytest
from sklearn.model_selection import GroupKFold
from notebooks.exploration.utils import cross_validation as cv

@pytest.fixture
def data():
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [10, 20, 30, 40, 50, 60],
        'group': ['group1', 'group1', 'group2', 'group2', 'group3', 'group3']
    }), pd.Series([0, 1, 0, 1, 0, 1]), pd.Series(['group1', 'group1', 'group2', 'group2', 'group3', 'group3'])

def split_data_check(data):
    X, y, groups = data
    X_train, y_train, X_test, y_test = cv.train_test_split(X, y, groups, 2)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)

def groups_by_index_check(data):
    X, y, groups = data
    result = cv.get_groups_by_index(X, groups)
    assert len(result) == len(X)
    assert (result == groups[X.index]).all()