# Third Party
import numpy as np
import pytest
import xgboost as xgb

# First Party
from smdebug.xgboost.utils import parse_tree_model, is_xgb_1_5_and_later

@pytest.mark.slow
def test_parse_tree_model():
    np.random.seed(42)
    num_row, num_col = 20, 10
    train_data = np.random.rand(num_row, num_col)
    train_label = np.random.randint(2, size=num_row)
    dtrain = xgb.DMatrix(train_data, label=train_label)
    valid_data = np.random.rand(num_row, num_col)
    valid_label = np.random.randint(2, size=num_row)
    dvalid = xgb.DMatrix(valid_data, label=valid_label)
    params = {"objective": "binary:logistic"}
    num_boost_round = 10
    bst = xgb.train(params, dtrain, evals=[(dtrain, "train")], num_boost_round=num_boost_round)

    columns = ["Tree", "Node", "ID", "Feature", "Split", "Yes", "No", "Missing", "Gain", "Cover"]
    if is_xgb_1_5_and_later():
        columns.append("Category")

    try:
        from pandas import DataFrame  # noqa

        pandas_installed = True
    except ImportError:
        pandas_installed = False

    for iteration in range(num_boost_round):
        tree = parse_tree_model(bst, iteration)
        if pandas_installed:
            df = bst.trees_to_dataframe()
            df = df.loc[df["Tree"] == iteration].reset_index(drop=True)
            for col in columns:
                x, y = tree[col], df[col].values
                if x.dtype != y.dtype:
                    y = y.astype(x.dtype)
                # We first check if arrays are equal. Due to the fact that
                # np.nan != np.nan, equality check will fail if we have any
                # np.nan's. If all items are not equal, then we also check for
                # inequality in the same positions because np.nan != np.nan.
                assert ((x == y) | ((x != x) & (y != y))).all()
        assert sorted(tree.keys()) == sorted(columns)
        assert len(tree["Tree"]) > 0
        assert all(len(tree["Tree"]) == len(tree[col]) for col in columns)
        assert all(x == iteration for x in tree["Tree"])
        assert (sorted(tree["Node"]) == tree["Node"]).all()
