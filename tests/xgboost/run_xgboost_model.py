# Third Party
import numpy as np
import xgboost


def run_xgboost_model(hook, num_round=10, seed=42, params=None):

    np.random.seed(seed)

    train_data = np.random.rand(5, 10)
    train_label = np.random.randint(2, size=5)
    dtrain = xgboost.DMatrix(train_data, label=train_label)

    test_data = np.random.rand(5, 10)
    test_label = np.random.randint(2, size=5)
    dtest = xgboost.DMatrix(test_data, label=test_label)

    params = params if params else {}

    xgboost.train(
        params,
        dtrain,
        evals=[(dtrain, "train"), (dtest, "test")],
        num_boost_round=num_round,
        callbacks=[hook],
    )
