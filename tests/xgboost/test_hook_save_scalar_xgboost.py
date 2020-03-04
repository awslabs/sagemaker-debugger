# Standard Library
import os
import time
from datetime import datetime

# Third Party
import numpy as np
import pytest
import xgboost
from tests.core.utils import check_tf_events, delete_local_trials, verify_files

# First Party
from smdebug.core.modes import ModeKeys
from smdebug.core.save_config import SaveConfig, SaveConfigMode
from smdebug.xgboost import Hook as XG_Hook

SMDEBUG_XG_HOOK_TESTS_DIR = "/tmp/test_output/smdebug_xg/tests/"


def simple_xg_model(hook, num_round=10, seed=42, with_timestamp=False):

    np.random.seed(seed)

    train_data = np.random.rand(5, 10)
    train_label = np.random.randint(2, size=5)
    dtrain = xgboost.DMatrix(train_data, label=train_label)

    test_data = np.random.rand(5, 10)
    test_label = np.random.randint(2, size=5)
    dtest = xgboost.DMatrix(test_data, label=test_label)

    params = {}

    scalars_to_be_saved = dict()
    ts = time.time()
    hook.save_scalar(
        "xg_num_steps", num_round, sm_metric=True, timestamp=ts if with_timestamp else None
    )
    scalars_to_be_saved["scalar/xg_num_steps"] = (ts, num_round)

    ts = time.time()
    hook.save_scalar(
        "xg_before_train", 1, sm_metric=False, timestamp=ts if with_timestamp else None
    )
    scalars_to_be_saved["scalar/xg_before_train"] = (ts, 1)

    hook.set_mode(ModeKeys.TRAIN)
    xgboost.train(
        params,
        dtrain,
        evals=[(dtrain, "train"), (dtest, "test")],
        num_boost_round=num_round,
        callbacks=[hook],
    )
    ts = time.time()
    hook.save_scalar("xg_after_train", 1, sm_metric=False, timestamp=ts if with_timestamp else None)
    scalars_to_be_saved["scalar/xg_after_train"] = (ts, 1)
    return scalars_to_be_saved


def helper_xgboost_tests(collection, save_config, with_timestamp):
    coll_name, coll_regex = collection

    run_id = "trial_" + coll_name + "-" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join(SMDEBUG_XG_HOOK_TESTS_DIR, run_id)

    hook = XG_Hook(
        out_dir=trial_dir,
        include_collections=[coll_name],
        save_config=save_config,
        export_tensorboard=True,
    )

    saved_scalars = simple_xg_model(hook, with_timestamp=with_timestamp)
    hook.close()
    verify_files(trial_dir, save_config, saved_scalars)
    if with_timestamp:
        check_tf_events(trial_dir, saved_scalars)


@pytest.mark.parametrize("collection", [("all", ".*"), ("scalars", "^scalar")])
@pytest.mark.parametrize(
    "save_config",
    [
        SaveConfig(save_steps=[0, 2, 4, 6, 8]),
        SaveConfig(
            {
                ModeKeys.TRAIN: SaveConfigMode(save_interval=2),
                ModeKeys.GLOBAL: SaveConfigMode(save_interval=3),
                ModeKeys.EVAL: SaveConfigMode(save_interval=1),
            }
        ),
    ],
)
@pytest.mark.parametrize("with_timestamp", [True, False])
def test_xgboost_save_scalar(collection, save_config, with_timestamp):
    helper_xgboost_tests(collection, save_config, with_timestamp)
    delete_local_trials([SMDEBUG_XG_HOOK_TESTS_DIR])
