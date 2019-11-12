# Standard Library
import os
import shutil
from datetime import datetime

# First Party
import smdebug.tensorflow as smd
from smdebug.core.json_config import CONFIG_FILE_PATH_ENV_STR
from smdebug.core.reduction_config import ALLOWED_NORMS, ALLOWED_REDUCTIONS
from smdebug.exceptions import *

# Local
from .utils import *


def helper_test_reductions(trial_dir, hook, save_raw_tensor):
    simple_model(hook)
    _, files = get_dirs_files(trial_dir)
    from smdebug.trials import create_trial

    tr = create_trial(trial_dir)
    assert len(tr.tensors()) == 3, tr.tensors()
    for tname in tr.tensors():
        t = tr.tensor(tname)
        if tname in tr.tensors_in_collection("losses"):
            # no reductions
            assert t.value(0) is not None
        else:
            if save_raw_tensor is True:
                assert t.value(0) is not None
            else:
                try:
                    print(t.value(0))
                    assert False, (tname, e)
                except TensorUnavailableForStep as e:
                    pass
            assert len(t.reduction_values(0)) == 18
            for r in ALLOWED_REDUCTIONS + ALLOWED_NORMS:
                for b in [False, True]:
                    assert t.reduction_value(0, reduction_name=r, abs=b, worker=None) is not None


def test_reductions(save_raw_tensor=False):
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = os.path.join("/tmp/tornasole_rules_tests/", run_id)
    pre_test_clean_up()
    rdnc = ReductionConfig(
        reductions=ALLOWED_REDUCTIONS,
        abs_reductions=ALLOWED_REDUCTIONS,
        norms=ALLOWED_NORMS,
        abs_norms=ALLOWED_NORMS,
        save_raw_tensor=save_raw_tensor,
    )
    hook = TornasoleHook(
        out_dir=trial_dir, save_config=SaveConfig(save_interval=1), reduction_config=rdnc
    )
    helper_test_reductions(trial_dir, hook, save_raw_tensor)


def test_reductions_with_raw_tensor():
    test_reductions(save_raw_tensor=True)


def test_reductions_json():
    trial_dir = "newlogsRunTest1/test_reductions"
    shutil.rmtree(trial_dir, ignore_errors=True)
    os.environ[
        CONFIG_FILE_PATH_ENV_STR
    ] = "tests/tensorflow/hooks/test_json_configs/test_reductions.json"
    pre_test_clean_up()
    hook = smd.TornasoleHook.hook_from_config()
    helper_test_reductions(trial_dir, hook, False)
