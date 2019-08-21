import os
import shutil
from datetime import datetime
from tornasole.core.reduction_config import ALLOWED_REDUCTIONS, ALLOWED_NORMS
from tornasole.core.json_config import TORNASOLE_CONFIG_FILE_PATH_ENV_STR
from tornasole.exceptions import *
import tornasole.tensorflow as ts
from .utils import *


def helper_test_reductions(trial_dir, hook):
    simple_model(hook)
    _, files = get_dirs_files(trial_dir)
    coll = ts.get_collections()
    from tornasole.trials import create_trial

    tr = create_trial(trial_dir)
    assert len(tr.tensors()) == 2
    for tname in tr.tensors():
        t = tr.tensor(tname)
        try:
            t.value(0)
            assert False
        except TensorUnavailableForStep:
            pass
        assert len(t.reduction_values(0)) == 18
        for r in ALLOWED_REDUCTIONS + ALLOWED_NORMS:
            for b in [False, True]:
                assert t.reduction_value(0, reduction_name=r, abs=b) is not None


def test_reductions():
    run_id = 'trial_' + datetime.now().strftime('%Y%m%d-%H%M%S%f')
    trial_dir = os.path.join('/tmp/tornasole_rules_tests/', run_id)
    pre_test_clean_up()
    rdnc = ReductionConfig(reductions=ALLOWED_REDUCTIONS,
                           abs_reductions=ALLOWED_REDUCTIONS,
                           norms=ALLOWED_NORMS,
                           abs_norms=ALLOWED_NORMS)
    hook = TornasoleHook(out_dir=trial_dir,
                         save_config=SaveConfig(save_interval=1),
                         reduction_config=rdnc)
    helper_test_reductions(trial_dir, hook)


def test_reductions_json():
    trial_dir = "newlogsRunTest1/test_reductions"
    shutil.rmtree(trial_dir, ignore_errors=True)
    os.environ[
        TORNASOLE_CONFIG_FILE_PATH_ENV_STR] = "tests/tensorflow/hooks/test_json_configs/test_reductions.json"
    pre_test_clean_up()
    hook = ts.TornasoleHook.hook_from_config()
    helper_test_reductions(trial_dir, hook)
