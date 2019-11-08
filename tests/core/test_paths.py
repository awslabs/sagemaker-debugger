from tornasole.core.hook_utils import verify_and_get_out_dir
from tornasole.core.json_config import DEFAULT_SAGEMAKER_TORNASOLE_PATH
from tornasole.core.access_layer.file import (
    get_temp_path,
    SAGEMAKER_TEMP_PATH_SUFFIX,
    NON_SAGEMAKER_TEMP_PATH_PREFIX,
)
from tornasole.core.access_layer.utils import training_has_ended
from tornasole.core.utils import SagemakerSimulator, ScriptSimulator
import tornasole.tensorflow as ts
import os
import shutil
import uuid


def test_outdir_non_sagemaker():
    id = str(uuid.uuid4())
    path = "/tmp/tests/" + id
    out_dir = verify_and_get_out_dir(path)
    assert out_dir == path
    os.makedirs(path)
    training_has_ended(path)
    try:
        verify_and_get_out_dir(path)
        # should raise exception as dir present
        assert False
    except RuntimeError as e:
        pass


def test_outdir_sagemaker():
    os.environ["TRAINING_JOB_NAME"] = "a"
    id = str(uuid.uuid4())
    paths = ["/tmp/tests/" + id, "s3://tmp/tests/" + id]
    for path in paths:
        out_dir = verify_and_get_out_dir(path)
        assert out_dir == DEFAULT_SAGEMAKER_TORNASOLE_PATH
    del os.environ["TRAINING_JOB_NAME"]


def test_tensorboard_dir_sagemaker():
    """ In Sagemaker, we read the tensorboard_dir from a separate JSON config file. """
    with SagemakerSimulator() as sim:
        ts.del_hook()
        hook = ts.get_hook()
        assert hook.out_dir == sim.out_dir
        assert hook.tensorboard_dir == sim.tensorboard_dir


def test_tensorboard_dir_script_default():
    """ In script mode, we default to no tensorboard. """
    with ScriptSimulator() as sim:
        hook = ts.TornasoleHook(out_dir=sim.out_dir)
        assert hook.tensorboard_dir is None


def test_tensorboard_dir_script_export_tensorboard():
    """ In script mode, passing `export_tensorboard=True` results in tensorboard_dir=out_dir. """
    with ScriptSimulator() as sim:
        hook = ts.TornasoleHook(out_dir=sim.out_dir, export_tensorboard=True)
        assert hook.tensorboard_dir == hook.out_dir


def test_tensorboard_dir_script_specify_tensorboard_dir():
    """ In script mode, passing `export_tensorboard` and `tensorboard_dir` works. """
    with ScriptSimulator(tensorboard_dir="/tmp/tensorboard_dir") as sim:
        hook = ts.TornasoleHook(
            out_dir=sim.out_dir, export_tensorboard=True, tensorboard_dir=sim.tensorboard_dir
        )
        assert hook.tensorboard_dir == sim.tensorboard_dir


def test_tensorboard_dir_non_sagemaker_forgot_export_tensorboard():
    """ In script mode, passing tensorboard_dir will work. """
    with ScriptSimulator(tensorboard_dir="/tmp/tensorboard_dir") as sim:
        hook = ts.TornasoleHook(out_dir=sim.out_dir, tensorboard_dir=sim.tensorboard_dir)
        assert hook.tensorboard_dir == sim.tensorboard_dir


def test_temp_paths():
    for path in [
        "/opt/ml/output/tensors/events/a",
        "/opt/ml/output/tensors/a",
        "/opt/ml/output/tensors/events/a/b",
    ]:
        tp = get_temp_path(path)
        assert tp.endswith(SAGEMAKER_TEMP_PATH_SUFFIX)
        assert not tp.startswith(NON_SAGEMAKER_TEMP_PATH_PREFIX)

    for path in ["/a/b/c", "/opt/ml/output/a", "a/b/c"]:
        tp = get_temp_path(path)
        assert not SAGEMAKER_TEMP_PATH_SUFFIX in tp
        assert tp.startswith(NON_SAGEMAKER_TEMP_PATH_PREFIX)


def test_s3_path_that_exists_without_end_of_job():
    path = "s3://tornasole-testing/s3-path-without-end-of-job"
    verify_and_get_out_dir(path)
    try:
        verify_and_get_out_dir(path)
        # should not raise as dir present but does not have the end of job file
    except RuntimeError as e:
        assert False
