# Standard Library
import os
import uuid

# First Party
import smdebug.pytorch as smd
from smdebug.core.access_layer.file import (
    NON_SAGEMAKER_TEMP_PATH_PREFIX,
    SAGEMAKER_TEMP_PATH_SUFFIX,
    get_temp_path,
)
from smdebug.core.access_layer.utils import training_has_ended
from smdebug.core.hook_utils import verify_and_get_out_dir
from smdebug.core.json_config import DEFAULT_SAGEMAKER_OUTDIR
from smdebug.core.utils import SagemakerSimulator, ScriptSimulator


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


def test_tensorboard_dir_sagemaker():
    """ In Sagemaker, we read the tensorboard_dir from a separate JSON config file. """
    with SagemakerSimulator() as sim:
        smd.del_hook()
        hook = smd.get_hook(create_if_not_exists=True)
        assert hook.out_dir == sim.out_dir
        assert hook.tensorboard_dir == sim.tensorboard_dir


def test_tensorboard_dir_script_default():
    """ In script mode, we default to no tensorboard. """
    with ScriptSimulator() as sim:
        hook = smd.Hook(out_dir=sim.out_dir)
        assert hook.tensorboard_dir is None


def test_tensorboard_dir_script_export_tensorboard():
    """ In script mode, passing `export_tensorboard=True` results in tensorboard_dir=out_dir. """
    with ScriptSimulator() as sim:
        hook = smd.Hook(out_dir=sim.out_dir, export_tensorboard=True)
        assert hook.tensorboard_dir == os.path.join(hook.out_dir, "tensorboard")


def test_tensorboard_dir_script_specify_tensorboard_dir():
    """ In script mode, passing `export_tensorboard` and `tensorboard_dir` works. """
    with ScriptSimulator(tensorboard_dir="/tmp/tensorboard_dir") as sim:
        hook = smd.Hook(
            out_dir=sim.out_dir, export_tensorboard=True, tensorboard_dir=sim.tensorboard_dir
        )
        assert hook.tensorboard_dir == sim.tensorboard_dir


def test_tensorboard_dir_non_sagemaker_forgot_export_tensorboard():
    """ In script mode, passing tensorboard_dir will work. """
    with ScriptSimulator(tensorboard_dir="/tmp/tensorboard_dir") as sim:
        hook = smd.Hook(out_dir=sim.out_dir, tensorboard_dir=sim.tensorboard_dir)
        assert hook.tensorboard_dir == sim.tensorboard_dir


def test_temp_paths():
    with SagemakerSimulator() as sim:
        for path in [
            "/opt/ml/output/tensors/events/a",
            "/opt/ml/output/tensors/a",
            "/opt/ml/output/tensors/events/a/b",
        ]:
            temp_path = get_temp_path(path)
            assert temp_path.endswith(SAGEMAKER_TEMP_PATH_SUFFIX)
            assert not temp_path.startswith(NON_SAGEMAKER_TEMP_PATH_PREFIX)

    with ScriptSimulator() as sim:
        for path in ["/a/b/c", "/opt/ml/output/a", "a/b/c"]:
            temp_path = get_temp_path(path)
            assert not SAGEMAKER_TEMP_PATH_SUFFIX in temp_path
            assert temp_path.startswith(NON_SAGEMAKER_TEMP_PATH_PREFIX)


def test_s3_path_that_exists_without_end_of_job():
    path = "s3://tornasole-testing/s3-path-without-end-of-job"
    verify_and_get_out_dir(path)
    try:
        verify_and_get_out_dir(path)
        # should not raise as dir present but does not have the end of job file
    except RuntimeError as e:
        assert False
