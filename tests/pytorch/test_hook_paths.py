# Standard Library
import os

# First Party
import smdebug.pytorch as smd
from smdebug.core.utils import SagemakerSimulator, ScriptSimulator


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
