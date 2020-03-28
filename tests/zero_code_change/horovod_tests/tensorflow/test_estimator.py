# Standard Library
import os

# Third Party
from tests.zero_code_change.utils import build_json

# First Party
from smdebug.trials import create_trial

# Local
from .utils import get_available_gpus, launch_horovod_job

"""
Tested on current DLAMI p3.8xlarge when run from the main directory
"""

HOROVOD_MNIST_SCRIPT_NAME = "horovod_estimator_mnist.py"
HOROVOD_MNIST_ARGS = ["--num_steps", "1000"]


def basic_test(out_dir, mode):
    path = build_json(out_dir, include_workers="one", include_collections=["weights", "gradients"])
    num_workers = len(get_available_gpus())
    mode_args = list(HOROVOD_MNIST_ARGS) + ["--model_dir", os.path.join(out_dir, "checkpoint")]
    if mode == "cpu":
        mode_args += ["--use_only_cpu", "true"]
    launch_horovod_job(
        script_file_path=f"examples/tensorflow/sagemaker_official_container/{HOROVOD_MNIST_SCRIPT_NAME}",
        script_args=mode_args,
        num_workers=num_workers,
        config_file_path=path,
        mode=mode,
    )

    tr = create_trial(out_dir)
    print(tr.tensor_names())
    assert len(tr.workers()) == 1
    assert len(tr.tensor_names()) == 13
    assert len(tr.tensor(tr.tensor_names(collection="weights")[0]).workers(0)) == 1


def test_cpu(out_dir):
    basic_test(out_dir, "cpu")


def test_gpu(out_dir):
    basic_test(out_dir, "gpu")


def mode_allworkers(out_dir, mode):
    path = build_json(out_dir, include_workers="all", include_collections=["weights", "gradients"])
    num_workers = len(get_available_gpus())
    mode_args = list(HOROVOD_MNIST_ARGS) + ["--model_dir", os.path.join(out_dir, "checkpoint")]
    if mode == "cpu":
        mode_args += ["--use_only_cpu", "true"]
    launch_horovod_job(
        script_file_path=f"examples/tensorflow/sagemaker_official_container/{HOROVOD_MNIST_SCRIPT_NAME}",
        script_args=mode_args,
        num_workers=num_workers,
        config_file_path=path,
        mode=mode,
    )
    tr = create_trial(out_dir)
    assert len(tr.workers()) == num_workers
    assert len(tr.tensor_names()) == 13
    assert len(tr.tensor(tr.tensor_names(collection="weights")[0]).workers(0)) == num_workers


def test_cpu_allworkers(out_dir):
    mode_allworkers(out_dir, "cpu")


def test_gpu_allworkers(out_dir):
    mode_allworkers(out_dir, "gpu")


def mode_allworkers_saveall(out_dir, mode):
    path = build_json(
        out_dir, include_workers="all", save_all=True, include_collections=["weights", "gradients"]
    )
    num_workers = len(get_available_gpus())
    mode_args = list(HOROVOD_MNIST_ARGS) + ["--model_dir", os.path.join(out_dir, "checkpoint")]
    if mode == "cpu":
        mode_args += ["--use_only_cpu", "true"]
    launch_horovod_job(
        script_file_path=f"examples/tensorflow/sagemaker_official_container/{HOROVOD_MNIST_SCRIPT_NAME}",
        script_args=mode_args,
        num_workers=num_workers,
        config_file_path=path,
        mode=mode,
    )
    tr = create_trial(out_dir)
    assert len(tr.workers()) == num_workers
    assert len(tr.tensor_names()) > 99
    assert len(tr.tensor(tr.tensor_names(collection="weights")[0]).workers(0)) == num_workers
    assert len(tr.tensor(tr.tensor_names(collection="losses")[0]).workers(0)) == num_workers


def test_gpu_allworkers_saveall(out_dir):
    mode_allworkers_saveall(out_dir, "gpu")


def test_cpu_allworkers_saveall(out_dir):
    mode_allworkers_saveall(out_dir, "cpu")
