# First Party
# Third Party
from tests.tensorflow2.utils import is_tf_2_2
from tests.zero_code_change.horovod_tests.constants import (
    HOROVOD_KERAS_TEST_SCRIPT_ARGS,
    HOROVOD_TF2_TEST_MNIST_SCRIPT,
)
from tests.zero_code_change.horovod_tests.tf_utils import get_available_gpus
from tests.zero_code_change.horovod_tests.utils import launch_horovod_job
from tests.zero_code_change.utils import build_json

from smdebug.trials import create_trial


def basic_test(out_dir, mode):
    path = build_json(
        out_dir, include_workers="one", include_collections=["weights", "optimizer_variables"]
    )
    num_workers = len(get_available_gpus())
    mode_args = list(HOROVOD_KERAS_TEST_SCRIPT_ARGS) + ["--model_dir", out_dir]
    if mode == "cpu":
        mode_args += ["--use_only_cpu", "true"]
    launch_horovod_job(
        script_file_path=HOROVOD_TF2_TEST_MNIST_SCRIPT,
        script_args=mode_args,
        num_workers=num_workers,
        config_file_path=path,
        mode=mode,
    )

    tr = create_trial(out_dir)
    print(tr.tensor_names())
    assert len(tr.workers()) == 1
    assert len(tr.tensor_names()) == (13 if is_tf_2_2() else 14)
    assert len(tr.tensor(tr.tensor_names(collection="weights")[0]).workers(0)) == 1


def test_cpu(out_dir):
    basic_test(out_dir, "cpu")


def test_gpu(out_dir):
    basic_test(out_dir, "gpu")


def mode_allworkers(out_dir, mode):
    path = build_json(
        out_dir, include_workers="all", include_collections=["weights", "optimizer_variables"]
    )
    num_workers = len(get_available_gpus())
    mode_args = list(HOROVOD_KERAS_TEST_SCRIPT_ARGS) + ["--model_dir", out_dir]
    if mode == "cpu":
        mode_args += ["--use_only_cpu", "true"]
    launch_horovod_job(
        script_file_path=HOROVOD_TF2_TEST_MNIST_SCRIPT,
        script_args=mode_args,
        num_workers=num_workers,
        config_file_path=path,
        mode=mode,
    )
    tr = create_trial(out_dir)
    assert len(tr.workers()) == num_workers
    assert len(tr.tensor_names()) == (13 if is_tf_2_2() else 14)
    assert len(tr.tensor(tr.tensor_names(collection="weights")[0]).workers(0)) == num_workers


def test_cpu_allworkers(out_dir):
    mode_allworkers(out_dir, "cpu")


def test_gpu_allworkers(out_dir):
    mode_allworkers(out_dir, "gpu")


def mode_allworkers_saveall(out_dir, mode):
    path = build_json(
        out_dir,
        include_workers="all",
        save_all=True,
        include_collections=["weights", "optimizer_variables"],
    )
    num_workers = len(get_available_gpus())
    mode_args = list(HOROVOD_KERAS_TEST_SCRIPT_ARGS) + ["--model_dir", out_dir]
    if mode == "cpu":
        mode_args += ["--use_only_cpu", "true"]
    launch_horovod_job(
        script_file_path=HOROVOD_TF2_TEST_MNIST_SCRIPT,
        script_args=mode_args,
        num_workers=num_workers,
        config_file_path=path,
        mode=mode,
    )
    tr = create_trial(out_dir)
    assert len(tr.workers()) == num_workers
    assert len(tr.tensor_names()) == (17 if is_tf_2_2() else 18)
    assert len(tr.tensor(tr.tensor_names(collection="weights")[0]).workers(0)) == num_workers
    assert len(tr.tensor("loss").workers(0)) == num_workers


def test_gpu_allworkers_saveall(out_dir):
    mode_allworkers_saveall(out_dir, "gpu")


def test_cpu_allworkers_saveall(out_dir):
    mode_allworkers_saveall(out_dir, "cpu")
