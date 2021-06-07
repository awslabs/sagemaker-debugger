# Third Party
import pytest
import tensorflow.compat.v2 as tf
from tests.zero_code_change.smdataparallel_tests.constants import (
    SMDATAPARALLEL_TF2_TEST_MNIST_SCRIPT,
)
from tests.zero_code_change.smdataparallel_tests.utils import launch_smdataparallel_job
from tests.zero_code_change.tf_utils import get_available_gpus
from tests.zero_code_change.utils import build_json

# First Party
from smdebug.tensorflow.constants import TF_DEFAULT_SAVED_COLLECTIONS
from smdebug.trials import create_trial


def basic_test(out_dir, mode):
    path = build_json(
        out_dir, include_workers="one", include_collections=["weights", "optimizer_variables"]
    )
    num_workers = len(get_available_gpus())
    mode_args = ["--model_dir", out_dir]
    launch_smdataparallel_job(
        script_file_path=SMDATAPARALLEL_TF2_TEST_MNIST_SCRIPT,
        script_args=mode_args,
        num_workers=num_workers,
        config_file_path=path,
        mode=mode,
    )

    tr = create_trial(out_dir)
    print(tr.tensor_names())
    assert len(tr.workers()) == 1
    assert len(tr.tensor_names()) == 5
    assert len(tr.tensor(tr.tensor_names(collection="weights")[0]).workers(0)) == 1


@pytest.mark.skipif(
    tf.__version__ < "2.3.0",
    reason="smdistributed.dataparallel supports TF version 2.3.0 and above",
)
@pytest.mark.skip(
    reason="Requires SMDataParallel docker image which is private as of now. It would be available in general DLC sometime in mid of November 2020"
)
def test_gpu(out_dir):
    basic_test(out_dir, "gpu")


def mode_allworkers(out_dir, mode):
    path = build_json(
        out_dir, include_workers="all", include_collections=["weights", "optimizer_variables"]
    )
    num_workers = len(get_available_gpus())
    mode_args = ["--model_dir", out_dir]
    launch_smdataparallel_job(
        script_file_path=SMDATAPARALLEL_TF2_TEST_MNIST_SCRIPT,
        script_args=mode_args,
        num_workers=num_workers,
        config_file_path=path,
        mode=mode,
    )
    tr = create_trial(out_dir)
    assert len(tr.workers()) == num_workers
    print("tensor names: ", tr.tensor_names())
    assert len(tr.tensor_names()) == 5
    assert len(tr.tensor(tr.tensor_names(collection="weights")[0]).workers(0)) == num_workers


@pytest.mark.skipif(
    tf.__version__ < "2.3.0",
    reason="smdistributed.dataparallel supports TF version 2.3.0 and above",
)
@pytest.mark.skip(
    reason="Requires SMDataParallel docker image which is private as of now. It would be available in general DLC sometime in mid of November 2020"
)
def test_gpu_allworkers(out_dir):
    mode_allworkers(out_dir, "gpu")


def mode_allworkers_saveall(out_dir, mode):
    path = build_json(out_dir, include_workers="all", save_all=True)
    num_workers = len(get_available_gpus())
    mode_args = ["--model_dir", out_dir]
    launch_smdataparallel_job(
        script_file_path=SMDATAPARALLEL_TF2_TEST_MNIST_SCRIPT,
        script_args=mode_args,
        num_workers=num_workers,
        config_file_path=path,
        mode=mode,
    )
    tr = create_trial(out_dir)
    assert len(tr.workers()) == num_workers
    assert len(tr.tensor_names()) == 35
    assert len(tr.tensor(tr.tensor_names(collection="weights")[0]).workers(0)) == num_workers
    assert len(tr.tensor("loss").workers(0)) == num_workers


@pytest.mark.skipif(
    tf.__version__ < "2.3.0",
    reason="smdistributed.dataparallel supports TF version 2.3.0 and above",
)
@pytest.mark.skip(
    reason="Requires SMDataParallel docker image which is private as of now. It would be available in general DLC sometime in mid of November 2020"
)
def test_gpu_allworkers_saveall(out_dir):
    mode_allworkers_saveall(out_dir, "gpu")


def mode_allworkers_default_collections(out_dir, mode):
    path = build_json(
        out_dir, include_workers="all", include_collections=TF_DEFAULT_SAVED_COLLECTIONS
    )
    num_workers = len(get_available_gpus())
    mode_args = ["--model_dir", out_dir]
    launch_smdataparallel_job(
        script_file_path=SMDATAPARALLEL_TF2_TEST_MNIST_SCRIPT,
        script_args=mode_args,
        num_workers=num_workers,
        config_file_path=path,
        mode=mode,
    )
    tr = create_trial(out_dir)
    assert len(tr.workers()) == num_workers
    assert len(tr.tensor_names()) == 1
    assert len(tr.tensor(tr.tensor_names(collection="losses")[0]).workers(0)) == num_workers


@pytest.mark.skipif(
    tf.__version__ < "2.3.0",
    reason="smdistributed.dataparallel supports TF version 2.3.0 and above",
)
@pytest.mark.skip(
    reason="Requires SMDataParallel docker image which is private as of now. It would be available in general DLC sometime in mid of November 2020"
)
def test_gpu_allworkers_default_collections(out_dir):
    mode_allworkers_default_collections(out_dir, "gpu")
