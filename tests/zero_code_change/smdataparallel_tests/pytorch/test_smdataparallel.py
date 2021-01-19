# Standard Library

import json
import os
from pathlib import Path

# Third Party
import pytest
from tests.zero_code_change.smdataparallel_tests.constants import (
    SMDATAPARALLEL_PYTORCH_TEST_MNIST_ARGS,
    SMDATAPARALLEL_PYTORCH_TEST_MNIST_SCRIPT,
)
from tests.zero_code_change.smdataparallel_tests.utils import launch_smdataparallel_job
from tests.zero_code_change.utils import build_json
from torch.cuda import device_count

# First Party
from smdebug.profiler.profiler_config_parser import ProfilerConfigParser
from smdebug.profiler.profiler_constants import DEFAULT_PREFIX, SMDATAPARALLELTIMELINE_SUFFIX
from smdebug.profiler.tf_profiler_parser import SMDataParallelProfilerEvents
from smdebug.trials import create_trial


"""
Tested on current DLAMI p3.16xlarge when run from the main directory
"""


def mode_allworkers(out_dir, mode):
    path = build_json(out_dir, include_workers="all", include_collections=["weights", "gradients"])
    print("build_json_path: ", path)
    num_workers = 1 if bool(device_count()) is False else device_count()
    mode_args = list(SMDATAPARALLEL_PYTORCH_TEST_MNIST_ARGS)
    launch_smdataparallel_job(
        script_file_path=SMDATAPARALLEL_PYTORCH_TEST_MNIST_SCRIPT,
        script_args=mode_args,
        num_workers=num_workers,
        config_file_path=path,
        mode=mode,
    )
    tr = create_trial(out_dir)
    assert len(tr.workers()) == num_workers
    assert len(tr.tensor_names()) == 13
    assert len(tr.tensor(tr.tensor_names(collection="weights")[0]).workers(0)) == num_workers


@pytest.mark.skip(
    reason="Requires SMDataParallel docker image which is private as of now. It would be available in general DLC sometime in mid of November 2020"
)
def test_gpu_allworkers(out_dir):
    mode_allworkers(out_dir, "gpu")


@pytest.fixture
def smdataparallel_profiler_config_path(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "smdataparallel_profiler_config.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    yield config_path
    if os.path.isfile(config_path):
        os.remove(config_path)


@pytest.mark.skip(
    reason="Requires SMDataParallel docker image which is private as of now. It would be available in general DLC sometime in mid of November 2020"
)
@pytest.mark.parametrize("mode", ["gpu"])
@pytest.mark.parametrize("worker_function", [mode_allworkers])
def test_mode_workers_dynamic_smdataparallel_profiler(
    out_dir, smdataparallel_profiler_config_path, mode, worker_function
):
    """
        This test is meant to verify dynamically turning ON/OFF SMDataParallel profiler with PyTorch.
        :param mode: gpu
        :param worker_function: basic test all workers
    """

    def _convert_to_string(item):
        return '"{0}"'.format(item) if isinstance(item, str) else item

    def _convert_key_and_value(key, value):
        return "{0}: {1}, ".format(_convert_to_string(key), _convert_to_string(value))

    smdataparallel_profiler_config = "{"
    smdataparallel_profiler_config += _convert_key_and_value("StartStep", 2)
    smdataparallel_profiler_config += _convert_key_and_value("NumSteps", 1)
    smdataparallel_profiler_config += "}"

    full_config = {
        "ProfilingParameters": {
            "ProfilerEnabled": True,
            "SMDataparallelProfilingConfig": smdataparallel_profiler_config,
            "localpath": out_dir,
        }
    }

    with open(smdataparallel_profiler_config_path, "w") as f:
        json.dump(full_config, f)

    profiler_config_parser = ProfilerConfigParser()
    assert profiler_config_parser.profiling_enabled

    # start the training job
    worker_function(out_dir, mode)

    files = []
    for path in Path(out_dir + "/" + DEFAULT_PREFIX).rglob(f"*{SMDATAPARALLELTIMELINE_SUFFIX}"):
        files.append(path)

    assert len(files) == 8

    trace_file = str(files[0])
    t_events = SMDataParallelProfilerEvents()

    t_events.read_events_from_file(trace_file)

    all_trace_events = t_events.get_all_events()
    num_trace_events = len(all_trace_events)

    print(f"Number of events read = {num_trace_events}")

    # The number of events is varying by a small number on
    # consecutive runs. Hence, the approximation in the below asserts.
    assert num_trace_events >= 8


def mode_allworkers_saveall(out_dir, mode):
    path = build_json(out_dir, include_workers="all", save_all=True)
    num_workers = 1 if bool(device_count()) is False else device_count()
    mode_args = list(SMDATAPARALLEL_PYTORCH_TEST_MNIST_ARGS)
    launch_smdataparallel_job(
        script_file_path=SMDATAPARALLEL_PYTORCH_TEST_MNIST_SCRIPT,
        script_args=mode_args,
        num_workers=num_workers,
        config_file_path=path,
        mode=mode,
    )
    tr = create_trial(out_dir)
    assert len(tr.workers()) == num_workers
    assert len(tr.tensor_names()) > 25
    assert len(tr.tensor(tr.tensor_names(collection="weights")[0]).workers(0)) == num_workers
    assert len(tr.tensor(tr.tensor_names(collection="losses")[0]).workers(0)) == num_workers


@pytest.mark.skip(
    reason="Requires SMDataParallel docker image which is private as of now. It would be available in general DLC sometime in mid of November 2020"
)
def test_gpu_allworkers_saveall(out_dir):
    mode_allworkers_saveall(out_dir, "gpu")
