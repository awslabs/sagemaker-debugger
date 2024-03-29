# First Party

# Standard Library
import json
import os
from collections import defaultdict
from pathlib import Path

# Third Party
import pytest
from tests.tensorflow2.utils import is_greater_than_tf_2_2
from tests.zero_code_change.horovod_tests.constants import (
    HOROVOD_KERAS_TEST_SCRIPT_ARGS,
    HOROVOD_TF2_TEST_MNIST_SCRIPT,
)
from tests.zero_code_change.horovod_tests.tf_utils import get_available_gpus
from tests.zero_code_change.horovod_tests.utils import launch_horovod_job
from tests.zero_code_change.utils import build_json

from smdebug.profiler.profiler_config_parser import ProfilerConfigParser
from smdebug.profiler.profiler_constants import DEFAULT_PREFIX, HOROVODTIMELINE_SUFFIX
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
    assert len(tr.tensor_names()) == (13 if is_greater_than_tf_2_2() else 14)
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
    assert len(tr.tensor_names()) == (13 if is_greater_than_tf_2_2() else 14)
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
    assert len(tr.tensor_names()) == (17 if is_greater_than_tf_2_2() else 18)
    assert len(tr.tensor(tr.tensor_names(collection="weights")[0]).workers(0)) == num_workers
    assert len(tr.tensor("loss").workers(0)) == num_workers


def test_gpu_allworkers_saveall(out_dir):
    mode_allworkers_saveall(out_dir, "gpu")


def test_cpu_allworkers_saveall(out_dir):
    mode_allworkers_saveall(out_dir, "cpu")


"""
HVD event file rotation tests
"""


@pytest.fixture
def hvd_rotation_profiler_config_parser(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "hvd_rotation_profiler_config_parser.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser()


@pytest.mark.parametrize("mode", ["cpu", "gpu"])
@pytest.mark.parametrize("worker_function", [basic_test, mode_allworkers])
def test_mode_workers_event_file_rotation(
    out_dir, monkeypatch, hvd_rotation_profiler_config_parser, mode, worker_function
):
    """
        This test is meant to verify the working of the Horovod trace file reader with
        Horovod and Tensorflow 2.x.
        :param hvd_rotation_profiler_config_parser: Profiler Config Parser
        :param mode: cpu or gpu
        :param worker_function: basic test (one worker) or all workers
    """
    # check if Profiler config has been parsed and is enabled
    assert hvd_rotation_profiler_config_parser.profiling_enabled

    # enable Horovod timeline
    hvd_file = out_dir + "/hvd_timeline.json"
    monkeypatch.setenv("HOROVOD_TIMELINE", hvd_file)

    # start the training job
    worker_function(out_dir, mode)

    # check if files have been split
    files = []
    for path in Path(out_dir + "/" + DEFAULT_PREFIX).rglob(f"*{HOROVODTIMELINE_SUFFIX}"):
        files.append(path)

    assert files

    # after the training completes, read the Horovod timeline file and make note of all
    # events. This will be used to verify reader functionality.
    json_dict = []
    with open(hvd_file) as json_data:
        for line in json_data:
            try:
                event = json.loads(line[:-2]) if line.endswith(",\n") else json.loads(line[:-1])
                json_dict.append(event)
            except Exception as e:
                # if JSON string is invalid, skip
                pass

    # populate the horovod event dictionary
    # key = phase, value = list of events of that particular phase
    hvd_event_dict = defaultdict(list)
    for event in json_dict:
        hvd_event_dict[event["ph"]].append(event)

    # populate the events that were written by the timeline file writer
    # in the rotated files. These events are the ones that the trace file reader
    # thread read from the large Horovod timeline file.
    rotated_event_dict = defaultdict(list)
    for file_name in files:
        with open(file_name) as timeline_file:
            events_dict = json.load(timeline_file)
            for idx, e in enumerate(events_dict):
                if idx < 2:
                    # skip the 1st 2 events of each file as they are
                    # metadata filled by the timeline file writer
                    continue
                rotated_event_dict[e["ph"]].append(e)

                # check if all timestamps are positive
                if "ts" in e:
                    assert int(e["ts"]) >= 0

    # check that the rotated files have the same number of event types
    # as the original horovod timeline file
    assert len(rotated_event_dict) == len(hvd_event_dict)

    # for every type of event, check that the rotated files have at least
    # the same number of events of that type in the hvd file.
    # We check for the condition >= because metadata events in rotated files
    # could be more in number as metadata could be repeated in multiple files
    # based on the time of rotation.
    for key in hvd_event_dict:
        assert len(rotated_event_dict[key]) >= len(hvd_event_dict[key])
