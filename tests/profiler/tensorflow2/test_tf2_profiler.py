# Standard Library
import os

# Third Party
import pytest
from tests.tensorflow2.test_keras import helper_keras_fit

# First Party
from smdebug.profiler.profiler_config_parser import ProfilerConfigParser
from smdebug.profiler.tf_profiler_parser import TensorboardProfilerEvents
from smdebug.tensorflow import KerasHook as Hook


@pytest.fixture()
def tf2_profiler_config_parser_by_step(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "test_tf2_profiler_config_parser_by_step.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser()


@pytest.fixture()
def tf2_profiler_config_parser_by_time(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "test_tf2_profiler_config_parser_by_time.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser()


def test_tf2_profiler_by_step(tf2_profiler_config_parser_by_step, out_dir):
    """
    This test executes a TF2 training script, enables detailed TF profiling by step, and
    verifies the number of events.
    """
    assert tf2_profiler_config_parser_by_step.profiling_enabled

    hook = Hook(out_dir=out_dir)
    helper_keras_fit(
        trial_dir=out_dir, hook=hook, eager=True, steps=["train", "eval", "predict", "train"]
    )
    hook.close()

    t_events = TensorboardProfilerEvents()

    log_dir = os.path.join(
        tf2_profiler_config_parser_by_step.config.local_path,
        "framework",
        "tensorflow",
        "detailed_profiling",
    )

    for root, _, _ in os.walk(log_dir, topdown=True):
        # walking through log_dir to find the folder that's 2 levels below
        # this is being done to get the parent of plugins/profile
        if root.count(os.path.sep) - log_dir.count(os.path.sep) == 2:
            trace_json_file = t_events._get_trace_file(root)

            t_events.read_events_from_file(trace_json_file)

            all_trace_events = t_events.get_all_events()
            num_trace_events = len(all_trace_events)

            print(f"Number of events read = {num_trace_events}")

            # The number of events is varying by a small number on
            # consecutive runs. Hence, the approximation in the below asserts.
            assert pytest.approx(249, 3) == num_trace_events


def test_tf2_profiler_by_time(tf2_profiler_config_parser_by_time, out_dir):
    """
    This test executes a TF2 training script, enables detailed TF profiling by time, and
    verifies the number of events.
    """
    assert tf2_profiler_config_parser_by_time.profiling_enabled

    hook = Hook(out_dir=out_dir)
    helper_keras_fit(
        trial_dir=out_dir, hook=hook, eager=True, steps=["train", "eval", "predict", "train"]
    )
    hook.close()

    t_events = TensorboardProfilerEvents()

    log_dir = os.path.join(
        tf2_profiler_config_parser_by_time.config.local_path,
        "framework",
        "tensorflow",
        "detailed_profiling",
    )

    count_step_folders = 0
    for root, _, _ in os.walk(log_dir, topdown=True):
        # walking through log_dir to find the folder that's 2 levels below
        # this is being done to get the parent of plugins/profile
        if root.count(os.path.sep) - log_dir.count(os.path.sep) == 2:
            count_step_folders += 1
            trace_json_file = t_events._get_trace_file(root)

            t_events.read_events_from_file(trace_json_file)

            all_trace_events = t_events.get_all_events()
            num_trace_events = len(all_trace_events)

            print(f"Number of events read = {num_trace_events}")

            # The number of events is varying by a small number on
            # consecutive runs. Hence, the approximation in the below asserts.
            if count_step_folders == 1:
                assert pytest.approx(1874, 5) == num_trace_events
            elif count_step_folders == 2:
                assert pytest.approx(3140, 5) == num_trace_events

    # folder created for step 0 and step 10
    assert count_step_folders == 2
