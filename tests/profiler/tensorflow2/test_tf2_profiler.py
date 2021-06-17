# Standard Library
import os

# Third Party
import pytest
from tests.tensorflow2.test_keras import helper_keras_fit

# First Party
from smdebug.profiler.profiler_config_parser import ProfilerConfigParser
from smdebug.tensorflow import KerasHook as Hook

# Local
from .utils import verify_detailed_profiling


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


def test_tf2_profiler_by_step(set_up_resource_config, tf2_profiler_config_parser_by_step, out_dir):
    """
    This test executes a TF2 training script, enables detailed TF profiling by step, and
    verifies the number of events.
    """
    assert tf2_profiler_config_parser_by_step.profiling_enabled

    hook = Hook(out_dir=out_dir)
    helper_keras_fit(trial_dir=out_dir, hook=hook, eager=True, steps=["train", "eval", "predict"])
    hook.close()

    verify_detailed_profiling(out_dir, 230)


def test_tf2_profiler_by_time(tf2_profiler_config_parser_by_time, out_dir):
    """
    This test executes a TF2 training script, enables detailed TF profiling by time, and
    verifies the number of events.
    """
    assert tf2_profiler_config_parser_by_time.profiling_enabled

    hook = Hook(out_dir=out_dir)
    helper_keras_fit(trial_dir=out_dir, hook=hook, eager=True, steps=["train", "eval", "predict"])
    hook.close()

    verify_detailed_profiling(out_dir, 700)
