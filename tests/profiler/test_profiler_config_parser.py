# Standard Library
import os

# Third Party
import pytest

# First Party
from smdebug.profiler.profiler_config_parser import ProfilerConfigParser
from smdebug.profiler.profiler_constants import (
    CLOSE_FILE_INTERVAL_DEFAULT,
    DEFAULT_PREFIX,
    FILE_OPEN_FAIL_THRESHOLD_DEFAULT,
    MAX_FILE_SIZE_DEFAULT,
)


@pytest.fixture
def config_folder():
    """Path to folder used for storing different config artifacts for testing timeline writer and
    profiler config parser.
    """
    return "tests/core/json_configs"


@pytest.fixture
def current_step():
    return 1


@pytest.fixture()
def simple_profiler_config_parser(config_folder, monkeypatch, current_step):
    config_path = os.path.join(config_folder, "simple_profiler_config_parser.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser(current_step)


@pytest.fixture
def missing_config_profiler_config_parser(config_folder, monkeypatch, current_step):
    config_path = os.path.join(config_folder, "missing_profile_config_parser.json")  # doesn't exist
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser(current_step)


@pytest.fixture
def user_disabled_profiler_config_parser(config_folder, monkeypatch, current_step):
    config_path = os.path.join(config_folder, "user_disabled_profile_config_parser.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser(current_step)


def test_disabled_profiler(
    missing_config_profiler_config_parser, user_disabled_profiler_config_parser
):
    """
    This test is meant to test that a missing config file or the user setting `ProfilerEnabled`
    to `false` will disable the profiler.
    """
    assert not missing_config_profiler_config_parser.enabled
    assert not user_disabled_profiler_config_parser.enabled


def test_default_values(simple_profiler_config_parser):
    """
    This test is meant to test setting default values when the config is present.
    """
    assert simple_profiler_config_parser.enabled

    trace_file_config = simple_profiler_config_parser.config.trace_file
    assert trace_file_config.file_open_fail_threshold == FILE_OPEN_FAIL_THRESHOLD_DEFAULT

    rotation_policy = trace_file_config.rotation_policy
    assert rotation_policy.file_max_size == MAX_FILE_SIZE_DEFAULT
    assert rotation_policy.file_close_interval == CLOSE_FILE_INTERVAL_DEFAULT

    profile_range = simple_profiler_config_parser.config.profile_range
    assert not profile_range.profile_type
    assert not profile_range.profiler_start
    assert not profile_range.profiler_end
