# Standard Library
import json
import os

# Third Party
import pytest
from tests.profiler.resources.profiler_config_parser_utils import (
    current_step,
    current_time,
    detailed_profiling_test_cases,
)

# First Party
from smdebug.profiler.profiler_config_parser import ProfilerConfigParser
from smdebug.profiler.profiler_constants import (
    CLOSE_FILE_INTERVAL_DEFAULT,
    FILE_OPEN_FAIL_THRESHOLD_DEFAULT,
    MAX_FILE_SIZE_DEFAULT,
)


@pytest.fixture
def detailed_profiler_config_path(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "detailed_profiler_config.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    yield config_path
    if os.path.isfile(config_path):
        os.remove(config_path)


@pytest.fixture
def missing_config_profiler_config_parser(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "missing_profile_config_parser.json")  # doesn't exist
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser()


@pytest.fixture
def user_disabled_profiler_config_parser(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "user_disabled_profile_config_parser.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser()


@pytest.fixture
def string_data_profiler_config_parser(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "string_data_profiler_config_parser.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser()


@pytest.fixture
def invalid_string_data_profiler_config_parser(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "invalid_string_data_profiler_config_parser.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser()


@pytest.fixture
def case_insensitive_profiler_config_parser(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "case_insensitive_profiler_config_parser.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser()


def _convert_to_string(item):
    return '"{0}"'.format(item) if isinstance(item, str) else item


def _convert_key_and_value(key, value):
    return "{0}: {1}, ".format(_convert_to_string(key), _convert_to_string(value))


@pytest.mark.parametrize("test_case", detailed_profiling_test_cases)
def test_profiling_ranges(detailed_profiler_config_path, test_case):
    detailed_profiling_parameters, expected_detailed_profiling_enabled, expected_can_detailed_profile, expected_values = (
        test_case
    )
    start_step, num_steps, start_time, duration = detailed_profiling_parameters
    detailed_profiler_config = "{"
    if start_step:
        detailed_profiler_config += _convert_key_and_value("StartStep", start_step)
    if num_steps:
        detailed_profiler_config += _convert_key_and_value("NumSteps", num_steps)
    if start_time:
        detailed_profiler_config += _convert_key_and_value("StartTimeInSecSinceEpoch", start_time)
    if duration:
        detailed_profiler_config += _convert_key_and_value("DurationInSeconds", duration)
    detailed_profiler_config += "}"

    full_config = {
        "ProfilingParameters": {
            "ProfilerEnabled": True,
            "DetailedProfilingConfig": detailed_profiler_config,
        }
    }

    with open(detailed_profiler_config_path, "w") as f:
        json.dump(full_config, f)

    profiler_config_parser = ProfilerConfigParser()
    assert profiler_config_parser.profiling_enabled
    assert profiler_config_parser.detailed_profiling_enabled == expected_detailed_profiling_enabled

    if profiler_config_parser.detailed_profiling_enabled:
        profile_range = profiler_config_parser.config.profile_range
        assert (
            profile_range.can_start_detailed_profiling(current_step, current_time)
            == expected_can_detailed_profile
        )
        expected_start_step, expected_end_step, expected_start_time, expected_end_time = (
            expected_values
        )
        assert profile_range.start_step == expected_start_step
        assert profile_range.end_step == expected_end_step
        assert profile_range.start_time_in_sec == expected_start_time
        assert profile_range.end_time == expected_end_time


def test_disabled_profiler(
    missing_config_profiler_config_parser, user_disabled_profiler_config_parser
):
    """
    This test is meant to test that a missing config file or the user setting `ProfilerEnabled`
    to `false` will disable the profiler.
    """
    assert not missing_config_profiler_config_parser.profiling_enabled
    assert not user_disabled_profiler_config_parser.profiling_enabled


def test_default_values(simple_profiler_config_parser):
    """
    This test is meant to test setting default values when the config is present.
    """
    assert simple_profiler_config_parser.profiling_enabled

    trace_file_config = simple_profiler_config_parser.config.trace_file
    assert trace_file_config.file_open_fail_threshold == FILE_OPEN_FAIL_THRESHOLD_DEFAULT

    rotation_policy = trace_file_config.rotation_policy
    assert rotation_policy.file_max_size == MAX_FILE_SIZE_DEFAULT
    assert rotation_policy.file_close_interval == CLOSE_FILE_INTERVAL_DEFAULT

    profile_range = simple_profiler_config_parser.config.profile_range
    assert not any(
        [
            profile_range.start_step,
            profile_range.num_steps,
            profile_range.start_time_in_sec,
            profile_range.duration_in_sec,
        ]
    )


def test_string_data_in_config(string_data_profiler_config_parser):
    """
    This test is meant to test that the profiler config parser can handle string data
    and typecast to appropriate types before use.
    """
    assert string_data_profiler_config_parser.profiling_enabled

    assert isinstance(
        string_data_profiler_config_parser.config.trace_file.rotation_policy.file_max_size, int
    )
    assert isinstance(
        string_data_profiler_config_parser.config.trace_file.rotation_policy.file_close_interval,
        float,
    )
    assert isinstance(
        string_data_profiler_config_parser.config.trace_file.file_open_fail_threshold, int
    )

    assert isinstance(string_data_profiler_config_parser.config.profile_range.start_step, int)
    assert isinstance(string_data_profiler_config_parser.config.profile_range.num_steps, int)
    assert isinstance(
        string_data_profiler_config_parser.config.profile_range.start_time_in_sec, float
    )
    assert isinstance(
        string_data_profiler_config_parser.config.profile_range.duration_in_sec, float
    )


def test_invalid_string_data_in_config(invalid_string_data_profiler_config_parser):
    """
    This test is meant to test that the profiler config parser can handle invalid string data
    and fallback gracefully.
    """
    # Profiler is enabled even if data is invalid
    assert invalid_string_data_profiler_config_parser.profiling_enabled

    # Fallback to default values for profiling parameters
    assert (
        invalid_string_data_profiler_config_parser.config.trace_file.rotation_policy.file_max_size
        == MAX_FILE_SIZE_DEFAULT
    )
    assert (
        invalid_string_data_profiler_config_parser.config.trace_file.rotation_policy.file_close_interval
        == CLOSE_FILE_INTERVAL_DEFAULT
    )
    assert (
        invalid_string_data_profiler_config_parser.config.trace_file.file_open_fail_threshold
        == FILE_OPEN_FAIL_THRESHOLD_DEFAULT
    )

    # Disable detailed profiling config if any of the fields are invalid
    assert not invalid_string_data_profiler_config_parser.detailed_profiling_enabled

    assert not invalid_string_data_profiler_config_parser.config.profile_range.start_step
    assert not invalid_string_data_profiler_config_parser.config.profile_range.num_steps
    assert not invalid_string_data_profiler_config_parser.config.profile_range.start_time_in_sec
    assert not invalid_string_data_profiler_config_parser.config.profile_range.duration_in_sec


def test_case_insensitive_profiler_config_parser(case_insensitive_profiler_config_parser):
    """
    This test is meant to test that the keys in the profiler config JSON are case insensitive. In other words,
    the profiler config parser can successfully parse the values from the config even if the case of the key is not
    camel case.
    """
    assert case_insensitive_profiler_config_parser.profiling_enabled
    assert (
        case_insensitive_profiler_config_parser.config.trace_file.rotation_policy.file_max_size
        == 100
    )
    assert (
        case_insensitive_profiler_config_parser.config.trace_file.rotation_policy.file_close_interval
        == 1
    )
    assert case_insensitive_profiler_config_parser.config.trace_file.file_open_fail_threshold == 5
    assert case_insensitive_profiler_config_parser.config.use_pyinstrument is True

    assert case_insensitive_profiler_config_parser.detailed_profiling_enabled
    assert case_insensitive_profiler_config_parser.config.profile_range.start_step == 2
    assert case_insensitive_profiler_config_parser.config.profile_range.num_steps == 3
