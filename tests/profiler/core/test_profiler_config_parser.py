# Standard Library
import json
import os

# Third Party
import pytest
from tests.profiler.resources.profiler_config_parser_utils import (
    current_step,
    current_time,
    dataloader_test_cases,
    detailed_profiling_test_cases,
    general_profiling_test_cases,
    herring_profiling_test_cases,
    python_profiling_test_cases,
)

# First Party
from smdebug.profiler.profiler_config_parser import MetricsCategory, ProfilerConfigParser
from smdebug.profiler.profiler_constants import (
    CLOSE_FILE_INTERVAL_DEFAULT,
    DATALOADER_PROFILING_START_STEP_DEFAULT,
    DETAILED_PROFILING_START_STEP_DEFAULT,
    FILE_OPEN_FAIL_THRESHOLD_DEFAULT,
    MAX_FILE_SIZE_DEFAULT,
    PROFILING_NUM_STEPS_DEFAULT,
    PYTHON_PROFILING_NUM_STEPS_DEFAULT,
    PYTHON_PROFILING_START_STEP_DEFAULT,
)


@pytest.fixture
def general_profiler_config_path(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "general_profiler_config.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    yield config_path
    if os.path.isfile(config_path):
        os.remove(config_path)


@pytest.fixture
def detailed_profiler_config_path(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "detailed_profiler_config.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    yield config_path
    if os.path.isfile(config_path):
        os.remove(config_path)


@pytest.fixture
def dataloader_profiler_config_path(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "dataloader_profiler_config.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    yield config_path
    if os.path.isfile(config_path):
        os.remove(config_path)


@pytest.fixture
def python_profiler_config_path(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "python_profiler_config.json")
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


@pytest.fixture
def herring_profiler_config_path(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "herring_profiler_config.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    yield config_path
    if os.path.isfile(config_path):
        os.remove(config_path)


def _convert_to_string(item):
    return '"{0}"'.format(item) if isinstance(item, str) else item


def _convert_key_and_value(key, value):
    return "{0}: {1}, ".format(_convert_to_string(key), _convert_to_string(value))


@pytest.mark.parametrize("test_case", general_profiling_test_cases)
def test_general_profiling_ranges(general_profiler_config_path, test_case):
    profiling_parameters, expected_can_save, expected_values = test_case
    general_profiling_parameters, detailed_profiling_parameters = profiling_parameters

    general_start_step, general_num_steps = general_profiling_parameters
    general_metrics_config = "{"
    if general_start_step:
        general_metrics_config += _convert_key_and_value("StartStep", general_start_step)
    if general_num_steps:
        general_metrics_config += _convert_key_and_value("NumSteps", general_num_steps)
    general_metrics_config += "}"

    detailed_start_step, detailed_num_steps = detailed_profiling_parameters
    detailed_profiler_config = "{"
    if detailed_start_step:
        detailed_profiler_config += _convert_key_and_value("StartStep", detailed_start_step)
    if detailed_num_steps:
        detailed_profiler_config += _convert_key_and_value("NumSteps", detailed_num_steps)
    detailed_profiler_config += "}"

    full_config = {
        "ProfilingParameters": {
            "ProfilerEnabled": True,
            "GeneralMetricsConfig": general_metrics_config,
            "DetailedProfilingConfig": detailed_profiler_config,
        }
    }

    with open(general_profiler_config_path, "w") as f:
        json.dump(full_config, f)

    profiler_config_parser = ProfilerConfigParser()
    assert profiler_config_parser.profiling_enabled
    assert (
        profiler_config_parser.should_save_metrics(MetricsCategory.DETAILED_PROFILING, current_step)
        == expected_can_save
    )

    expected_general_values, expected_detailed_values = expected_values

    expected_general_start_step, expected_general_end_step = expected_general_values
    general_metrics_config = profiler_config_parser.config.general_metrics_config
    assert general_metrics_config.is_enabled()
    assert general_metrics_config.start_step == expected_general_start_step
    assert general_metrics_config.end_step == expected_general_end_step

    expected_detailed_start_step, expected_detailed_end_step = expected_detailed_values
    detailed_profiling_config = profiler_config_parser.config.detailed_profiling_config
    assert detailed_profiling_config.is_enabled()
    assert detailed_profiling_config.start_step == expected_detailed_start_step
    assert detailed_profiling_config.end_step == expected_detailed_end_step


@pytest.mark.parametrize("test_case", detailed_profiling_test_cases)
def test_detailed_profiling_ranges(detailed_profiler_config_path, test_case):
    profiling_parameters, expected_enabled, expected_can_save, expected_values = test_case
    start_step, num_steps, start_time, duration = profiling_parameters
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

    detailed_profiling_config = profiler_config_parser.config.detailed_profiling_config
    assert detailed_profiling_config.is_enabled() == expected_enabled
    assert (
        profiler_config_parser.should_save_metrics(
            MetricsCategory.DETAILED_PROFILING, current_step, current_time=current_time
        )
        == expected_can_save
    )

    expected_start_step, expected_end_step, expected_start_time, expected_end_time = expected_values
    assert detailed_profiling_config.start_step == expected_start_step
    assert detailed_profiling_config.end_step == expected_end_step
    assert detailed_profiling_config.start_time_in_sec == expected_start_time
    assert detailed_profiling_config.end_time == expected_end_time


@pytest.mark.parametrize("test_case", dataloader_test_cases)
def test_dataloader_profiling_ranges(detailed_profiler_config_path, test_case):
    profiling_parameters, expected_enabled, expected_can_save, expected_values = test_case
    start_step, metrics_regex, metrics_name = profiling_parameters
    dataloader_config = "{"
    if start_step:
        dataloader_config += _convert_key_and_value("StartStep", start_step)
    if metrics_regex:
        dataloader_config += _convert_key_and_value("MetricsRegex", metrics_regex)
    dataloader_config += "}"

    full_config = {
        "ProfilingParameters": {
            "ProfilerEnabled": True,
            "DataloaderMetricsConfig": dataloader_config,
        }
    }

    with open(detailed_profiler_config_path, "w") as f:
        json.dump(full_config, f)

    profiler_config_parser = ProfilerConfigParser()
    assert profiler_config_parser.profiling_enabled

    dataloader_metrics_config = profiler_config_parser.config.dataloader_metrics_config
    assert dataloader_metrics_config.is_enabled() == expected_enabled
    assert (
        profiler_config_parser.should_save_metrics(
            MetricsCategory.DATALOADER, current_step, metrics_name=metrics_name
        )
        == expected_can_save
    )

    expected_start_step, expected_end_step, expected_metrics_regex = expected_values
    assert dataloader_metrics_config.start_step == expected_start_step
    assert dataloader_metrics_config.end_step == expected_end_step
    assert dataloader_metrics_config.metrics_regex == expected_metrics_regex


@pytest.mark.parametrize("test_case", python_profiling_test_cases)
def test_python_profiling_ranges(python_profiler_config_path, test_case):
    profiling_parameters, expected_enabled, expected_can_save, expected_values = test_case
    start_step, num_steps, profiler_name, cprofile_timer = profiling_parameters
    python_profiler_config = "{"
    if start_step is not None:
        python_profiler_config += _convert_key_and_value("StartStep", start_step)
    if num_steps is not None:
        python_profiler_config += _convert_key_and_value("NumSteps", num_steps)
    if profiler_name is not None:
        python_profiler_config += _convert_key_and_value("ProfilerName", profiler_name)
    if cprofile_timer is not None:
        python_profiler_config += _convert_key_and_value("cProfileTimer", cprofile_timer)
    python_profiler_config += "}"

    full_config = {
        "ProfilingParameters": {
            "ProfilerEnabled": True,
            "PythonProfilingConfig": python_profiler_config,
        }
    }

    with open(python_profiler_config_path, "w") as f:
        json.dump(full_config, f)

    profiler_config_parser = ProfilerConfigParser()
    assert profiler_config_parser.profiling_enabled

    python_profiling_config = profiler_config_parser.config.python_profiling_config
    assert python_profiling_config.is_enabled() == expected_enabled
    assert (
        profiler_config_parser.should_save_metrics(MetricsCategory.PYTHON_PROFILING, current_step)
        == expected_can_save
    )

    expected_start_step, expected_end_step, expected_profiler_name, expected_cprofile_timer = (
        expected_values
    )
    assert python_profiling_config.start_step == expected_start_step
    assert python_profiling_config.end_step == expected_end_step
    assert python_profiling_config.profiler_name == expected_profiler_name
    assert python_profiling_config.cprofile_timer == expected_cprofile_timer


@pytest.mark.parametrize("test_case", herring_profiling_test_cases)
def test_herring_profiling_ranges(herring_profiler_config_path, test_case):
    profiling_parameters, expected_enabled, expected_can_save, expected_values = test_case
    start_step, num_steps = profiling_parameters

    herring_profiler_config = "{"
    if start_step:
        herring_profiler_config += _convert_key_and_value("StartStep", start_step)
    if num_steps:
        herring_profiler_config += _convert_key_and_value("NumSteps", num_steps)
    herring_profiler_config += "}"

    full_config = {
        "ProfilingParameters": {
            "ProfilerEnabled": True,
            "SMDataparallelProfilingConfig": herring_profiler_config,
        }
    }

    with open(herring_profiler_config_path, "w") as f:
        json.dump(full_config, f)

    profiler_config_parser = ProfilerConfigParser()
    assert profiler_config_parser.profiling_enabled

    herring_profiling_config = profiler_config_parser.config.herring_profiling_config
    assert herring_profiling_config.is_enabled() == expected_enabled
    assert (
        profiler_config_parser.should_save_metrics(
            MetricsCategory.HERRING_PROFILING, current_step, current_time=current_time
        )
        == expected_can_save
    )

    expected_start_step, expected_end_step = expected_values
    assert herring_profiling_config.start_step == expected_start_step
    assert herring_profiling_config.end_step == expected_end_step


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

    detailed_profiling_config = simple_profiler_config_parser.config.detailed_profiling_config
    assert detailed_profiling_config.start_step == DETAILED_PROFILING_START_STEP_DEFAULT
    assert detailed_profiling_config.num_steps == PROFILING_NUM_STEPS_DEFAULT

    dataloader_metrics_config = simple_profiler_config_parser.config.dataloader_metrics_config
    assert dataloader_metrics_config.start_step == DATALOADER_PROFILING_START_STEP_DEFAULT
    assert dataloader_metrics_config.num_steps == PROFILING_NUM_STEPS_DEFAULT

    python_profiling_config = simple_profiler_config_parser.config.python_profiling_config
    assert python_profiling_config.start_step == PYTHON_PROFILING_START_STEP_DEFAULT
    assert python_profiling_config.num_steps == PYTHON_PROFILING_NUM_STEPS_DEFAULT


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

    assert isinstance(
        string_data_profiler_config_parser.config.detailed_profiling_config.start_step, int
    )
    assert isinstance(
        string_data_profiler_config_parser.config.detailed_profiling_config.num_steps, int
    )
    assert isinstance(
        string_data_profiler_config_parser.config.detailed_profiling_config.start_time_in_sec, float
    )
    assert isinstance(
        string_data_profiler_config_parser.config.detailed_profiling_config.duration_in_sec, float
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
    assert (
        not invalid_string_data_profiler_config_parser.config.detailed_profiling_config.is_enabled()
    )

    assert (
        not invalid_string_data_profiler_config_parser.config.detailed_profiling_config.start_step
    )
    assert not invalid_string_data_profiler_config_parser.config.detailed_profiling_config.num_steps
    assert (
        not invalid_string_data_profiler_config_parser.config.detailed_profiling_config.start_time_in_sec
    )
    assert (
        not invalid_string_data_profiler_config_parser.config.detailed_profiling_config.duration_in_sec
    )


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

    assert case_insensitive_profiler_config_parser.config.detailed_profiling_config.is_enabled()
    assert case_insensitive_profiler_config_parser.config.detailed_profiling_config.start_step == 2
    assert case_insensitive_profiler_config_parser.config.detailed_profiling_config.num_steps == 3
