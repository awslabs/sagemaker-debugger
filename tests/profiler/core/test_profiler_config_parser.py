# Standard Library
import json
import os
import shutil
import time

# Third Party
import pytest
from tests.profiler.resources.profiler_config_parser_utils import (
    build_metrics_config,
    current_step,
    current_time,
    dataloader_test_cases,
    detailed_profiling_test_cases,
    python_profiling_test_cases,
    smdataparallel_profiling_test_cases,
)
from tests.profiler.utils import get_last_accessed_time

# First Party
from smdebug.profiler.profiler_config_parser import MetricsCategory, ProfilerConfigParser
from smdebug.profiler.profiler_constants import (
    CLOSE_FILE_INTERVAL_DEFAULT,
    FILE_OPEN_FAIL_THRESHOLD_DEFAULT,
    MAX_FILE_SIZE_DEFAULT,
    PROFILING_NUM_STEPS_DEFAULT,
)


@pytest.fixture
def profiler_config_path(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "profiler_config.json")
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
def invalid_config_profiler_config_parser(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "invalid_profile_config_parser.json")  # invalid JSON
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
def step_profiler_config_parser_path(config_folder):
    return os.path.join(config_folder, "step_profiler_config_parser.json")


@pytest.fixture
def old_step_profiler_config_parser_path(config_folder):
    return os.path.join(config_folder, "old_step_profiler_config_parser.json")


@pytest.fixture
def new_step_profiler_config_parser_path(config_folder):
    return os.path.join(config_folder, "new_step_profiler_config_parser.json")


@pytest.fixture
def time_profiler_config_parser_path(config_folder):
    return os.path.join(config_folder, "time_profiler_config_parser.json")


@pytest.fixture
def old_time_profiler_config_parser_path(config_folder):
    return os.path.join(config_folder, "old_time_profiler_config_parser.json")


@pytest.fixture
def new_time_profiler_config_parser_path(config_folder):
    return os.path.join(config_folder, "new_time_profiler_config_parser.json")


@pytest.mark.parametrize("test_case", detailed_profiling_test_cases)
def test_detailed_profiling_ranges(profiler_config_path, test_case):
    profiling_parameters, expected_enabled, expected_can_save, expected_values = test_case
    start_step, num_steps, start_time, duration = profiling_parameters
    detailed_profiling_config = build_metrics_config(
        StartStep=start_step,
        NumSteps=num_steps,
        StartTimeInSecSinceEpoch=start_time,
        DurationInSeconds=duration,
    )

    full_config = {
        "ProfilingParameters": {
            "ProfilerEnabled": True,
            "DetailedProfilingConfig": detailed_profiling_config,
        }
    }

    with open(profiler_config_path, "w") as f:
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
def test_dataloader_profiling_ranges(profiler_config_path, test_case):
    profiling_parameters, expected_enabled, expected_can_save, expected_values = test_case
    start_step, metrics_regex, metrics_name = profiling_parameters
    dataloader_profiling_config = build_metrics_config(
        StartStep=start_step, MetricsRegex=metrics_regex
    )

    full_config = {
        "ProfilingParameters": {
            "ProfilerEnabled": True,
            "DataloaderProfilingConfig": dataloader_profiling_config,
        }
    }

    with open(profiler_config_path, "w") as f:
        json.dump(full_config, f)

    profiler_config_parser = ProfilerConfigParser()
    assert profiler_config_parser.profiling_enabled

    dataloader_profiling_config = profiler_config_parser.config.dataloader_profiling_config
    assert dataloader_profiling_config.is_enabled() == expected_enabled
    assert (
        profiler_config_parser.should_save_metrics(
            MetricsCategory.DATALOADER_PROFILING, current_step, metrics_name=metrics_name
        )
        == expected_can_save
    )

    expected_start_step, expected_end_step, expected_metrics_regex = expected_values
    assert dataloader_profiling_config.start_step == expected_start_step
    assert dataloader_profiling_config.end_step == expected_end_step
    assert dataloader_profiling_config.metrics_regex == expected_metrics_regex


@pytest.mark.parametrize("test_case", python_profiling_test_cases)
def test_python_profiling_ranges(profiler_config_path, test_case):
    profiling_parameters, expected_enabled, expected_can_save, expected_values = test_case
    start_step, num_steps, profiler_name, cprofile_timer = profiling_parameters
    python_profiling_config = build_metrics_config(
        StartStep=start_step,
        NumSteps=num_steps,
        ProfilerName=profiler_name,
        cProfileTimer=cprofile_timer,
    )

    full_config = {
        "ProfilingParameters": {
            "ProfilerEnabled": True,
            "PythonProfilingConfig": python_profiling_config,
        }
    }

    with open(profiler_config_path, "w") as f:
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


@pytest.mark.parametrize("test_case", smdataparallel_profiling_test_cases)
def test_smdataparallel_profiling_ranges(profiler_config_path, test_case):
    profiling_parameters, expected_enabled, expected_can_save, expected_values = test_case
    start_step, num_steps = profiling_parameters

    smdataparallel_profiling_config = build_metrics_config(StartStep=start_step, NumSteps=num_steps)

    full_config = {
        "ProfilingParameters": {
            "ProfilerEnabled": True,
            "SMDataparallelProfilingConfig": smdataparallel_profiling_config,
        }
    }

    with open(profiler_config_path, "w") as f:
        json.dump(full_config, f)

    profiler_config_parser = ProfilerConfigParser()
    assert profiler_config_parser.profiling_enabled

    smdataparallel_profiling_config = profiler_config_parser.config.smdataparallel_profiling_config
    assert smdataparallel_profiling_config.is_enabled() == expected_enabled
    assert (
        profiler_config_parser.should_save_metrics(
            MetricsCategory.SMDATAPARALLEL_PROFILING, current_step, current_time=current_time
        )
        == expected_can_save
    )

    expected_start_step, expected_end_step = expected_values
    assert smdataparallel_profiling_config.start_step == expected_start_step
    assert smdataparallel_profiling_config.end_step == expected_end_step


def test_disabled_profiler(
    missing_config_profiler_config_parser,
    invalid_config_profiler_config_parser,
    user_disabled_profiler_config_parser,
):
    """
    This test is meant to test that a missing config JSON or invalid config JSON or the user setting `ProfilerEnabled`
    to `false` will disable the profiler.
    """
    assert not missing_config_profiler_config_parser.profiling_enabled
    assert not invalid_config_profiler_config_parser.profiling_enabled
    assert not user_disabled_profiler_config_parser.profiling_enabled


def test_default_trace_file_values(simple_profiler_config_parser):
    """
    This test is meant to test setting default trace file values when the config is present.
    """
    assert simple_profiler_config_parser.profiling_enabled

    trace_file_config = simple_profiler_config_parser.config.trace_file
    assert trace_file_config.file_open_fail_threshold == FILE_OPEN_FAIL_THRESHOLD_DEFAULT

    rotation_policy = trace_file_config.rotation_policy
    assert rotation_policy.file_max_size == MAX_FILE_SIZE_DEFAULT
    assert rotation_policy.file_close_interval == CLOSE_FILE_INTERVAL_DEFAULT


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


def test_update_step_profiler_config_parser(
    monkeypatch,
    step_profiler_config_parser_path,
    old_step_profiler_config_parser_path,
    new_step_profiler_config_parser_path,
):
    """
    This test is meant to test two behaviors when profiler config parser dynamically reloads a config JSON with step
    fields:
        - Calling load_config when the JSON hasn't changed will not reload the JSON into memory and its old step fields.
        (this is important when the JSON does not have specified step parameters, for example).
        - Calling load_config when the JSON has changed will reload the JSON into memory with the new step fields.
    """
    # sanity check that the parser first parses the range fields as is.
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", step_profiler_config_parser_path)
    shutil.copy(old_step_profiler_config_parser_path, step_profiler_config_parser_path)
    profiler_config_parser = ProfilerConfigParser()
    first_accessed_time = get_last_accessed_time(step_profiler_config_parser_path)
    assert profiler_config_parser.profiling_enabled
    assert profiler_config_parser.config.detailed_profiling_config.is_enabled()
    assert profiler_config_parser.config.detailed_profiling_config.start_step is None
    assert profiler_config_parser.config.detailed_profiling_config.num_steps == 2

    # sanity check that calling should save metrics will replace unspecified range fields, and leave the rest as is.
    profiler_config_parser.should_save_metrics(MetricsCategory.DETAILED_PROFILING, 5)
    assert profiler_config_parser.config.detailed_profiling_config.start_step == 5
    assert profiler_config_parser.config.detailed_profiling_config.num_steps == 2

    profiler_config_parser.load_config()

    # check that the config wasn't loaded into memory again.
    last_accessed_time = get_last_accessed_time(step_profiler_config_parser_path)
    assert first_accessed_time == last_accessed_time

    # check that reloading the config when it hasn't changed won't change the config fields.
    assert profiler_config_parser.config.detailed_profiling_config.start_step == 5
    assert profiler_config_parser.config.detailed_profiling_config.num_steps == 2

    from smdebug.profiler.utils import get_last_modified_time

    mtime = get_last_modified_time(step_profiler_config_parser_path)
    atime = get_last_accessed_time(step_profiler_config_parser_path)

    time.sleep(0.001)  # allow time to pass so new modified time will be different
    shutil.copy(new_step_profiler_config_parser_path, step_profiler_config_parser_path)
    profiler_config_parser.load_config()

    print(vars(profiler_config_parser.config.detailed_profiling_config))

    mtime2 = get_last_modified_time(step_profiler_config_parser_path)
    atime2 = get_last_accessed_time(step_profiler_config_parser_path)
    assert mtime != mtime2
    assert atime != atime2

    # verify that the config was loaded into memory again.
    last_accessed_time = get_last_accessed_time(step_profiler_config_parser_path)
    assert first_accessed_time != last_accessed_time

    # check that reloading the config when it has changed will update the config fields.
    assert profiler_config_parser.profiling_enabled
    assert profiler_config_parser.config.detailed_profiling_config.is_enabled()
    assert profiler_config_parser.config.detailed_profiling_config.start_step == 10
    assert profiler_config_parser.config.detailed_profiling_config.num_steps == 5


def test_update_time_profiler_config_parser(
    monkeypatch,
    time_profiler_config_parser_path,
    old_time_profiler_config_parser_path,
    new_time_profiler_config_parser_path,
):
    """
    This test is meant to test two behaviors when profiler config parser dynamically reloads a config JSON with time
    fields:
        - Calling load_config when the JSON hasn't changed will not reload the JSON into memory and its old time fields.
        (this is important when the JSON does not have specified time parameters, for example).
        - Calling load_config when the JSON has changed will reload the JSON into memory with the new time fields.
    """
    # sanity check that the parser first parses the range fields as is.
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", time_profiler_config_parser_path)
    shutil.copy(old_time_profiler_config_parser_path, time_profiler_config_parser_path)
    profiler_config_parser = ProfilerConfigParser()
    first_accessed_time = get_last_accessed_time(time_profiler_config_parser_path)
    assert profiler_config_parser.profiling_enabled
    assert profiler_config_parser.config.detailed_profiling_config.is_enabled()
    assert profiler_config_parser.config.detailed_profiling_config.start_time_in_sec is None
    assert profiler_config_parser.config.detailed_profiling_config.duration_in_sec == 0.1

    # sanity check that calling should save metrics will replace unspecified range fields, and leave the rest as is.
    timestamp1 = int(time.time())
    profiler_config_parser.should_save_metrics(
        MetricsCategory.DETAILED_PROFILING, 5, current_time=timestamp1
    )
    assert profiler_config_parser.config.detailed_profiling_config.start_time_in_sec == timestamp1
    assert profiler_config_parser.config.detailed_profiling_config.duration_in_sec == 0.1

    profiler_config_parser.load_config()

    # check that the config wasn't loaded into memory again.
    last_accessed_time = get_last_accessed_time(time_profiler_config_parser_path)
    assert first_accessed_time == last_accessed_time

    # check that reloading the config when it hasn't changed won't change the config fields.
    assert profiler_config_parser.config.detailed_profiling_config.start_time_in_sec == timestamp1
    assert profiler_config_parser.config.detailed_profiling_config.duration_in_sec == 0.1

    from smdebug.profiler.utils import get_last_modified_time

    mtime = get_last_modified_time(time_profiler_config_parser_path)
    atime = get_last_accessed_time(time_profiler_config_parser_path)

    time.sleep(0.001)  # allow time to pass so new modified time will be different
    shutil.copy(new_step_profiler_config_parser_path, step_profiler_config_parser_path)
    profiler_config_parser.load_config()

    print(vars(profiler_config_parser.config.detailed_profiling_config))

    mtime2 = get_last_modified_time(time_profiler_config_parser_path)
    atime2 = get_last_accessed_time(time_profiler_config_parser_path)
    assert mtime != mtime2
    assert atime != atime2

    # verify that the config was loaded into memory again.
    last_accessed_time = get_last_accessed_time(time_profiler_config_parser_path)
    assert first_accessed_time != last_accessed_time

    # check that reloading the config when it has changed will update the config fields.
    assert profiler_config_parser.profiling_enabled
    assert profiler_config_parser.config.detailed_profiling_config.is_enabled()
    assert profiler_config_parser.config.detailed_profiling_config.start_time_in_sec == 1700000000
    assert profiler_config_parser.config.detailed_profiling_config.duration_in_sec == 5


def test_update_disabled_profiler_config_parser(
    monkeypatch, user_disabled_profiler_config_parser, new_step_profiler_config_parser_path
):
    """
    This test is meant to test that reloading the config from a disabled config to an enabled config will actually
    shift the parser from having profiling disabled to profiling enabled.
    """
    # sanity check that the disabled parser has profiling enabled set to False.
    profiler_config_parser = user_disabled_profiler_config_parser
    assert not profiler_config_parser.profiling_enabled

    # check that reloading the config when it has changed will update the config fields.
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", new_step_profiler_config_parser_path)
    profiler_config_parser.load_config()
    assert profiler_config_parser.profiling_enabled
    assert profiler_config_parser.config.detailed_profiling_config.is_enabled()
    assert profiler_config_parser.config.detailed_profiling_config.start_step == 10
    assert profiler_config_parser.config.detailed_profiling_config.num_steps == 5
