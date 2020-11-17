# Standard Library
import re
import time
from enum import Enum

# First Party
from smdebug.profiler.profiler_constants import (
    CPROFILE_NAME,
    PROFILING_NUM_STEPS_DEFAULT,
    PYINSTRUMENT_NAME,
)
from smdebug.profiler.python_profiler import cProfileTimer


class MetricsConfigsField(Enum):
    """Enum to track each field parsed from any particular metrics config.
    """

    START_STEP = "startstep"
    NUM_STEPS = "numsteps"
    START_TIME = "starttimeinsecsinceepoch"
    DURATION = "durationinseconds"
    DEFAULT_PROFILING = "defaultprofiling"
    METRICS_REGEX = "metricsregex"
    PROFILER_NAME = "profilername"
    CPROFILE_TIMER = "cprofiletimer"


class RotationPolicy:
    """Configuration corresponding to rotation policy of trace event files.
    """

    def __init__(self, file_max_size, file_close_interval):
        self.file_max_size = file_max_size
        self.file_close_interval = file_close_interval


class TraceFile:
    """Configuration corresponding to trace event files.
    """

    def __init__(self, file_max_size, file_close_interval, file_open_fail_threshold):
        self.rotation_policy = RotationPolicy(file_max_size, file_close_interval)
        self.file_open_fail_threshold = file_open_fail_threshold


class ProfileRange:
    """Configuration common to all of the configs: the start of profiling, how long profiling should take and whether
    an error occurred in parsing the config. Assigns each field from the dictionary, (or `None` if it doesn't exist)
    """

    def __init__(self, name, profile_range):
        self.name = name
        self.error_message = None
        self.disabled = False

        # user did not specify this config, so it is disabled.
        if profile_range == {}:
            self.disabled = True
            return

        # step range
        self.start_step = profile_range.get(MetricsConfigsField.START_STEP.value)
        self.num_steps = profile_range.get(MetricsConfigsField.NUM_STEPS.value)
        self.end_step = None

        # time range
        self.start_time_in_sec = profile_range.get(MetricsConfigsField.START_TIME.value)
        self.duration_in_sec = profile_range.get(MetricsConfigsField.DURATION.value)
        self.end_time = None

        # convert to the correct type
        try:
            if self.start_step is not None:
                self.start_step = int(self.start_step)
            if self.num_steps is not None:
                self.num_steps = int(self.num_steps)
            if self.start_time_in_sec is not None:
                self.start_time_in_sec = float(self.start_time_in_sec)
            if self.duration_in_sec is not None:
                self.duration_in_sec = float(self.duration_in_sec)
        except ValueError as e:
            self.error_message = f"{e} encountered in {self.name} config. Disabling {self.name}."
            self.reset_profile_range()
            return

        if self.has_step_range() and self.has_time_range():
            self.error_message = (
                f"Found step and time fields in {self.name} config! Disabling {self.name }."
            )

    def has_step_range(self):
        """Return whether one of the step fields (start step or num steps) has been specified in the config.
        """
        return self.start_step or self.num_steps

    def has_time_range(self):
        """Return whether one of the time fields (start time or duration) has been specified in the config.
        """
        return self.start_time_in_sec or self.duration_in_sec

    def reset_profile_range(self):
        """Helper function to reset fields in profile range. Used primarily when an error parsing config occurs.
        """
        self.start_step = self.num_steps = self.start_time_in_sec = self.duration_in_sec = None

    def is_enabled(self):
        """Return whether this config is enabled (specified by the user and no errors during parsing"""
        return not self.disabled and self.error_message is None

    def can_start_profiling(self, current_step, current_time):
        """Determine whether the values from the config are valid for profiling."""
        if not self.is_enabled():
            return False

        if current_time is None:
            current_time = time.time()
        if self.has_step_range():
            if self.start_step is None:
                self.start_step = current_step
            if self.num_steps is None:
                self.num_steps = PROFILING_NUM_STEPS_DEFAULT
            if not self.end_step:
                self.end_step = self.start_step + self.num_steps
            return self.start_step <= current_step < self.end_step
        elif self.has_time_range():
            if self.start_time_in_sec is None:
                self.start_time_in_sec = current_time
            if self.duration_in_sec:
                if self.end_time is None:
                    self.end_time = self.start_time_in_sec + self.duration_in_sec
                return self.start_time_in_sec <= current_time < self.end_time
            else:
                if self.start_time_in_sec <= current_time:
                    if self.end_step is None:
                        self.end_step = current_step + 1
                    return current_step < self.end_step
        return False


class DetailedProfilingConfig(ProfileRange):
    """Configuration corresponding to the detailed profiling config."""

    def __init__(self, detailed_profiling_config):
        super().__init__("detailed profiling", detailed_profiling_config)


class DataloaderProfilingConfig(ProfileRange):
    """Configuration corresponding to the dataloader profiling config.

    TODO: Use this config to collect dataloader metrics only for the specified steps.
    """

    def __init__(self, dataloader_config):
        super().__init__("dataloader profiling", dataloader_config)

        if self.error_message:
            return

        try:
            self.metrics_regex = re.compile(
                dataloader_config.get(MetricsConfigsField.METRICS_REGEX.value, ".*")
            )
        except re.error as e:
            self.metrics_regex = None
            self.error_message = f"{e} encountered in {self.name} config. Disabling {self.name}."
            self.reset_profile_range()

    def valid_metrics_name(self, metrics_name):
        """Check if the metrics regex matches the provided metrics name. Note: this is case insensitive.
        """
        if self.metrics_regex is None:
            return False
        return self.metrics_regex.match(metrics_name.lower()) is not None


class PythonProfilingConfig(ProfileRange):
    """Configuration corresponding to the python profiling config. """

    def __init__(self, python_profiling_config):
        super().__init__("python profiling", python_profiling_config)

        if self.error_message:
            return

        self.profiler_name = python_profiling_config.get(
            MetricsConfigsField.PROFILER_NAME.value, CPROFILE_NAME
        )
        if self.profiler_name not in (CPROFILE_NAME, PYINSTRUMENT_NAME):
            self.error_message = f"Profiler name must be {CPROFILE_NAME} or {PYINSTRUMENT_NAME}!"
            self.profiler_name = self.cprofile_timer = None
            self.reset_profile_range()
            return

        try:
            self.cprofile_timer = None
            if self.profiler_name == CPROFILE_NAME:
                # only parse this field if pyinstrument is not specified and we are not doing the default three steps
                # of profiling.
                self.cprofile_timer = cProfileTimer(
                    python_profiling_config.get(
                        MetricsConfigsField.CPROFILE_TIMER.value, "total_time"
                    )
                )
        except ValueError as e:
            self.error_message = f"{e} encountered in {self.name} config. Disabling {self.name}."
            self.profiler_name = self.cprofile_timer = None
            self.reset_profile_range()


class SMDataParallelProfilingConfig(ProfileRange):
    """Configuration corresponding to the smdataparallel profiling config."""

    def __init__(self, smdataparallel_profiling_config):
        super().__init__("smdataparallel profiling", smdataparallel_profiling_config)


class ProfilerConfig:
    """Overall profiler configuration
    """

    def __init__(
        self,
        local_path,
        file_max_size,
        file_close_interval,
        file_open_fail_threshold,
        detailed_profiling_config,
        dataloader_profiling_config,
        python_profiling_config,
        smdataparallel_profiling_config,
    ):
        """
        :param local_path: path where profiler events have to be saved.
        :param file_max_size: Max size a trace file can be, before being rotated.
        :param file_close_interval: Interval in seconds from the last close, before being rotated.
        :param file_open_fail_threshold: Number of times to attempt to open a trace fail before marking the writer as unhealthy.
        :param detailed_profiling_config Dictionary holding the detailed profiling config.
        :param dataloader_profiling_config Dictionary holding the dataloader profiling config.
        :param python_profiling_config Dictionary holding the python profiling config.
        :param smdataparallel_profiling_config Dictionary holding the SMDataParallel profiling config.
        """
        self.local_path = local_path
        self.trace_file = TraceFile(file_max_size, file_close_interval, file_open_fail_threshold)
        self.detailed_profiling_config = DetailedProfilingConfig(detailed_profiling_config)
        self.dataloader_profiling_config = DataloaderProfilingConfig(dataloader_profiling_config)
        self.python_profiling_config = PythonProfilingConfig(python_profiling_config)
        self.smdataparallel_profiling_config = SMDataParallelProfilingConfig(
            smdataparallel_profiling_config
        )
