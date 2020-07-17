# Standard Library
import json
import os
from enum import Enum

# First Party
from smdebug.core.logger import get_logger
from smdebug.profiler.profiler_config import ProfilerConfig
from smdebug.profiler.profiler_constants import (
    BASE_FOLDER_DEFAULT,
    CLOSE_FILE_INTERVAL_DEFAULT,
    CONFIG_PATH_DEFAULT,
    FILE_OPEN_FAIL_THRESHOLD_DEFAULT,
    MAX_FILE_SIZE_DEFAULT,
)
from smdebug.profiler.utils import str2bool


class LastProfilingStatus(Enum):
    """Enum to track last profiling status so that we log for any changes
    """

    START = "START"
    CONFIG_NOT_FOUND = "CONFIG_NOT_FOUND"
    INVALID_CONFIG = "INVALID_CONFIG"
    PROFILER_DISABLED = "PROFILER_DISABLED"
    PROFILER_ENABLED = "PROFILER_ENABLED"
    INVALID_DETAILED_CONFIG = "INVALID_DETAILED_CONFIG"
    DETAILED_CONFIG_NOT_FOUND = "DETAILED_CONFIG_NOT_FOUND"


class ProfilerConfigParser:
    """Load the configuration file for the Profiler.
    Also set the provided values for the specified variables and default values for the rest.

    TODO: Poll for changes in the config by repeatedly calling `load_config`.
    """

    def __init__(self):
        """Initialize the parser to be disabled for profiling and detailed profiling.
        """
        self.config = None
        self.profiling_enabled = False
        self.detailed_profiling_enabled = False
        self.last_status = LastProfilingStatus.START
        self.load_config()

    def load_config(self):
        """Load the config file (if it exists) from $SMPROFILER_CONFIG_PATH.
        Set the provided values for the specified variables and default values for the rest.
        Validate the detailed profiling config (if it exists).
        """
        config_path = os.environ.get("SMPROFILER_CONFIG_PATH", CONFIG_PATH_DEFAULT)

        if os.path.isfile(config_path):
            with open(config_path) as json_data:
                try:
                    config = json.load(json_data).get("ProfilingParameters")
                except:
                    if self.last_status != LastProfilingStatus.INVALID_CONFIG:
                        get_logger("smdebug-profiler").error(
                            f"Error parsing config at {config_path}."
                        )
                        self.last_status = LastProfilingStatus.INVALID_CONFIG
                    self.profiling_enabled = False
                    return
            try:
                profiler_enabled = str2bool(config.get("ProfilerEnabled", True))
            except ValueError as e:
                get_logger("smdebug-profiler").info(
                    f"{e} in ProfilingParameters. Enabling profiling with default "
                    f"parameter values."
                )
                profiler_enabled = True
            if profiler_enabled is True:
                if self.last_status != LastProfilingStatus.PROFILER_ENABLED:
                    get_logger("smdebug-profiler").info(f"Using config at {config_path}.")
                    self.last_status = LastProfilingStatus.PROFILER_ENABLED
                self.profiling_enabled = True
            else:
                if self.last_status != LastProfilingStatus.PROFILER_DISABLED:
                    get_logger("smdebug-profiler").info(f"User has disabled profiler.")
                    self.last_status = LastProfilingStatus.PROFILER_DISABLED
                self.profiling_enabled = False
                return
        else:
            if self.last_status != LastProfilingStatus.CONFIG_NOT_FOUND:
                get_logger("smdebug-profiler").info(
                    f"Unable to find config at {config_path}. Profiler is disabled."
                )
                self.last_status = LastProfilingStatus.CONFIG_NOT_FOUND
            self.profiling_enabled = False
            return

        try:
            local_path = config.get("LocalPath", BASE_FOLDER_DEFAULT)
            file_max_size = int(config.get("RotateMaxFileSizeInBytes", MAX_FILE_SIZE_DEFAULT))
            file_close_interval = float(
                config.get("RotateFileCloseIntervalInSeconds", CLOSE_FILE_INTERVAL_DEFAULT)
            )
            file_open_fail_threshold = int(
                config.get("FileOpenFailThreshold", FILE_OPEN_FAIL_THRESHOLD_DEFAULT)
            )
            use_pyinstrument = str2bool(config.get("UsePyInstrument", False))
        except ValueError as e:
            get_logger("smdebug-profiler").info(
                f"{e} in ProfilingParameters. Enabling profiling with default " f"parameter values."
            )
            local_path = BASE_FOLDER_DEFAULT
            file_max_size = MAX_FILE_SIZE_DEFAULT
            file_close_interval = CLOSE_FILE_INTERVAL_DEFAULT
            file_open_fail_threshold = FILE_OPEN_FAIL_THRESHOLD_DEFAULT
            use_pyinstrument = False

        profile_range = config.get("DetailedProfilingConfig", {})

        self.config = ProfilerConfig(
            local_path,
            file_max_size,
            file_close_interval,
            file_open_fail_threshold,
            use_pyinstrument,
            profile_range,
        )

        if (
            self.config.profile_range.has_step_range()
            and self.config.profile_range.has_time_range()
        ):
            if self.last_status != LastProfilingStatus.INVALID_DETAILED_CONFIG:
                get_logger("smdebug-profiler").error(
                    "User must not specify both step and time fields for profile range! No sync metrics will be logged."
                )
                self.last_status = LastProfilingStatus.INVALID_DETAILED_CONFIG
            self.detailed_profiling_enabled = False
            return

        elif (
            not self.config.profile_range.has_step_range()
            and not self.config.profile_range.has_time_range()
        ):
            if self.last_status != LastProfilingStatus.DETAILED_CONFIG_NOT_FOUND:
                get_logger("smdebug-profiler").debug(
                    "No detailed profiler config provided! No sync metrics will be logged."
                )
                self.last_status = LastProfilingStatus.DETAILED_CONFIG_NOT_FOUND
            self.detailed_profiling_enabled = False
            return

        self.detailed_profiling_enabled = True

    def can_start_detailed_profiling(self, current_step):
        """Higher level check to make sure that profiler is enabled AND that detailed profiling is enabled
        AND the config values are valid for detailed profiling.
        """
        return (
            self.profiling_enabled
            and self.detailed_profiling_enabled
            and self.config.profile_range.can_start_detailed_profiling(current_step)
        )
