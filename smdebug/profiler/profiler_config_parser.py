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
    PROFILER_DURATION_DEFAULT,
    PROFILER_NUM_STEPS_DEFAULT,
)


class LastStatus(Enum):
    """Enum to track last status so that we log for any changes
    """

    START = "START"
    CONFIG_NOT_FOUND = "CONFIG_NOT_FOUND"
    INVALID_CONFIG = "INVALID_CONFIG"
    PROFILER_DISABLED = "PROFILER_DISABLED"
    PROFILER_ENABLED = "PROFILER_ENABLED"


class ProfilerConfigParser:
    """Load the configuration file for the Profiler.
    Also set the provided values for the specified variables and default values for the rest.

    TODO: Poll for changes in the config by repeatedly calling `load_config`.
    """

    def __init__(self, current_step):
        """Set up the logger and load the config."""
        self.logger = get_logger("smdebug-profiler")
        self.enabled = False
        self.last_status = LastStatus.START
        self.load_config(current_step)

    def load_config(self, current_step):
        """Load the config file (if it exists) from $PROFILER_CONFIG_PATH.
        Set the provided values for the specified variables and default values for the rest.
        General profiler variables are set as environment variables, while TF profiler specfic
            variables are retrieved in a separate function.
        """
        config_path = os.environ.get("SMPROFILER_CONFIG_PATH", CONFIG_PATH_DEFAULT)

        if os.path.isfile(config_path):
            with open(config_path) as json_data:
                try:
                    config = json.load(json_data).get("ProfilingParameters")
                except:
                    if self.last_status != LastStatus.INVALID_CONFIG:
                        self.logger.error(f"Error parsing config at {config_path}.")
                        self.last_status = LastStatus.INVALID_CONFIG
                    return
            if config.get("ProfilerEnabled", True):
                if self.last_status != LastStatus.PROFILER_ENABLED:
                    self.logger.info(f"Using config at {config_path}.")
                    self.last_status = LastStatus.PROFILER_ENABLED
                self.enabled = True
            else:
                if self.last_status != LastStatus.PROFILER_DISABLED:
                    self.logger.info(f"User has disabled profiler.")
                    self.last_status = LastStatus.PROFILER_DISABLED
                return
        else:
            if self.last_status != LastStatus.CONFIG_NOT_FOUND:
                self.logger.info(f"Unable to find config at {config_path}. Profiler is disabled.")
                self.last_status = LastStatus.CONFIG_NOT_FOUND
            return

        local_path = config.get("LocalPath", BASE_FOLDER_DEFAULT)
        file_max_size = config.get("RotateMaxFileSizeInBytes", MAX_FILE_SIZE_DEFAULT)
        file_close_interval = config.get(
            "RotateFileCloseIntervalInSeconds", CLOSE_FILE_INTERVAL_DEFAULT
        )
        file_open_fail_threshold = config.get(
            "FileOpenFailThreshold", FILE_OPEN_FAIL_THRESHOLD_DEFAULT
        )
        profile_range = config.get("DetailedProfilingConfig", {})

        has_step_var = "StartStep" in profile_range or "NumSteps" in profile_range
        has_time_var = "StartTime" in profile_range or "Duration" in profile_range

        profile_type, profiler_start, profile_length, profiler_end = None, None, None, None

        if has_step_var and has_time_var:
            self.logger.error(
                "User must not specify both step and time fields for profile range! No profiling will occur."
            )
        elif has_step_var:
            profile_type = "steps"
            profiler_start = profile_range.get("StartStep", current_step)
            profile_length = profile_range.get("NumSteps", PROFILER_NUM_STEPS_DEFAULT)
            profiler_end = profiler_start + profile_length
        elif has_time_var:
            profile_type = "time"
            profiler_start = profile_range.get("StartTime", None)  # profile immediately by default
            profile_length = profile_range.get(
                "Duration", PROFILER_DURATION_DEFAULT
            )  # profile until next step by default
            if profiler_start:
                profiler_end = profiler_start + profile_length
        else:
            self.logger.error("No detailed profiler config provided! No profiling will occur.")

        self.config = ProfilerConfig(
            local_path,
            file_max_size,
            file_close_interval,
            file_open_fail_threshold,
            profile_type,
            profiler_start,
            profile_length,
            profiler_end,
        )
