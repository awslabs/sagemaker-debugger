# Standard Library
import json
import os
from collections import defaultdict
from enum import Enum

# First Party
from smdebug.core.access_layer.file import TSAccessFile
from smdebug.core.logger import get_logger
from smdebug.core.utils import get_node_id
from smdebug.profiler.profiler_config import ProfilerConfig
from smdebug.profiler.profiler_constants import (
    BASE_FOLDER_DEFAULT,
    CLOSE_FILE_INTERVAL_DEFAULT,
    CONFIG_PATH_DEFAULT,
    FILE_OPEN_FAIL_THRESHOLD_DEFAULT,
    MAX_FILE_SIZE_DEFAULT,
    TF_STEP_NUMBER_FILENAME,
)
from smdebug.profiler.utils import str2bool


class LastProfilingStatus(Enum):
    """Enum to track last profiling status so that we log for any changes
    """

    START = 1
    CONFIG_NOT_FOUND = 2
    INVALID_CONFIG = 3
    DEFAULT_ENABLED = 4
    PROFILER_DISABLED = 5
    PROFILER_ENABLED = 6
    DEFAULT_VALUES = 7
    DEFAULT_PYTHON_PROFILING = 8
    INVALID_GENERAL_METRICS_CONFIG = 9
    INVALID_DETAILED_PROFILING_CONFIG = 10
    INVALID_DATALOADER_METRICS_CONFIG = 11
    INVALID_PYTHON_PROFILING_CONFIG = 12
    INVALID_GENERAL_CONFIG_FIELDS = 13
    INVALID_DETAILED_CONFIG_FIELDS = 14
    INVALID_DATALOADER_CONFIG_FIELDS = 15
    INVALID_PYTHON_CONFIG_FIELDS = 16
    DETAILED_CONFIG_NOT_FOUND = 17
    INVALID_HERRING_PROFILING_CONFIG = 18


class MetricsCategory(Enum):
    """Enum to track each possible metrics that can be collected.
    """

    DETAILED_PROFILING = 1
    DATALOADER = 2
    PYTHON_PROFILING = 3
    HERRING_PROFILING = 4


class ProfilingParametersField(Enum):
    """Enum to track each field parsed from the profiling parameters.
    """

    PROFILING_PARAMETERS = "profilingparameters"
    PROFILER_ENABLED = "profilerenabled"
    LOCAL_PATH = "localpath"
    FILE_MAX_SIZE = "rotatemaxfilesizeinbytes"
    FILE_CLOSE_INTERVAL = "rotatefilecloseintervalinseconds"
    FILE_OPEN_FAIL_THRESHOLD = "fileopenfailthreshold"
    GENERAL_METRICS_CONFIG = "generalmetricsconfig"
    DETAILED_PROFILING_CONFIG = "detailedprofilingconfig"
    DATALOADER_METRICS_CONFIG = "dataloadermetricsconfig"
    PYTHON_PROFILING_CONFIG = "pythonprofilingconfig"
    HERRING_PROFILING_CONFIG = "smdataparallelprofilingconfig"


class ProfilerConfigParser:
    """Load the configuration file for the Profiler.
    Also set the provided values for the specified variables and default values for the rest.

    TODO: Poll for changes in the config by repeatedly calling `load_config`.
    """

    def __init__(self):
        """Initialize the parser to be disabled for profiling and detailed profiling.
        """
        self.config = None
        self.tf_step_number_writer = None
        self.profiling_enabled = False
        self.logger = get_logger()
        self.last_logging_statuses = defaultdict(lambda: False)
        self.current_logging_statuses = defaultdict(lambda: False)
        self.load_config()

    def _reset_statuses(self):
        """Set the last logging statuses to be the current logging statuses and reset the current logging statuses.
        """
        self.last_logging_statuses = self.current_logging_statuses
        self.current_logging_statuses = defaultdict(lambda: False)

    def _log_new_message(self, status, log_function, message):
        """Helper function to log the given message only if the given status was not True in the last call to
        load_config. In other words, only log this message if it wasn't already logged before. Use the provided log
        function to log the message

        Also mark this status as True so that we do not log the message in the next call to load_config if the status
        is once again True.
        """
        if not self.last_logging_statuses[status]:
            log_function(message)
        self.current_logging_statuses[status] = True

    def _parse_metrics_config(self, config, config_name, invalid_status):
        """Helper function to parse each metrics config from config given the ProfilingParametersField (config_name).
        If an error occurs in parsing, log the message with the provided LastProfilingStatus (invalid_status).
        """
        try:
            metrics_config = eval(config.get(config_name.value, "{}"))
            assert isinstance(metrics_config, dict)
            return metrics_config
        except (ValueError, AssertionError) as e:
            self._log_new_message(
                invalid_status,
                self.logger.error,
                f"{e} in {config_name.value}. Default metrics collection will be enabled.",
            )
            return {}

    def load_config(self):
        """Load the config file (if it exists) from $SMPROFILER_CONFIG_PATH.
        Set the provided values for the specified variables and default values for the rest.
        Validate the detailed profiling config (if it exists).
        """
        self.config = None
        self.tf_step_number_writer = None
        config_path = os.environ.get("SMPROFILER_CONFIG_PATH", CONFIG_PATH_DEFAULT)

        if os.path.isfile(config_path):
            with open(config_path) as json_data:
                try:
                    config = json.loads(json_data.read().lower()).get(
                        ProfilingParametersField.PROFILING_PARAMETERS.value
                    )
                except:
                    self._log_new_message(
                        LastProfilingStatus.INVALID_CONFIG,
                        self.logger.error,
                        f"Error parsing config at {config_path}.",
                    )
                    self._reset_statuses()
                    self.profiling_enabled = False
                    return
            try:
                profiler_enabled = str2bool(
                    config.get(ProfilingParametersField.PROFILER_ENABLED.value, True)
                )
            except ValueError as e:
                self._log_new_message(
                    LastProfilingStatus.DEFAULT_ENABLED,
                    self.logger.info,
                    f"{e} in {ProfilingParametersField.PROFILING_PARAMETERS}. Profiler is enabled.",
                )
                profiler_enabled = True
            if profiler_enabled is True:
                self._log_new_message(
                    LastProfilingStatus.PROFILER_ENABLED,
                    self.logger.info,
                    f"Using config at {config_path}.",
                )
                self.profiling_enabled = True
            else:
                self._log_new_message(
                    LastProfilingStatus.PROFILER_DISABLED,
                    self.logger.info,
                    f"User has disabled profiler.",
                )
                self._reset_statuses()
                self.profiling_enabled = False
                return
        else:
            self._log_new_message(
                LastProfilingStatus.CONFIG_NOT_FOUND,
                self.logger.info,
                f"Unable to find config at {config_path}. Profiler is disabled.",
            )
            self._reset_statuses()
            self.profiling_enabled = False
            return

        try:
            local_path = config.get(ProfilingParametersField.LOCAL_PATH.value, BASE_FOLDER_DEFAULT)
            file_max_size = int(
                float(
                    config.get(ProfilingParametersField.FILE_MAX_SIZE.value, MAX_FILE_SIZE_DEFAULT)
                )
            )
            file_close_interval = float(
                config.get(
                    ProfilingParametersField.FILE_CLOSE_INTERVAL.value, CLOSE_FILE_INTERVAL_DEFAULT
                )
            )
            file_open_fail_threshold = int(
                config.get(
                    ProfilingParametersField.FILE_OPEN_FAIL_THRESHOLD.value,
                    FILE_OPEN_FAIL_THRESHOLD_DEFAULT,
                )
            )
        except ValueError as e:
            self._log_new_message(
                LastProfilingStatus.DEFAULT_VALUES,
                self.logger.info,
                f"{e} in {ProfilingParametersField.PROFILING_PARAMETERS}. Enabling profiling with default "
                f"parameter values.",
            )
            local_path = BASE_FOLDER_DEFAULT
            file_max_size = MAX_FILE_SIZE_DEFAULT
            file_close_interval = CLOSE_FILE_INTERVAL_DEFAULT
            file_open_fail_threshold = FILE_OPEN_FAIL_THRESHOLD_DEFAULT

        general_metrics_config = self._parse_metrics_config(
            config,
            ProfilingParametersField.GENERAL_METRICS_CONFIG,
            LastProfilingStatus.INVALID_GENERAL_METRICS_CONFIG,
        )
        detailed_profiling_config = self._parse_metrics_config(
            config,
            ProfilingParametersField.DETAILED_PROFILING_CONFIG,
            LastProfilingStatus.INVALID_DETAILED_PROFILING_CONFIG,
        )
        dataloader_metrics_config = self._parse_metrics_config(
            config,
            ProfilingParametersField.DATALOADER_METRICS_CONFIG,
            LastProfilingStatus.INVALID_DATALOADER_METRICS_CONFIG,
        )
        python_profiling_config = self._parse_metrics_config(
            config,
            ProfilingParametersField.PYTHON_PROFILING_CONFIG,
            LastProfilingStatus.INVALID_PYTHON_PROFILING_CONFIG,
        )
        herring_profiling_config = self._parse_metrics_config(
            config,
            ProfilingParametersField.HERRING_PROFILING_CONFIG,
            LastProfilingStatus.INVALID_HERRING_PROFILING_CONFIG,
        )

        self.config = ProfilerConfig(
            local_path,
            file_max_size,
            file_close_interval,
            file_open_fail_threshold,
            general_metrics_config,
            detailed_profiling_config,
            dataloader_metrics_config,
            python_profiling_config,
            herring_profiling_config,
        )

        if self.config.general_metrics_config.error_message is not None:
            self._log_new_message(
                LastProfilingStatus.INVALID_GENERAL_CONFIG_FIELDS,
                self.logger.error,
                self.config.general_metrics_config.error_message,
            )

        if (
            self.config.detailed_profiling_config.error_message is not None
            and detailed_profiling_config != general_metrics_config
        ):
            self._log_new_message(
                LastProfilingStatus.INVALID_DETAILED_CONFIG_FIELDS,
                self.logger.error,
                self.config.detailed_profiling_config.error_message,
            )

        if (
            self.config.dataloader_metrics_config.error_message is not None
            and dataloader_metrics_config != general_metrics_config
        ):
            self._log_new_message(
                LastProfilingStatus.INVALID_DATALOADER_CONFIG_FIELDS,
                self.logger.error,
                self.config.dataloader_metrics_config.error_message,
            )

        if (
            self.config.python_profiling_config.error_message is not None
            and python_profiling_config != general_metrics_config
        ):
            self._log_new_message(
                LastProfilingStatus.INVALID_PYTHON_CONFIG_FIELDS,
                self.logger.error,
                self.config.python_profiling_config.error_message,
            )

        if (
            self.config.herring_profiling_config.error_message is not None
            and herring_profiling_config != general_metrics_config
        ):
            self._log_new_message(
                LastProfilingStatus.INVALID_HERRING_CONFIG_FIELDS,
                self.logger.error,
                self.config.herring_profiling_config.error_message,
            )

        self._reset_statuses()

    def should_save_metrics(
        self, metrics_category, current_step, metrics_name=None, current_time=None
    ):
        """Takes in a metrics category and current step and returns whether to collect metrics for that step. Metrics
        category must be one of the metrics specified in MetricNames. If metrics category is Dataloader, then metrics
        name is required and check if the metrics regex specified in the dataloader config matches this name.
        """
        if not self.profiling_enabled:
            return False

        if metrics_category == MetricsCategory.DETAILED_PROFILING:
            metric_config = self.config.detailed_profiling_config
        elif metrics_category == MetricsCategory.DATALOADER:
            metric_config = self.config.dataloader_metrics_config
            if not metric_config.valid_metrics_name(metrics_name):
                return False
        elif metrics_category == MetricsCategory.PYTHON_PROFILING:
            metric_config = self.config.python_profiling_config
        elif metrics_category == MetricsCategory.HERRING_PROFILING:
            metric_config = self.config.herring_profiling_config
        else:
            return False  # unrecognized metrics category

        # need to call can_start_profiling for both so that end step/time is updated if necessary
        can_profile_general = self.config.general_metrics_config.can_start_profiling(
            current_step, current_time
        )
        can_profile_metric = metric_config.can_start_profiling(current_step, current_time)
        return can_profile_general or can_profile_metric

    def write_tf_step_number(self, current_step):
        """If dataloader metrics collection is enabled, write the current step number to:
        <local_path>/<node_id>/tf_step_number. We simply update the file but never close the writer,
        since we don't want the file to be uploaded to s3.
        """
        if not self.profiling_enabled or not self.config.dataloader_metrics_config.is_enabled():
            return

        if self.tf_step_number_writer is None:
            self.tf_step_number_writer = TSAccessFile(
                os.path.join(self.config.local_path, get_node_id(), TF_STEP_NUMBER_FILENAME), "w"
            )

        self.tf_step_number_writer.write(str(current_step))
        self.tf_step_number_writer.flush()
