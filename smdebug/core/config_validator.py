# Standard Library
import os

# First Party
import smdebug.core.utils
from smdebug.core.logger import get_logger
from smdebug.core.utils import is_framework_version_supported
from smdebug.profiler.profiler_config_parser import ProfilerConfigParser

logger = get_logger()

_config_validator = None


class ConfigValidator(object):
    def __init__(self, framework):
        self._create_hook = os.getenv("USE_SMDEBUG", "").upper() not in [
            "OFF",
            "0",
            "NO",
            "FALSE",
            "N",
        ]
        self._summary = ""
        self._framework = framework
        self.autograd_profiler_supported = True

    def _validate_training_environment(self):
        """
        The function checks whether the debugger functionality can be supported in the given training environment.
        Currently, if the training job is using a framework that is not supported by the debugger, we disable the
        creation of hook and disable the debugger functionality.
        :return:
        """
        if self._framework == "pytorch":
            if is_framework_version_supported(self._framework) is False:
                self._create_hook = False
                return
        if self._framework == "tensorflow":
            if is_framework_version_supported(self._framework) is False:
                self._create_hook = False

    def _validate_profiler_config(self):
        """
        The function parses the profiler configuration and ensures that hte configuration can be supported for the
        current training job.
        Currently, if the training job is using smmodel parallel we will disable the autograd profiler
        :return:
        """
        # Since we support the functionality to update profiler config during training, we would need to reload the
        # configuration and re-evaluate whether we can support autograd profiler.
        profiler_config = ProfilerConfigParser()
        if (
            profiler_config.profiling_enabled
            and profiler_config.config.detailed_profiling_config is not None
        ):
            if smdebug.core.utils.check_smmodelparallel_training():
                self.autograd_profiler_supported = False
                logger.warning("Detailed profiling for model parallel training job is disabled.")

    def _validate_debugger_config(self):
        """
        TODO: Analyze the debugger configuration and summarize the performance impact caused by parameters set in
        collection configuration. A saveall collection or saving tensors for every step wil have a significant
        performance impact. We can let the user know about this impact through this function.
        :return:
        """

    def validate_training_Job(self):
        # If Hook is disabled we need to validate the training job again.
        if self._create_hook is False:
            return
        self._validate_training_environment()
        self._validate_debugger_config()
        self._validate_profiler_config()
        if self._create_hook is False:
            logger.warning(f"Setting the USE_SMDEBUG flag to False")
            os.environ["USE_SMDEBUG"] = "False"
        return self._create_hook


def get_config_validator(framework):
    global _config_validator
    if _config_validator is None:
        from smdebug.core.config_validator import ConfigValidator

        _config_validator = ConfigValidator(framework)
    return _config_validator


def reset_config_validator():
    global _config_validator
    _config_validator = None
