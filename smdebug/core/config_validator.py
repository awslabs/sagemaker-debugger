# Standard Library
import os
from distutils.util import strtobool

# First Party
import smdebug.core.utils
from smdebug.core.logger import get_logger
from smdebug.core.utils import FRAMEWORK, is_framework_version_supported
from smdebug.profiler.profiler_config_parser import ProfilerConfigParser

logger = get_logger()

_config_validator = None


SupportedFrameworks = [FRAMEWORK.PYTORCH, FRAMEWORK.TENSORFLOW, FRAMEWORK.MXNET, FRAMEWORK.XGBOOST]


class ConfigValidator(object):
    def __init__(self, framework_type: FRAMEWORK):
        self._create_hook = strtobool(os.getenv("USE_SMDEBUG", "true").lower())
        self._summary = ""
        self._framework_type = framework_type

    def _validate_training_environment(self):
        """
        The function checks whether the debugger functionality can be supported in the given training environment.
        Currently, if the training job is using a framework that is not supported by the debugger, we disable the
        creation of hook and disable the debugger functionality.
        :return:
        """
        if (
            self._framework_type in SupportedFrameworks
            and is_framework_version_supported(self._framework_type) is False
        ):
            self._create_hook = False

    @staticmethod
    def validate_profiler_config(profiler_config_parser: ProfilerConfigParser):
        """
        The function parses the profiler configuration and ensures that the configuration can be supported for the
        current training job.
        Currently, if the training job is using smmodel parallel we will disable the autograd profiler
        The function is called whenever the ProfilerConfigParser loads the config.
        :return:
        """
        # Since we support the functionality to update profiler config during training, we would need to reload the
        # configuration and re-evaluate whether we can support autograd profiler.
        if (
            profiler_config_parser.profiling_enabled
            and profiler_config_parser.config.detailed_profiling_config is not None
        ):
            if smdebug.core.utils.check_smmodelparallel_training():
                profiler_config_parser.config.detailed_profiling_config.disabled = True
                logger.warning("Detailed profiling for model parallel training job is disabled.")

    def _validate_debugger_config(self):
        """
        TODO: Analyze the debugger configuration and summarize the performance impact caused by parameters set in
        collection configuration. A saveall collection or saving tensors for every step wil have a significant
        performance impact. We can let the user know about this impact through this function.
        :return:
        """

    def validate_training_job(self):
        # If Hook is disabled we need to validate the training job again.
        if self._create_hook is False:
            return
        self._validate_training_environment()
        self._validate_debugger_config()
        if self._create_hook is False:
            logger.warning(f"Setting the USE_SMDEBUG flag to False")
            os.environ["USE_SMDEBUG"] = "False"
        return self._create_hook


def get_config_validator(framework):
    """
    Returns the ConfigValidator object for the given framework.
    We will create this object only once.
    :param framework:
    :return: ConfigValidator
    """
    global _config_validator
    if _config_validator is None:
        from smdebug.core.config_validator import ConfigValidator

        _config_validator = ConfigValidator(framework)
    return _config_validator


def reset_config_validator():
    """
    Reset the ConfigValidator object.
    :return:
    """
    global _config_validator
    _config_validator = None
