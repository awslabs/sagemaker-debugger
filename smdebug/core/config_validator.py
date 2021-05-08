# Standard Library
import os

# First Party
from smdebug.core.logger import get_logger
from smdebug.core.utils import is_framework_version_supported

logger = get_logger()


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

    def _validate_training_environment(self):
        logger.info("Validting the training environment")
        if self._framework == "pytorch":
            if is_framework_version_supported() is False:
                self._create_hook = False
                return
        if self._framework == "tensorflow":
            if is_framework_version_supported() is False:
                self._create_hook = False

    def _validate_profiler_config(self):
        logger.info("Validting the profiler configuration")

    def _validate_debugger_config(self):
        logger.info("Validting the debugger configuration")

    def validate_training_Job(self):
        if self._create_hook is False:
            return
        self._validate_training_environment()
        self._validate_debugger_config()
        self._validate_profiler_config()
        if self._create_hook is False:
            logger.warning(f"Setting the USE_SMDEBUG flag to False")
            os.environ["USE_SMDEBUG"] = "False"
        return self._create_hook
