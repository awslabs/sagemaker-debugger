# Standard Library
import os

# First Party
from smdebug.core.logger import get_logger

logger = get_logger()


class ConfigValidator(object):
    def __init__(self, framework):
        self._create_hook = True
        self._summary = ""
        self._framework = framework

    def _validate_training_environment(self):
        logger.info("Validting the training environment")
        if self._framework == "pytorch":
            from smdebug.pytorch.utils import PT_VERSION, is_current_version_supported

            if is_current_version_supported() is False:
                logger.warning(f"The available {PT_VERSION} is not supported.")
                self._create_hook = False
            else:
                logger.info(f"The available {PT_VERSION} is supported.")
        if self._framework == "tensorflow":
            from smdebug.tensorflow.utils import TF_VERSION, is_current_version_supported

            if is_current_version_supported() is False:
                logger.warning(f"The available {TF_VERSION} is not supported.")
                self._create_hook = False
            else:
                logger.info(f"The available {TF_VERSION} is supported.")

    def _validate_profiler_config(self):
        logger.info("Validting the profiler configuration")

    def _validate_debugger_config(self):
        logger.info("Validting the debugger configuration")

    def validate_training_Job(self):
        self._validate_training_environment()
        self._validate_debugger_config()
        self._validate_profiler_config()
        if self._create_hook is False:
            logger.warning(f"Setting the USE_SMDEBUG flag to False")
            os.environ["USE_SMDEBUG"] = "False"
        return self._create_hook
