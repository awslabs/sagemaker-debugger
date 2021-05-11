"""
Easy-to-use methods for getting the singleton SessionHook.
This is abstracted into its own module to prevent circular import problems.

Sample usage (in AWS-PyTorch repo):

import smdebug.pytorch as smd
hook = smd.hook()
"""

# First Party
import smdebug.core.singleton_utils as sutils
from smdebug.core.singleton_utils import del_hook, set_hook  # noqa
from smdebug.core.utils import error_handling_agent

_config_validator_for_pytorch = None


def get_pytorch_config_validator():
    global _config_validator_for_pytorch
    if _config_validator_for_pytorch is None:
        from smdebug.core.config_validator import ConfigValidator

        _config_validator_for_pytorch = ConfigValidator(framework="pytorch")
    return _config_validator_for_pytorch


@error_handling_agent.catch_smdebug_errors()
def get_hook(json_config_path=None, create_if_not_exists: bool = False) -> "Hook":
    if get_pytorch_config_validator().validate_training_Job():
        from smdebug.pytorch.hook import Hook

        return sutils.get_hook(
            json_config_path=json_config_path,
            hook_class=Hook,
            create_if_not_exists=create_if_not_exists,
        )
    else:
        return None
