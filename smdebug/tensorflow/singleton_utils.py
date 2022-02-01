"""
Easy-to-use methods for getting the singleton SessionHook.
This is abstracted into its own module to prevent circular import problems.

Sample usage (in AWS-TensorFlow repo):

import smdebug.tensorflow as smd
hook = smd.get_hook()
"""

# First Party
import smdebug.core.singleton_utils as sutils
from smdebug.core.singleton_utils import del_hook, set_hook  # noqa
from smdebug.core.utils import error_handling_agent


@error_handling_agent.catch_smdebug_errors()
def get_hook(
    hook_type: str = None, json_config_path: str = None, create_if_not_exists: bool = False
) -> "SessionHook":
    """
    hook_type can be one of ['session', 'estimator', 'keras', None].

    If create_if_not_exists is False, it only returns any hook which was already created or None.

    If create_if_not_exists is True, it looks for any hook which was already created.
    If none exists, it tries to create a hook. In this case hook_type needs to be specified.
    """
    from smdebug.tensorflow import session, keras

    if hook_type == "session":
        hook_class = session.SessionHook
    elif hook_type == "estimator":
        hook_class = session.EstimatorHook
    elif hook_type == "keras":
        hook_class = keras.KerasHook
    else:
        hook_class = None

    from smdebug.core.config_validator import get_config_validator, FRAMEWORK

    if get_config_validator(FRAMEWORK.TENSORFLOW).validate_training_job():
        return sutils.get_hook(
            json_config_path=json_config_path,
            hook_class=hook_class,
            create_if_not_exists=create_if_not_exists,
        )
    else:
        # we choose not to raise an exception, because sagemaker debugger is running
        # on ALL training jobs by default, and we don't know when `get_hook()` is being called
        # where it might fail to retrieve the hook
        return None
