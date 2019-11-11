"""
Easy-to-use methods for getting the singleton TornasoleHook.
This is abstracted into its own module to prevent circular import problems.

Sample usage (in AWS-TensorFlow repo):

import tornasole.tensorflow as ts
hook = ts.get_hook()
"""

import tornasole.core.singleton_utils as sutils
from tornasole.core.singleton_utils import set_hook, del_hook


def get_hook(
    hook_type: str = None, json_config_path: str = None, create_if_not_exists: bool = False
) -> "TornasoleHook":
    """
    hook_type can be one of ['session', 'estimator', 'keras', None].

    If create_if_not_exists is False, it only returns any hook which was already created or None.

    If create_if_not_exists is True, it looks for any hook which was already created.
    If none exists, it tries to create a hook. In this case hook_type needs to be specified.
    """
    from tornasole.tensorflow import session, keras

    if hook_type == "session":
        tornasole_hook_class = session.TornasoleSessionHook
    elif hook_type == "estimator":
        tornasole_hook_class = session.TornasoleEstimatorHook
    elif hook_type == "keras":
        tornasole_hook_class = keras.TornasoleKerasHook
    else:
        tornasole_hook_class = None

    return sutils.get_hook(
        json_config_path=json_config_path,
        tornasole_hook_class=tornasole_hook_class,
        create_if_not_exists=create_if_not_exists,
    )
