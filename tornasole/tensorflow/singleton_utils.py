"""
Easy-to-use methods for getting the singleton TornasoleHook.
This is abstracted into its own module to prevent circular import problems.

Sample usage (in AWS-TensorFlow repo):

import tornasole.tensorflow as ts
hook = ts.hook()
"""

import tornasole.core.singleton_utils as sutils
from tornasole.core.singleton_utils import set_hook, del_hook


def get_hook(json_config_path: str = None, hook_type: str = None) -> "TornasoleHook":
    """

    hook_type is one of ['session', 'estimator', 'keras', None].
    """
    from tornasole.tensorflow.hook import TornasoleHook

    # TODO(huilgolr): Choose tornasole_hook_type based on `hook_type`.
    return sutils.get_hook(json_config_path=json_config_path, tornasole_hook_class=TornasoleHook)
