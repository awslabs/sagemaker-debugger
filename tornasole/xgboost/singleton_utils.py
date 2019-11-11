"""
Easy-to-use methods for getting the singleton TornasoleHook.
This is abstracted into its own module to prevent circular import problems.

Sample usage:

import tornasole.xgboost as ts
hook = ts.hook()
"""

import tornasole.core.singleton_utils as sutils
from tornasole.core.singleton_utils import set_hook, del_hook


def get_hook(json_config_path=None, create_if_not_exists: bool = False) -> "TornasoleHook":
    from tornasole.xgboost.hook import TornasoleHook

    return sutils.get_hook(
        json_config_path=json_config_path,
        tornasole_hook_class=TornasoleHook,
        create_if_not_exists=create_if_not_exists,
    )
