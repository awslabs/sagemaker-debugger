"""
Easy-to-use methods for getting the singleton TornasoleHook.

Sample usage:

import tornasole.(pytorch | tensorflow | mxnet) as ts
hook = ts.hook()
"""

import logging
import os

_ts_hook = None


def get_hook(json_config_path, tornasole_hook_class) -> "TornasoleHook":
    """Return a singleton TornasoleHook or None.

    If the singleton hook exists, we return it. No questions asked, `json_config_path` is a no-op.
    Otherwise return hook_from_config().
    """
    global _ts_hook

    # If global hook exists, return it
    if _ts_hook:
        if json_config_path is not None:
            logging.error(
                f"`json_config_path` was passed, but TornasoleHook already exists. "
                f"Using the existing hook."
            )
        return _ts_hook
    # Otherwise return hook_from_config
    else:
        # Either returns a hook or None
        try:
            set_hook(
                custom_hook=tornasole_hook_class.hook_from_config(json_config_path=json_config_path)
            )
        except FileNotFoundError:
            pass

        return _ts_hook


def set_hook(custom_hook: "TornasoleHook") -> None:
    """Overwrite the current hook with the passed hook."""
    from tornasole.core.hook import BaseHook  # prevent circular imports

    if not isinstance(custom_hook, BaseHook):
        raise TypeError(f"custom_hook={custom_hook} must be type TornasoleHook")

    global _ts_hook
    _ts_hook = custom_hook
