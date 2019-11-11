"""
Easy-to-use methods for getting the singleton TornasoleHook.

Sample usage:

import tornasole.(pytorch | tensorflow | mxnet) as ts
hook = ts.hook()
"""

from tornasole.core.logger import get_logger

logger = get_logger()
_ts_hook = None


def _create_hook(json_config_path, tornasole_hook_class):
    from tornasole.core.hook import BaseHook  # prevent circular imports

    if tornasole_hook_class is None:
        raise ValueError("tornasole_hook_class can not be None.", tornasole_hook_class)
    if not issubclass(tornasole_hook_class, BaseHook):
        raise TypeError(
            "tornasole_hook_class needs to be a subclass of BaseHook", tornasole_hook_class
        )

    # Either returns a hook or None
    try:
        hook = tornasole_hook_class.hook_from_config(json_config_path=json_config_path)
        set_hook(custom_hook=hook)
    except FileNotFoundError:
        logger.info(
            f"Tornasole is disabled as hook was not created in code "
            f"as well as json config file to create hook from was not found."
        )


def get_hook(json_config_path, tornasole_hook_class, create_if_not_exists) -> "TornasoleHook":
    """Return a singleton TornasoleHook or None.

    If the singleton hook exists, we return it. No questions asked, `json_config_path` is a no-op.
    Otherwise return hook_from_config().
    """
    global _ts_hook

    if create_if_not_exists:
        # If global hook exists, return it
        if _ts_hook:
            logger.info(f"Tornasole will use the existing TornasoleHook.")
        else:
            _create_hook(json_config_path, tornasole_hook_class)
    return _ts_hook


def set_hook(custom_hook: "TornasoleHook") -> None:
    """Overwrite the current hook with the passed hook."""
    from tornasole.core.hook import BaseHook  # prevent circular imports

    if not isinstance(custom_hook, BaseHook):
        raise TypeError(f"custom_hook={custom_hook} must be type TornasoleHook")

    global _ts_hook
    _ts_hook = custom_hook


def del_hook() -> None:
    """ Set the hook singleton to None. """
    global _ts_hook
    _ts_hook = None
