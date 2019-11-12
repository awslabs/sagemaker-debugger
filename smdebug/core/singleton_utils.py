"""
Easy-to-use methods for getting the singleton SessionHook.

Sample usage:

import smdebug.(pytorch | tensorflow | mxnet) as smd
hook = smd.hook()
"""

# First Party
from smdebug.core.logger import get_logger

logger = get_logger()
_ts_hook = None


def _create_hook(json_config_path, hook_class):
    from smdebug.core.hook import BaseHook  # prevent circular imports

    if hook_class is None:
        raise ValueError("hook_class can not be None.", hook_class)
    if not issubclass(hook_class, BaseHook):
        raise TypeError("hook_class needs to be a subclass of BaseHook", hook_class)

    # Either returns a hook or None
    try:
        hook = hook_class.hook_from_config(json_config_path=json_config_path)
        set_hook(custom_hook=hook)
    except FileNotFoundError:
        logger.info(f"smdebug is disabled, since hook not created in code and no json config file.")


def get_hook(json_config_path, hook_class, create_if_not_exists) -> "SessionHook":
    """Return a singleton SessionHook or None.

    If the singleton hook exists, we return it. No questions asked, `json_config_path` is a no-op.
    Otherwise return hook_from_config().
    """
    global _ts_hook

    if create_if_not_exists:
        # If global hook exists, return it
        if _ts_hook:
            logger.info(f"Tornasole will use the existing SessionHook.")
        else:
            _create_hook(json_config_path, hook_class)
    return _ts_hook


def set_hook(custom_hook: "SessionHook") -> None:
    """Overwrite the current hook with the passed hook."""
    from smdebug.core.hook import BaseHook  # prevent circular imports

    if not isinstance(custom_hook, BaseHook):
        raise TypeError(f"custom_hook={custom_hook} must be type SessionHook")

    global _ts_hook
    _ts_hook = custom_hook


def del_hook() -> None:
    """ Set the hook singleton to None. """
    global _ts_hook
    _ts_hook = None
