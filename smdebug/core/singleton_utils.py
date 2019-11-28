"""
Easy-to-use methods for getting the singleton SessionHook.

Sample usage:

import smdebug.(pytorch | tensorflow | mxnet) as smd
hook = smd.hook()
"""

# Standard Library
import atexit

# First Party
from smdebug.core.logger import get_logger

logger = get_logger()
_ts_hook = None


def _create_hook(json_config_path, hook_class):
    from smdebug.core.hook import BaseHook  # prevent circular imports

    if not issubclass(hook_class, BaseHook):
        raise TypeError("hook_class needs to be a subclass of BaseHook", hook_class)

    # Either returns a hook or None
    try:
        hook = hook_class.create_from_json_file(json_file_path=json_config_path)
        set_hook(custom_hook=hook)
    except FileNotFoundError:
        pass


def get_hook(*, json_config_path: str, hook_class, create_if_not_exists: bool) -> "Hook":
    """Return a singleton SessionHook or None.

    If the singleton hook exists, we return it. No questions asked, `json_config_path` is a no-op.
    Otherwise return create_from_json_file().
    """
    global _ts_hook

    # Cannot create a hook if hook_class is not passed; invalid to call isinstance(obj, None).
    if hook_class is None:
        if create_if_not_exists:
            raise ValueError("Cannot create hook because hook_class is None")
        return _ts_hook

    # If hooks exists, either return or reset it based on hook_class
    if _ts_hook:
        # Return the hook if it exists and is the right type
        if isinstance(_ts_hook, hook_class):
            return _ts_hook
        # Reset the hook if it is the wrong type (user runs Session, then Estimator in same script).
        else:
            logger.info(f"Current hook is not instance of {hook_class}, so clearing it.")
            _ts_hook = None

    # Create if the user desires
    if create_if_not_exists and not _ts_hook:
        _create_hook(json_config_path, hook_class)

    return _ts_hook


def set_hook(custom_hook: "BaseHook") -> None:
    """Overwrite the current hook with the passed hook."""
    from smdebug.core.hook import BaseHook  # prevent circular imports

    if not isinstance(custom_hook, BaseHook):
        raise TypeError(f"custom_hook={custom_hook} must be type BaseHook")

    global _ts_hook
    _ts_hook = custom_hook

    atexit.register(del_hook)


def del_hook() -> None:
    """ Set the hook singleton to None. """
    global _ts_hook
    _ts_hook = None
