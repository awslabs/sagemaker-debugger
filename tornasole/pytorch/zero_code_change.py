"""
Easy-to-use methods for getting the TornasoleHook in zero-code-change mode.
This is abstracted into its own module to prevent circular import problems.
As long as `import tornasole.pytorch as ts` is called within PT methods, no problem.

Sample usage (in AWS-PyTorch repo):

import tornasole.pytorch as ts
if ts.use_zero_code_change():
    hook = ts.get_zero_code_change_hook()
    hook.register_hook(...)
"""


from tornasole.core.save_config import SaveConfig
from tornasole.pytorch.hook import TornasoleHook

should_use = False
ts_hook = None


def set_use_zero_code_change(val: bool):
    global should_use
    should_use = val

def use_zero_code_change() -> bool:
    """Defaults to False unless set."""
    return should_use

def get_zero_code_change_hook() -> TornasoleHook:
    """Access a global variable TornasoleHook, creating if does not exist."""
    global ts_hook
    if ts_hook is None:
        print("Creating ZCC-PT TornasoleHook")
        # The default configuration will obviously need to change.
        # This is a stub implementation until we work out the details of how ZCC will
        # be implemented.
        ts_hook = TornasoleHook(
            out_dir='/tmp/zcc',
            save_config=SaveConfig()
        )
    return ts_hook