# First Party
from smdebug.core.modes import ModeKeys


class ProfilerContextManager(object):
    def __init__(self, hook, mode=ModeKeys.TRAIN):
        self.hook = hook
        self.mode = mode

    def __enter__(self):
        self.hook.is_profiler_enabled_for_native_training = True
        self.hook.profiling_start_batch(self.mode)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.hook.profiling_end_batch(self.mode)
        self.hook.is_profiler_enabled_for_native_training = False
