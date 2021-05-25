# First Party


class ProfilerContextManager(object):
    def __init__(self, hook, mode):
        self.hook = hook
        self.mode = mode

    def __enter__(self):

        self.hook.profiling_start_batch(self.mode)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.hook.profiling_end_batch(self.mode)
        self.hook.is_profiler_enabled_for_native_training = False
