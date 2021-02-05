class ProfilerContextManager(object):
    def __init__(self, hook):
        self.hook = hook

    def __enter__(self):
        self.hook.profiling_start_batch(mode=self.hook.mode)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.hook.profiling_end_batch(mode=self.hook.mode)
