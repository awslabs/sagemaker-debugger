class RotationPolicy:
    """Configuration corresponding to rotation policy of trace event files.
    """

    def __init__(self, file_max_size, file_close_interval):
        self.file_max_size = file_max_size
        self.file_close_interval = file_close_interval


class TraceFile:
    """Configuration corresponding to trace event files.
    """

    def __init__(self, file_max_size, file_close_interval, file_open_fail_threshold):
        self.rotation_policy = RotationPolicy(file_max_size, file_close_interval)
        self.file_open_fail_threshold = file_open_fail_threshold


class ProfileRange:
    """Configuration corresponding to what batches to profile..
    """

    def __init__(self, profiler_type, profiler_start, profiler_end):
        self.profile_type = profiler_type
        self.profiler_start = profiler_start
        self.profiler_end = profiler_end


class ProfilerConfig:
    """Overall profiler configuration
    """

    def __init__(
        self,
        local_path,
        file_max_size,
        file_close_interval,
        file_open_fail_threshold,
        profile_type,
        profiler_start,
        profiler_end,
    ):
        """
        :param local_path:
        :param file_max_size: Timestamp in UTC for when profiling should start.
        :param file_close_interval: Duration in seconds for how long profiling should be done.
        :param file_open_fail_threshold
        :param profile_type Type of profile range to profile. Must be "steps" or "time" (or None if no profiling will take place).
        :param profiler_start The step/time that profiling should start.
        :param profiler_end The step/time that profiling should end.
        """
        self.local_path = local_path
        self.trace_file = TraceFile(file_max_size, file_close_interval, file_open_fail_threshold)
        self.profile_range = ProfileRange(profile_type, profiler_start, profiler_end)
