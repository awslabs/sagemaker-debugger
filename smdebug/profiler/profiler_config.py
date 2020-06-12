# Standard library
# Standard Library
import time

# First Party
# First party
from smdebug.profiler.profiler_constants import PROFILER_DURATION_DEFAULT


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

    def __init__(self, profiler_type, profiler_start, profile_length, profiler_end):
        self.profile_type = profiler_type
        self.profiler_start = profiler_start
        self.profile_length = profile_length
        self.profiler_end = profiler_end

    def can_enable_profiling(self, current_step):
        if self.profile_type == "steps":
            return current_step >= self.profiler_start and current_step <= self.profiler_end
        elif self.profile_type == "time":
            current_time = time.time()
            if not self.profiler_start:
                self.profiler_start = current_time
                self.profiler_end = self.profiler_start + self.profile_length
            return current_time >= self.profiler_start and current_time <= self.profiler_end
        return False

    def can_disable_profiling(self, current_step):
        return not self.can_enable_profiling() or self.profiler_end != PROFILER_DURATION_DEFAULT


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
        profile_length,
        profiler_end,
    ):
        """
        :param local_path: path where profiler events have to be saved.
        :param file_max_size: Max size a trace file can be, before being rotated.
        :param file_close_interval: Interval in seconds from the last close, before being rotated.
        :param file_open_fail_threshold: Number of times to attempt to open a trace fail before marking the writer as unhealthy.
        :param profile_type Type of profile range to profile. Must be "steps" or "time" (or None if no profiling will take place).
        :param profiler_start The step/time that profiling should start. Time in seconds (UTC).
        :param profile_length The length of profiling in steps or time.
        :param profiler_end The step/time that profiling should end.
        """
        self.local_path = local_path
        self.trace_file = TraceFile(file_max_size, file_close_interval, file_open_fail_threshold)
        self.profile_range = ProfileRange(
            profile_type, profiler_start, profile_length, profiler_end
        )
