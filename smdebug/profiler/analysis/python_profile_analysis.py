# Standard Library
import os
import pstats

# First Party
from smdebug.profiler.profiler_constants import CONVERT_TO_MICROSECS, PYTHON_STATS_FILENAME


class PythonProfileRecord:
    def __init__(
        self, step, start_time_since_epoch_in_micros, end_time_since_epoch_in_micros, stats_path
    ):
        """A class used to represent the metadata for python profiling done on a single step or before step 0.
        Used so that users can easily filter through which steps they want python profiling stats of.

        :param step: The step that was profiled. -1 if before step 0.
        :param start_time_since_epoch_in_micros: The UTC time (in microseconds) at which profiling started for this step.
        :param end_time_since_epoch_in_micros: The UTC time (in microseconds) at which profiling finished for this step.
        :param stats_path The path to the dumped python stats resulting from profiling this step.
        """
        self.step = step
        self.start_time_since_epoch_in_micros = start_time_since_epoch_in_micros
        self.end_time_since_epoch_in_micros = end_time_since_epoch_in_micros
        self.stats_path = stats_path

    def in_time_interval(self, start_time_since_epoch_in_micros, end_time_since_epoch_in_micros):
        """Returns whether this record is in the provided time interval.
        This is defined as whether there is any overlap between the time interval
        of the step and the provided time interval.
        """
        return (
            start_time_since_epoch_in_micros
            <= self.start_time_since_epoch_in_micros
            <= end_time_since_epoch_in_micros
            or start_time_since_epoch_in_micros
            <= self.end_time_since_epoch_in_micros
            <= end_time_since_epoch_in_micros
        )

    def in_step_interval(self, start_step, end_step):
        """Returns whether this is in the provided step interval.
        """
        return start_step <= self.step < end_step


class FunctionStats:
    """Class used to represent a single function that was profiled and stats pertaining to this function.
    Processes the stats dictionary's (key, value) pair to get the function name and the specific stats.
    Key is a tuple of (filename, lineno, function).
    Value is a tuple of (prim_calls, total_calls, total_time, cumulative_time, callers). See below for details.
    ...

    Attributes
    ----------
    function_name : str
        The full function name, derived from the key tuple. Defined as filename:lineno(function).
    prim_calls: int
        The number of primitive (non-recursive) calls to this function.
    total_calls: int
        The total number of calls to this function.
    total_time: int
        The total amount of time spent in the scope of this function alone, in seconds.
    cumulative_time: int
        The total amount of time spent in the scope of this function and in the scope of all other functions
        that this function calls, in seconds.
    callers: list of str
        The list of functions that call this function. Organized as a list of function names, which follow the above
        format for function_name: filename:lineno(function)
    """

    def __init__(self, key, value):
        self.function_name = pstats.func_std_string(key)
        self.prim_calls, self.total_calls, self.total_time, self.cumulative_time, callers = value
        self.callers = [pstats.func_std_string(k) for k in callers.keys()]


class PythonProfileAnalysis:
    def __init__(self, python_profile_folder):
        """Analysis class that takes in path to python profile folder, loads metadata of the profiling done for each
        step, and provides functions for analysis on this profiling.
        ...

        Attributes
        ----------
        records: list of PythonProfileRecord
            List of records, which holds the metadata for each instance of profiling (one per step).
        """
        self.records = []
        for folder in os.listdir(python_profile_folder):
            start_time, end_time, _, step = folder.split("_")
            stats_path = os.path.join(python_profile_folder, folder, PYTHON_STATS_FILENAME)
            self.records.append(
                PythonProfileRecord(int(step), float(start_time), float(end_time), stats_path)
            )
        self.records.sort(key=lambda x: x.step)  # sort records by step.

    def fetch_python_profile_stats_by_time(self, start_time, end_time):
        """API function to fetch stats based on time interval. See `_fetch_python_profile_stats` for more info.
        Returns `None` if no such stats exist.
        """
        start_time_since_epoch_in_micros = start_time * CONVERT_TO_MICROSECS
        end_time_since_epoch_in_micros = end_time * CONVERT_TO_MICROSECS
        requested_files = [
            record.stats_path
            for record in self.records
            if record.in_time_interval(
                start_time_since_epoch_in_micros, end_time_since_epoch_in_micros
            )
        ]
        if not requested_files:
            return None
        return self._fetch_python_profile_stats(requested_files)

    def fetch_python_profile_stats_by_step(self, start_step, end_step):
        """API function to fetch stats based on step interval. See `_fetch_python_profile_stats` for more info.
        Returns `None` if no such stats exist.
        """
        requested_files = [
            record.stats_path
            for record in self.records
            if record.in_step_interval(start_step, end_step)
        ]
        if not requested_files:
            return None
        return self._fetch_python_profile_stats(requested_files)

    def fetch_pre_step_zero_profile_stats(self):
        """API function that fetches stats from profiling until step 0.
        """
        return self.fetch_python_profile_stats_by_step(-1, 0)

    def _fetch_python_profile_stats(self, filenames):
        """Aggregate the stats files corresponding to the requested interval.
        Then returns a list of `FunctionStats` objects (each holds the stats for a called function during profiling).
        """
        ps = pstats.Stats()
        for filename in filenames:
            ps.add(filename)
        return [FunctionStats(k, v) for k, v in ps.stats.items()]

    def list_python_profile_stats(self):
        """API function that the list of records, which holds the metadata for each instance of profiling
        (one per step).
        """
        return self.records
