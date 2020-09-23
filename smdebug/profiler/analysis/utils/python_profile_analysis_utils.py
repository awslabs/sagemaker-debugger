# Standard Library
import pstats
from enum import Enum

# Third Party
import pandas as pd

# First Party
from smdebug.profiler.python_profile_utils import (
    PythonProfileModes,
    StepPhase,
    str_to_python_profile_mode,
)


class Metrics(Enum):
    """
    Enum to describe the types of metrics recorded in cProfile profiling.
    """

    # total amount of time spent in the scope of this function alone, in seconds.
    TOTAL_TIME = "tottime"

    # total amount of time spent in the scope of this function and in the scope of all other functions that this
    # function calls, in seconds.
    CUMULATIVE_TIME = "cumtime"

    # number of primitive (non-recursive) calls to this function
    PRIMITIVE_CALLS = "pcalls"

    # total number of calls to this function, recursive or non-recursive.
    TOTAL_CALLS = "ncalls"


class StepPythonProfileStats:
    """
    Class that represents the metadata for a single instance of profiling: before step 0, during a step, between steps,
    end of script, etc. Used so that users can easily filter through which exact portion of their session that
    they want profiling stats of. In addition, printing this class will result in a dictionary of the attributes and
    its corresponding values.

    ...

    Attributes
    ----------
    profiler_name: str
        The name of the profiler used to generate this stats file, cProfile or pyinstrument
    framework: str
        The machine learning framework used in training.
    node_id: str
        The node ID of the node used in the session.
    start_mode: str
        The training phase (TRAIN/EVAL/GLOBAL) at which profiling started.
    start_phase: str
        The step phase (start of step, end of step, etc.) at which python profiling was started.
    start_step: float
        The step at which python profiling was started. -1 if profiling before step 0.
    start_time_since_epoch_in_micros: int
        The UTC time (in microseconds) at which profiling started for this step.
    end_mode: str
        The training phase (TRAIN/EVAL/GLOBAL) at which profiling was stopped.
    end_step: float
        The step at which python profiling was stopped. Infinity if end of script.
    end_phase: str
        The step phase (start of step, end of step, etc.) at which python profiling was stopped.
    end_time_since_epoch_in_micros: int
        The UTC time (in microseconds) at which profiling finished for this step.
    stats_path: str
        The path to the dumped python stats or html resulting from profiling this step.
            """

    def __init__(self, framework, profiler_name, node_id, stats_dir, stats_path):
        start_metadata, end_metadata = stats_dir.split("_")
        start_mode, start_step, start_phase, start_time_since_epoch_in_micros = start_metadata.split(
            "-"
        )
        end_mode, end_step, end_phase, end_time_since_epoch_in_micros = end_metadata.split("-")

        self.profiler_name = profiler_name
        self.framework = framework
        self.node_id = node_id

        self.start_mode = PythonProfileModes(str_to_python_profile_mode(start_mode))
        self.start_step = -1 if start_step == "*" else int(start_step)
        self.start_phase = StepPhase(start_phase)
        self.start_time_since_epoch_in_micros = float(start_time_since_epoch_in_micros)

        self.end_mode = PythonProfileModes(str_to_python_profile_mode(end_mode))
        self.end_step = float("inf") if end_step == "*" else int(end_step)
        self.end_phase = StepPhase(end_phase)
        self.end_time_since_epoch_in_micros = float(end_time_since_epoch_in_micros)

        self.stats_path = stats_path

    def has_start_and_end_mode(self, start_mode, end_mode):
        return self.start_mode == start_mode and self.end_mode == end_mode

    def in_time_interval(self, start_time_since_epoch_in_micros, end_time_since_epoch_in_micros):
        """Returns whether this step is in the provided time interval.
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

    def in_step_interval(self, start_step, end_step, start_phase, end_phase):
        """Returns whether this is in the provided step interval. This is defined as:
        1. This start step is greater than the provided start step and the end step is greater than the provided end
            step.
        2. If this start step equals the provided start step, verify that this start phase does not occur before the
            provided start phase.
        3. If this end step equals the provided end step, verify that this end phase does not occur after the provided
            end phase.
        """
        if start_step < self.start_step and end_step > self.end_step:
            return True
        elif start_step > self.start_step or end_step < self.end_step:
            return False
        else:
            if (
                start_step == self.start_step
                and start_phase in (StepPhase.STEP_END, StepPhase.FORWARD_PASS_END)
                and self.start_phase != start_phase
            ):
                return False
            if (
                end_step == self.end_step
                and end_phase == StepPhase.STEP_START
                and self.end_phase != end_phase
            ):
                return False
            return True

    def has_pre_step_zero_profile_stats(self):
        return self.start_phase == StepPhase.START

    def has_post_hook_close_profile_stats(self):
        return self.end_phase == StepPhase.END

    def has_node_id(self, node_id):
        return self.node_id == node_id

    def __repr__(self):
        return repr(self.__dict__)


class cProfileStats:
    """
    Class used to represent cProfile stats captured, given the pStats.Stats object of the desired interval.
    ...
    Attributes
    ----------
    ps: pstats.Stats
        The cProfile stats of Python functions as a pStats.Stats object. Useful for high level analysis like sorting
        functions by a desired metric and printing the list of profiled functions.
    function_stats_list: list of cProfileFunctionStats
        The cProfile stats of Python functions as a list of cProfileFunctionStats objects, which contain specific
        metrics corresponding to each function profiled. Parsed from the pStats.Stats object. Useful for more in
        depth analysis as it allows users physical access to the metrics for each function.
    """

    def __init__(self, ps):
        self.ps = ps
        self.function_stats_list = [cProfileFunctionStats(k, v) for k, v in ps.stats.items()]

    def print_top_n_functions(self, by, n=10):
        """Print the stats for the top n functions with respect to the provided metric.
        :param by The metric to sort the functions by. Must be one of the following from the Metrics enum: TOTAL_TIME,
            CUMULATIVE_TIME, PRIMITIVE_CALLS, TOTAL_CALLS.
        :param n The first n functions and stats to print after sorting.

        For example, to print the top 20 functions with respect to cumulative time spent in function:
        >>> from smdebug.profiler.analysis.utils.python_profile_analysis_utils import Metrics
        >>> cprofile_stats.print_top_n_function(self, Metrics.CUMULATIVE_TIME, n=20)
        """
        assert isinstance(by, Metrics), "by must be valid metric from Metrics!"
        assert isinstance(n, int), "n must be an integer!"
        self.ps.sort_stats(by.value).print_stats(n)

    def get_function_stats(self):
        """Return the function stats list as a DataFrame, where each row represents a cProfileFunctionStats object.
        """
        return pd.DataFrame([repr(function_stats) for function_stats in self.function_stats_list])


class cProfileFunctionStats:
    """Class used to represent a single profiled function and parsed cProfile stats pertaining to this function.
    Processes the stats dictionary's (key, value) pair to get the function name and the specific stats.
    Key is a tuple of (filename, lineno, function).
    Value is a tuple of (prim_calls, total_calls, total_time, cumulative_time, callers). See below for details.
    ...
    Attributes
    ----------
    function_name: str
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

    def __repr__(self):
        return repr(
            {
                "function name": self.function_name,
                "# of primitive calls": self.prim_calls,
                "# of total calls": self.total_calls,
                "total time": self.total_time,
                "cumulative time": self.cumulative_time,
            }
        )


class PyinstrumentStepStats:
    def __init__(self, html_file_path, json_stats):
        self.html_file_path = html_file_path
        self.json_stats = json_stats
