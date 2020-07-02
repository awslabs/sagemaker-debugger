# Standard Library
import cProfile
import os
import pstats
import time

# First Party
from smdebug.core.locations import TraceFileLocation
from smdebug.core.logger import get_logger
from smdebug.profiler.profiler_constants import CONVERT_TO_MICROSECS, PYTHON_STATS_FILENAME


class PythonProfiler:
    def __init__(self, base_folder, framework):
        """Higher level class to manage execution of python profiler, dumping of python stats, and retrieval
        of stats based on time or step intervals.
        ...

        Attributes
        ----------
        base_folder: str
            The base folder path for profiling, retrieved from the profiler config parser.
        framework: str
            The name of the framework associated with the hook that the profiler is being run in.
        _profiler: cProfile.Profiler
            The python profiler object. Instantiated once, but enabled/disabled to create individual stats files.
        _step: int
            If the python profiler is running, this is the step that it is profiling. Otherwise, this is `None`.
        _start_time_since_epoch_in_micros: int
            If the python profiler is running, this is the UTC time (in microseconds) at which it started profiling.
            Otherwise, this is `None`.
        _is_profiling: bool
            Whether the profiler is currently running now or not.
        """
        self.base_folder = base_folder
        self.framework = framework
        self._profiler = cProfile.Profile()
        self._reset_profiler()

    def _reset_profiler(self):
        """Reset profiler and corresponding attributes to defaults
        """
        self._profiler = cProfile.Profile()
        self._step, self._start_time_since_epoch_in_micros, self._is_profiling = None, None, False

    def start_profiling(self, start_step=-1):
        """Start the python profiler with the provided start step.
        If start step is -1, then this is profiling from import time to step 0.
        """
        self._step = start_step
        self._start_time_since_epoch_in_micros = time.time() * CONVERT_TO_MICROSECS
        self._is_profiling = True
        self._profiler.enable()

    def stop_profiling(self):
        """Stop the python profiler.
        Dump the python stats for this step with a file path dependent on the base folder, framework, time and step.
        Append a record of this step's profiling with the corresponding metadata.
        Reset the attributes to prepare for the (possibly) next time we profile.
        """
        if not self._is_profiling:
            return

        self._profiler.disable()

        current_time_since_epoch_in_micros = time.time() * CONVERT_TO_MICROSECS
        stats_dir = TraceFileLocation.get_python_profiling_stats_dir(
            self.base_folder,
            self.framework,
            self._step,
            self._start_time_since_epoch_in_micros,
            current_time_since_epoch_in_micros,
        )
        stats_path = os.path.join(stats_dir, PYTHON_STATS_FILENAME)
        get_logger("smdebug-profiler").info(f"Dumping python profiler stats to {stats_path}.")
        pstats.Stats(self._profiler).dump_stats(stats_path)

        self._reset_profiler()
