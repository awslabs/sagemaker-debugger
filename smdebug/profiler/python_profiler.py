# Standard Library
import os
import pstats
import time
from cProfile import Profile as cProfileProfiler

# Third Party
from pyinstrument import Profiler as PyinstrumentProfiler
from pyinstrument.renderers import JSONRenderer

# First Party
from smdebug.core.locations import TraceFileLocation
from smdebug.core.logger import get_logger
from smdebug.profiler.profiler_constants import CONVERT_TO_MICROSECS


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
        _profiler: cProfile.Profiler | pyinstrument.Profiler
            The python profiler object. Enabled/disabled to create individual stats files. Instantiated for every
            step profiled. This will be set in subclass, depending on what profiler is used (cProfile, pyinstrument,
            etc.)
        _step: int
            If the python profiler is running, this is the step that it is profiling. Otherwise, this is `None`.
        _start_time_since_epoch_in_micros: int
            If the python profiler is running, this is the UTC time (in microseconds) at which it started profiling.
            Otherwise, this is `None`.
        _is_profiling: bool
            Whether the profiler is currently running now or not.
        """
        self._base_folder = base_folder
        self._framework = framework
        self._profiler = None  # placeholder
        self._reset_profiler()

    def _reset_profiler(self):
        """Reset attributes to defaults
        """
        self._step, self._start_time_since_epoch_in_micros, self._is_profiling = None, None, False
        self._start_stepphase = ""

    def _enable_profiler(self):
        """Enable the profiler (to be implemented in subclass, where the actual profiler is defined).
        """

    def _disable_profiler(self):
        """Disable the profiler (to be implemented in subclass, where the actual profiler is defined).
        """

    def _dump_stats(self, stats_path):
        """Dump the stats to the provided path (to be implemented in subclass, where the actual profiler is defined).
        """

    def _name(self):
        return "default"

    def _stats_filename(self):
        # this is default value
        return "python_stats"

    def start_profiling(self, start_step=-1, step_phase=""):
        """Start the python profiler with the provided start step.
        If start step is -1, then this is profiling from import time to step 0.
        """
        self._step = start_step
        self._start_time_since_epoch_in_micros = time.time() * CONVERT_TO_MICROSECS
        self._is_profiling = True
        self._start_stepphase = step_phase
        self._enable_profiler()

    def stop_profiling(self, step_phase=""):
        """Stop the python profiler.
        Dump the python stats for this step with a file path dependent on the base folder, framework, time and step.
        Append a record of this step's profiling with the corresponding metadata.
        Reset the attributes to prepare for the (possibly) next time we profile.
        """
        if not self._is_profiling:
            return

        self._disable_profiler()

        current_time_since_epoch_in_micros = time.time() * CONVERT_TO_MICROSECS
        step_phase_string = ""
        if self._start_stepphase != "" or step_phase != "":
            step_phase_string = f"{self._start_stepphase}-{step_phase}"
        stats_dir = TraceFileLocation.get_python_profiling_stats_dir(
            self._base_folder,
            self._framework,
            self._name(),
            self._step,
            self._start_time_since_epoch_in_micros,
            current_time_since_epoch_in_micros,
            step_phase_string,
        )
        self._dump_stats(os.path.join(stats_dir, self._stats_filename()))

        self._reset_profiler()

    @staticmethod
    def get_python_profiler(use_pyinstrument, base_folder, framework):
        python_profiler_class = (
            PyinstrumentPythonProfiler if use_pyinstrument else cProfilePythonProfiler
        )
        return python_profiler_class(base_folder, framework)


class cProfilePythonProfiler(PythonProfiler):
    """Higher level class to oversee profiling specific to cProfile, Python's native profiler.
    This is also the default Python profiler used if profiling is enabled.
    """

    def _reset_profiler(self):
        """Reset profiler and corresponding attributes to defaults
        """
        super()._reset_profiler()
        self._profiler = cProfileProfiler(self._total_time)

    def _name(self):
        return "cProfile"

    def _total_time(self):
        times = os.times()
        return times.elapsed

    def _off_cpu_time(self):
        times = os.times()
        return times.elapsed - (times.system + times.user)

    def _stats_filename(self):
        # this is default value
        return "python_stats"

    def _enable_profiler(self):
        """Enable the cProfile profiler.
        """
        self._profiler.enable()

    def _disable_profiler(self):
        """Disable the cProfile profiler.
        """
        self._profiler.disable()

    def _dump_stats(self, stats_file_path):
        """Dump the stats by via pstats object to a file `python_stats` in the provided directory
        """
        get_logger("smdebug-profiler").info(f"Dumping cProfile stats to {stats_file_path}.")
        pstats.Stats(self._profiler).dump_stats(stats_file_path)

class PyinstrumentPythonProfiler(PythonProfiler):
    """Higher level class to oversee profiling specific to Pyinstrument, a third party Python profiler.
    """

    def _reset_profiler(self):
        """Reset profiler and corresponding attributes to defaults
        """
        super()._reset_profiler()
        self._profiler = PyinstrumentProfiler()

    def _name(self):
        return "pyinstrument"

    def _stats_filename(self):
        return "python_stats.json"

    def _enable_profiler(self):
        """Enable the pyinstrument profiler.
        """
        self._profiler.start()

    def _disable_profiler(self):
        """Disable the pyinstrument profiler.
        """
        self._profiler.stop()

    def _dump_stats(self, stats_file_path):
        """Dump the stats as a JSON dictionary to a file `python_stats.json` with the provided file path.
        """
        get_logger("smdebug-profiler").info(f"Dumping pyinstrument stats to {stats_file_path}.")

        session = self._profiler.last_session
        renderer = JSONRenderer()
        with open(stats_file_path, "w") as f:
            f.write(renderer.render(session))
