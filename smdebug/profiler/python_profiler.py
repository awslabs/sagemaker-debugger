# Standard Library
import json
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
from smdebug.profiler.profiler_constants import (
    CONVERT_TO_MICROSECS,
    CPROFILE_NAME,
    CPROFILE_STATS_FILENAME,
    PYINSTRUMENT_HTML_FILENAME,
    PYINSTRUMENT_JSON_FILENAME,
    PYINSTRUMENT_NAME,
    PYTHON_PROFILING_START_STEP_DEFAULT,
)
from smdebug.profiler.python_profile_utils import (
    PythonProfileModes,
    cProfileTimer,
    cpu_time,
    off_cpu_time,
    python_profile_mode_to_str,
    total_time,
)


class PythonProfiler:
    _name = ""  # placeholder

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
        self._start_mode = None
        self._start_step = None
        self._start_phase = None
        self._start_time_since_epoch_in_micros = None
        self._is_profiling = None

    def _enable_profiler(self):
        """Enable the profiler (to be implemented in subclass, where the actual profiler is defined).
        """

    def _disable_profiler(self):
        """Disable the profiler (to be implemented in subclass, where the actual profiler is defined).
        """

    def _dump_stats(self, stats_path):
        """Dump the stats to the provided path (to be implemented in subclass, where the actual profiler is defined).
        """

    def start_profiling(
        self, start_phase, start_mode=PythonProfileModes.PRE_STEP_ZERO, start_step="*"
    ):
        """Start the python profiler with the provided start phase and start step and start mode.
        Start phase must be one of the specified step phases in StepPhase.
        Start mode must be one of the specified modes in ModeKeys.
        If start step is *, then this is profiling until step 0.
        """
        self._start_mode = start_mode
        self._start_step = start_step
        self._start_phase = start_phase
        self._start_time_since_epoch_in_micros = time.time() * CONVERT_TO_MICROSECS
        self._is_profiling = True
        self._enable_profiler()

    def stop_profiling(self, end_phase, end_mode=PythonProfileModes.POST_HOOK_CLOSE, end_step="*"):
        """Stop the python profiler with the provided end phase and end step and end mode.
        End phase must be one of the specified step phases in StepPhase.
        End mode must be one of the specified modes in ModeKeys.
        Dump the python stats for this step with a file path dependent on the base folder, framework, time and step.
        Append a record of this step's profiling with the corresponding metadata.
        Reset the attributes to prepare for the (possibly) next time we profile.
        If end step is *, then this is profiling until the end of the script.
        """
        if not self._is_profiling:
            return

        self._disable_profiler()

        end_time_since_epoch_in_micros = time.time() * CONVERT_TO_MICROSECS
        stats_dir = TraceFileLocation.get_python_profiling_stats_dir(
            self._base_folder,
            self._name,
            self._framework,
            python_profile_mode_to_str(self._start_mode),
            self._start_step,
            self._start_phase.value,
            self._start_time_since_epoch_in_micros,
            python_profile_mode_to_str(end_mode),
            end_step,
            end_phase.value,
            end_time_since_epoch_in_micros,
        )
        self._dump_stats(stats_dir)

        self._reset_profiler()

    @staticmethod
    def get_python_profiler(profiler_config, framework):
        base_folder = profiler_config.local_path
        python_profiling_config = profiler_config.python_profiling_config
        if python_profiling_config.profiler_name == CPROFILE_NAME:
            cprofile_timer = python_profiling_config.cprofile_timer
            if cprofile_timer == cProfileTimer.DEFAULT:
                return cProfileDefaultPythonProfiler(base_folder, framework)
            else:
                return cProfilePythonProfiler(base_folder, framework, cprofile_timer)
        else:
            return PyinstrumentPythonProfiler(base_folder, framework)


class cProfilePythonProfiler(PythonProfiler):
    """Higher level class to oversee profiling specific to cProfile, Python's native profiler in .
    This is also the default Python profiler used if profiling is enabled.
    """

    _name = CPROFILE_NAME
    timer_name_to_function = {
        cProfileTimer.TOTAL_TIME: total_time,
        cProfileTimer.CPU_TIME: cpu_time,
        cProfileTimer.OFF_CPU_TIME: off_cpu_time,
    }

    def __init__(self, base_folder, framework, cprofile_timer):
        super().__init__(base_folder, framework)
        if cprofile_timer == "default":
            self.cprofile_timer = None  # will be set in subclass
        else:
            self.cprofile_timer = self.timer_name_to_function[cprofile_timer]

    def _enable_profiler(self):
        """Enable the cProfile profiler with the current cProfile timer.
        """
        self._profiler = cProfileProfiler(self.cprofile_timer)
        self._profiler.enable()

    def _disable_profiler(self):
        """Disable the cProfile profiler.
        """
        self._profiler.disable()

    def _dump_stats(self, stats_dir):
        """Dump the stats by via pstats object to a file `python_stats` in the provided stats directory.
        """
        stats_file_path = os.path.join(stats_dir, CPROFILE_STATS_FILENAME)
        get_logger().info(f"Dumping cProfile stats to {stats_file_path}.")
        pstats.Stats(self._profiler).dump_stats(stats_file_path)


class cProfileDefaultPythonProfiler(cProfilePythonProfiler):
    """Higher level class on cProfilePythonProfiler to manage the default case where no python profiling config is
    specified. Three steps of python profiling are done starting at PYTHON_PROFILING_START_STEP_DEFAULT. For each of
    these steps, the cProfile timer function used cycles between total_time, cpu_time and off_cpu_time.
    """

    def __init__(self, base_folder, framework):
        super().__init__(base_folder, framework, "default")

    def _enable_profiler(self):
        """Set the current cProfile timer based on the step and enable the cProfile profiler.
        """
        if self._start_step == PYTHON_PROFILING_START_STEP_DEFAULT:
            # first step of default three steps of profiling
            self.cprofile_timer = total_time
        elif self._start_step == PYTHON_PROFILING_START_STEP_DEFAULT + 1:
            # second step of default three steps of profiling
            self.cprofile_timer = cpu_time
        elif self._start_step == PYTHON_PROFILING_START_STEP_DEFAULT + 2:
            # third step of default three steps of profiling
            self.cprofile_timer = off_cpu_time
        else:
            # for pre step zero or post hook close profiling, use total_time
            self.cprofile_timer = total_time
        super()._enable_profiler()


class PyinstrumentPythonProfiler(PythonProfiler):
    """Higher level class to oversee profiling specific to Pyinstrument, a third party Python profiler.
    """

    _name = PYINSTRUMENT_NAME

    def _enable_profiler(self):
        """Enable the pyinstrument profiler.
        """
        self._profiler = PyinstrumentProfiler()
        self._profiler.start()

    def _disable_profiler(self):
        """Disable the pyinstrument profiler.
        """
        self._profiler.stop()

    def _dump_stats(self, stats_dir):
        """Dump the stats as a JSON dictionary to a file `python_stats.json` in the provided stats directory.
        """
        stats_file_path = os.path.join(stats_dir, PYINSTRUMENT_JSON_FILENAME)
        html_file_path = os.path.join(stats_dir, PYINSTRUMENT_HTML_FILENAME)
        try:
            session = self._profiler.last_session
            json_stats = JSONRenderer().render(session)
            get_logger().info(f"JSON stats collected for pyinstrument: {json_stats}.")
            with open(stats_file_path, "w") as json_data:
                json_data.write(json_stats)
            get_logger().info(f"Dumping pyinstrument stats to {stats_file_path}.")

            with open(html_file_path, "w") as html_data:
                html_data.write(self._profiler.output_html())
            get_logger().info(f"Dumping pyinstrument output html to {html_file_path}.")
        except (UnboundLocalError, AssertionError):
            # Handles error that sporadically occurs within pyinstrument.
            get_logger().info(
                f"The pyinstrument profiling session has been corrupted for: {stats_file_path}."
            )
            with open(stats_file_path, "w") as json_data:
                json.dump({"root_frame": None}, json_data)

            with open(html_file_path, "w") as html_data:
                html_data.write("An error occurred during profiling!")
