# Standard Library
import json
import os
import pstats
from collections import defaultdict

# Third Party
import pandas as pd

# First Party
from smdebug.core.logger import get_logger
from smdebug.profiler.analysis.python_stats_reader import (
    LocalPythonStatsReader,
    S3PythonStatsReader,
)
from smdebug.profiler.analysis.utils.python_profile_analysis_utils import (
    PyinstrumentStepStats,
    StepPhase,
    StepPythonProfileStats,
    cProfileStats,
)
from smdebug.profiler.profiler_constants import (
    CONVERT_TO_MICROSECS,
    CPROFILE_NAME,
    PYINSTRUMENT_NAME,
)
from smdebug.profiler.python_profile_utils import PythonProfileModes, python_profile_mode_to_str


class PythonProfileAnalysis:
    name = ""  # placeholder

    def __init__(self, local_profile_dir="/tmp/python_stats", s3_path=None):
        """Analysis class that takes in path to the profile directory, and sets up the python stats reader, which
        fetches metadata of the python profiling done for each step. Also provides functions for analysis on this
        profiling, such as fetching stats by a specific step or time interval.

        If s3_path is provided, the S3PythonStatsReader is used and local_profile_dir will represent the local
        directory path that the reader will create the stats directory and then download the stats to.
        Otherwise, LocalPythonStatsReader is used and local_profile_dir represents the path to the stats directory,
        which already holds the stats.

        ...

        Attributes
        ----------
        python_stats_reader: PythonStatsReader
            The reader to use for loading the python stats.
        python_profile_stats: list of StepPythonProfileStats
            List of stats for each step profiled.
        """
        self.python_stats_reader = (
            S3PythonStatsReader(local_profile_dir, s3_path)
            if s3_path
            else LocalPythonStatsReader(local_profile_dir)
        )
        self._refresh_python_profile_stats(True)

    def _refresh_python_profile_stats(self, refresh_stats):
        """Helper function to load in the most recent python stats via the python stats reader.
        """
        if refresh_stats:
            get_logger().info("Refreshing python profile stats.")
            self.python_profile_stats = self.python_stats_reader.load_python_profile_stats()

    def _fetch_profile_stats_by_node_id(self, node_id):
        """Helper function to filter profile stats by node ID. If no specific node ID is provided, pick the first
        stats object's node ID.
        """
        if len(self.python_profile_stats) == 0:
            return []
        if node_id == "any":
            node_id = self.python_profile_stats[0].node_id
        return [
            step_stats for step_stats in self.python_profile_stats if step_stats.node_id == node_id
        ]

    def _aggregate_stats(self, requested_stats):
        """
        Helper function to the requested stats by the user. To be overriden in the subclass as this is profiler
        dependent.
        """
        return requested_stats  # placeholder

    def fetch_profile_stats_by_step(
        self,
        start_step,
        end_step=None,
        mode=PythonProfileModes.TRAIN,
        start_phase=StepPhase.STEP_START,
        end_phase=StepPhase.STEP_END,
        node_id="any",
        refresh_stats=True,
    ):
        """API function to fetch stats based on step interval.
        """
        self._refresh_python_profile_stats(refresh_stats)

        if end_step is None:
            end_step = start_step

        requested_stats = [
            step_stats
            for step_stats in self._fetch_profile_stats_by_node_id(node_id)
            if step_stats.in_step_interval(start_step, end_step, start_phase, end_phase)
            and step_stats.has_start_and_end_mode(mode, mode)
        ]
        return self._aggregate_stats(requested_stats)

    def fetch_profile_stats_by_time(
        self,
        start_time_since_epoch_in_secs,
        end_time_since_epoch_in_secs,
        node_id="any",
        refresh_stats=True,
    ):
        """API function to fetch stats based on time interval.
        """
        self._refresh_python_profile_stats(refresh_stats)
        start_time_since_epoch_in_micros = start_time_since_epoch_in_secs * CONVERT_TO_MICROSECS
        end_time_since_epoch_in_micros = end_time_since_epoch_in_secs * CONVERT_TO_MICROSECS
        requested_stats = [
            step_stats
            for step_stats in self._fetch_profile_stats_by_node_id(node_id)
            if step_stats.in_time_interval(
                start_time_since_epoch_in_micros, end_time_since_epoch_in_micros
            )
        ]
        return self._aggregate_stats(requested_stats)

    def fetch_profile_stats_between_modes(
        self, start_mode, end_mode, node_id="any", refresh_stats=True
    ):
        """API function that fetches stats with the provided start and end mode.
        """
        self._refresh_python_profile_stats(refresh_stats)
        requested_stats = [
            step_stats
            for step_stats in self._fetch_profile_stats_by_node_id(node_id)
            if step_stats.has_start_and_end_mode(start_mode, end_mode)
        ]
        return self._aggregate_stats(requested_stats)

    def fetch_pre_step_zero_profile_stats(self, node_id="any", refresh_stats=True):
        """API function that fetches stats from profiling until step 0.
        """
        self._refresh_python_profile_stats(refresh_stats)
        requested_stats = [
            step_stats
            for step_stats in self._fetch_profile_stats_by_node_id(node_id)
            if step_stats.has_pre_step_zero_profile_stats()
        ]
        return self._aggregate_stats(requested_stats)

    def fetch_post_hook_close_profile_stats(self, node_id="any", refresh_stats=True):
        """API function that fetches stats from profiling after the hook is closed.
        """
        self._refresh_python_profile_stats(refresh_stats)
        requested_stats = [
            step_stats
            for step_stats in self._fetch_profile_stats_by_node_id(node_id)
            if step_stats.has_post_hook_close_profile_stats()
        ]
        return self._aggregate_stats(requested_stats)

    def list_profile_stats(self, refresh_stats=True):
        """API function that returns a DataFrame of the python profile stats, where each row holds the metadata for
        each instance of profiling and the corresponding stats file (one per step).

        The columns of this DataFrame include:
            - profiler_name: The name of the profiler used to generate this stats file, cProfile or pyinstrument
            - framework: The machine learning framework used in training.
            - start_time_since_epoch_in_micros: The UTC time (in microseconds) at which profiling started for this step.
            - end_time_since_epoch_in_micros: The UTC time (in microseconds) at which profiling finished for this step.
            = node_id The node ID of the node used in the session.
            - start_phase The phase at which python profiling was started.
            - start_step: The step at which python profiling was started. -1 if before step 0.
            - end_phase The phase at which python profiling was stopped.
            - end_step: The step at which python profiling was stopped.
            - stats_path The path to the dumped python stats resulting from profiling this step.
        """
        self._refresh_python_profile_stats(refresh_stats)
        if len(self.python_profile_stats) == 0:
            print("No stats were found!")
            return pd.DataFrame()
        df = pd.DataFrame(list(map(lambda x: vars(x), self.python_profile_stats)))
        df["start_mode"] = df["start_mode"].map(python_profile_mode_to_str)
        df["end_mode"] = df["end_mode"].map(python_profile_mode_to_str)
        return df

    def list_available_node_ids(self, refresh_stats=True):
        """API function to list the available node IDs we have python profiling stats for.
        """
        self._refresh_python_profile_stats(refresh_stats)
        all_node_ids = map(lambda x: x.node_id, self.python_profile_stats)
        unique_node_ids = list(set(all_node_ids))
        unique_node_ids.sort()
        return unique_node_ids


class cProfileAnalysis(PythonProfileAnalysis):
    """Analysis class used specifically for python profiling with cProfile
    """

    name = CPROFILE_NAME

    def _refresh_python_profile_stats(self, refresh_stats):
        """Helper function to load in the most recent python stats via the python stats reader.
        Filters out any stats not generated by cProfile.
        """
        super()._refresh_python_profile_stats(refresh_stats)
        self.python_profile_stats = list(
            filter(lambda x: x.profiler_name == CPROFILE_NAME, self.python_profile_stats)
        )

    def _aggregate_stats(self, stats, refresh_stats=True):
        """Aggregate the stats files into one pStats.Stats object corresponding to the requested interval.
        Then returns a `cProfileStats` object (which holds the pStats.Stats object and parsed stats for each called
        function in these steps).
        """
        ps = pstats.Stats()

        if len(stats) == 0:
            print("No stats were found for the requested interval!")
            return None

        for step_stats in stats:
            ps.add(step_stats.stats_path)
        return cProfileStats(ps)

    def fetch_profile_stats_by_training_phase(self, node_id="any", refresh_stats=True):
        """API function that fetches and aggregates stats for every possible combination of start and end mode.

        For example, if training and validation are done while detailed profiling is enabled, the combinations are:
            - (PRE_STEP_ZERO, TRAIN)
            - (TRAIN, TRAIN)
            - (TRAIN, EVAL)
            - (EVAL, EVAL)
            - (EVAL, POST_HOOK_CLOSE)

        All stats files within each of these combinations are aggregated.
        """
        self._refresh_python_profile_stats(refresh_stats)
        training_phase_stats = {}
        for start_mode in PythonProfileModes:
            for end_mode in PythonProfileModes:
                step_stats = self.fetch_profile_stats_between_modes(
                    start_mode, end_mode, node_id=node_id, refresh_stats=False
                )
                if step_stats is not None:
                    training_phase_stats[(start_mode, end_mode)] = step_stats
        return training_phase_stats

    def fetch_profile_stats_by_job_phase(self, node_id="any", refresh_stats=True):
        """API function that fetches and aggregates stats by job phase.

        The job phases are:
            - initialization: profiling until step 0 (pre-step zero profiling)
            - training loop: training and validation
            - finalization: profiling after the hook is closed (post-hook-close profiling)
        """
        self._refresh_python_profile_stats(refresh_stats)
        training_phase_stats = self.fetch_profile_stats_by_training_phase(
            node_id=node_id, refresh_stats=False
        )
        job_phase_stats = {}
        ps = pstats.Stats()
        for (start_mode, end_mode), step_stats in training_phase_stats.items():
            if start_mode == PythonProfileModes.PRE_STEP_ZERO:
                job_phase_stats["initialization"] = step_stats
            elif end_mode == PythonProfileModes.POST_HOOK_CLOSE:
                job_phase_stats["finalization"] = step_stats
            else:
                ps.add(step_stats.ps)
        job_phase_stats["training_loop"] = cProfileStats(ps)
        return job_phase_stats


class PyinstrumentAnalysis(PythonProfileAnalysis):
    """Analysis class used specifically for python profiling with pyinstrument.
    """

    name = PYINSTRUMENT_NAME

    def _refresh_python_profile_stats(self, refresh_stats):
        """Helper function to load in the most recent python stats via the python stats reader.
        Filters out any stats not generated by pyinstrument.
        """
        super()._refresh_python_profile_stats(refresh_stats)
        self.python_profile_stats = list(
            filter(lambda x: x.profiler_name == PYINSTRUMENT_NAME, self.python_profile_stats)
        )

    def _aggregate_stats(self, stats):
        """Load and return a list of dictionaries corresponding to each step's stats file.
        """
        if len(stats) == 0:
            print("No stats were found for the requested interval!")
            return None

        aggregated_stats_dict = defaultdict(lambda: [])
        for step_stats in stats:
            aggregated_stats_dict[step_stats.start_time_since_epoch_in_micros].append(step_stats)

        aggregated_stats = []
        for aggregated_step_stats in aggregated_stats_dict.values():
            aggregated_step_stats.sort(key=lambda x: os.path.basename(x.stats_path))
            html_step_stats, json_step_stats = aggregated_step_stats
            with open(json_step_stats.stats_path, "r") as json_data:
                json_stats = json.load(json_data)
            aggregated_stats.append(PyinstrumentStepStats(html_step_stats.stats_path, json_stats))

        return aggregated_stats
