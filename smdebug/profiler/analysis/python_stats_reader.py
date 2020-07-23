# Standard Library
import os
import shutil

# First Party
from smdebug.core.access_layer.s3handler import ListRequest, ReadObjectRequest, S3Handler, is_s3
from smdebug.core.logger import get_logger
from smdebug.profiler.analysis.utils.python_profile_analysis_utils import StepPythonProfileStats
from smdebug.profiler.profiler_constants import (
    CPROFILE_NAME,
    CPROFILE_STATS_FILENAME,
    PYINSTRUMENT_NAME,
    PYINSTRUMENT_STATS_FILENAME,
)


class PythonStatsReader:
    """Basic framework for stats reader to retrieve stats from python profiling
    """

    def __init__(self, profile_dir):
        """
        :param profile_dir: The path to the directory where the python profile stats are.
        """
        self.profile_dir = profile_dir

    def load_python_profile_stats(self):
        """Load the python profile stats. To be implemented in subclass.
        """


class S3PythonStatsReader(PythonStatsReader):
    """Higher level stats reader to download python stats from s3.
    """

    def __init__(self, profile_dir, s3_path):
        """
        :param profile_dir: The path to the directory where the profile directory is created. The stats will then
            be downloaded to this newly created directory.
        :param s3_path: The path in s3 to the base folder of the logs.
        """
        assert os.path.isdir(profile_dir), "The provided profile directory does not exist!"
        super().__init__(os.path.join(profile_dir, "python_stats"))
        self._validate_s3_path(s3_path)

    def _set_up_profile_dir(self):
        """Recreate the profile directory, clearing any files that were in it.
        """
        shutil.rmtree(self.profile_dir, ignore_errors=True)
        os.makedirs(self.profile_dir)

    def _validate_s3_path(self, s3_path):
        """Validate the provided s3 path and set the bucket name and prefix.
        :param s3_path: The path in s3 to the base folder of the logs.
        """
        s3, bucket_name, base_folder = is_s3(s3_path)
        assert s3, "The provided s3 path should have the following format: s3://bucket_name/..."
        self.bucket_name = bucket_name
        self.prefix = os.path.join(base_folder, "framework")

    def _get_step_stepphase(self, step_phase_str):
        splits = step_phase_str.split("-", 1)
        step = splits[0]
        step_phase = step[1] if len(splits) > 1 else "full"
        return step, step_phase

    def load_python_profile_stats(self):
        """Load the stats in by creating the profile directory, downloading each stats directory from s3 to the
        profile directory, parsing the metadata from each stats directory name and creating a StepPythonProfileStats
        entry corresponding to the stats file in the stats directory.

        For cProfile, the stats file name is `python_stats`.
        For pyinstrument, the stats file name `python_stats.json`.
        """
        python_profile_stats = []

        self._set_up_profile_dir()

        list_request = ListRequest(Bucket=self.bucket_name, Prefix=self.prefix)
        s3_filepaths = S3Handler.list_prefix(list_request)
        object_requests = [
            ReadObjectRequest(os.path.join("s3://", self.bucket_name, s3_filepath))
            for s3_filepath in s3_filepaths
        ]
        objects = S3Handler.get_objects(object_requests)

        for full_s3_filepath, object_data in zip(s3_filepaths, objects):
            stats_file = os.path.basename(full_s3_filepath)

            if stats_file not in (CPROFILE_STATS_FILENAME, PYINSTRUMENT_STATS_FILENAME):
                continue

            profiler_name = os.path.basename(os.path.dirname(os.path.dirname(full_s3_filepath)))
            stats_dir = os.path.basename(os.path.dirname(full_s3_filepath))
            stats_dir_path = os.path.join(self.profile_dir, stats_dir)
            os.makedirs(stats_dir_path)
            stats_file_path = os.path.join(stats_dir_path, stats_file)

            with open(stats_file_path, "wb") as f:
                f.write(object_data)

            step, step_phase = _get_step_stepphase(step_phase_str)
            python_profile_stats.append(
                StepPythonProfileStats(
                    profiler_name,
                    int(step),
                    float(start_time),
                    float(end_time),
                    node_id,
                    stats_file_path,
                    step_phase,
                )
            )
        python_profile_stats.sort(
            key=lambda x: x.step
        )  # sort each step's stats by the step number.
        return python_profile_stats


class LocalPythonStatsReader(PythonStatsReader):
    """Higher level stats reader to load the python stats locally.
    """

    def __init__(self, profile_dir):
        """
        :param profile_dir: The path to the directory where the python profile stats are.
        """
        assert os.path.isdir(profile_dir), "The provided stats directory does not exist!"
        super().__init__(profile_dir)

    def load_python_profile_stats(self):
        """Load the stats in by scanning each stats directory in the profile directory, parsing the metadata from the
        stats directory name and creating a StepPythonProfileStats entry corresponding to the stats file in the
        stats directory.

        For cProfile, the stats file name is `python_stats`.
        For pyinstrument, the stats file name `python_stats.json`.
        """
        python_profile_stats = []
        for python_stat_dir in os.listdir(self.profile_dir):
            start_time, end_time, node_id, step_phase_str = python_stat_dir.split("_")
            step, step_phase = _get_step_stepphase(step_phase_str)

            stats_dir = os.path.join(self.profile_dir, python_stat_dir)
            if os.path.isfile(os.path.join(stats_dir, CPROFILE_STATS_FILENAME)):
                profiler_name = CPROFILE_NAME
                stats_file_path = os.path.join(stats_dir, CPROFILE_STATS_FILENAME)
            elif os.path.isfile(os.path.join(stats_dir, PYINSTRUMENT_STATS_FILENAME)):
                profiler_name = PYINSTRUMENT_NAME
                stats_file_path = os.path.join(stats_dir, PYINSTRUMENT_STATS_FILENAME)
            else:
                get_logger("smdebug-profiler").info(
                    f"Folder {python_stat_dir} is empty, skipping..."
                )
                continue
            python_profile_stats.append(
                StepPythonProfileStats(
                    profiler_name,
                    int(step),
                    float(start_time),
                    float(end_time),
                    node_id,
                    stats_file_path,
                )
            )
        python_profile_stats.sort(
            key=lambda x: x.step
        )  # sort each step's stats by the step number.
        return python_profile_stats
