# Standard Library
import json
import os
import pstats

# First Party
from smdebug.profiler.python_profile_utils import CPROFILE_NAME, PYINSTRUMENT_NAME
from smdebug.profiler.python_profiler import (
    CPROFILE_STATS_FILENAME,
    PYINSTRUMENT_HTML_FILENAME,
    PYINSTRUMENT_JSON_FILENAME,
)

ALLOWED_PYTHON_STATS_FILES = {
    CPROFILE_NAME: [CPROFILE_STATS_FILENAME],
    PYINSTRUMENT_NAME: [PYINSTRUMENT_JSON_FILENAME, PYINSTRUMENT_HTML_FILENAME],
}


def validate_python_profiling_stats(python_stats_dir, profiler_name, expected_stats_dir_count):
    allowed_files = ALLOWED_PYTHON_STATS_FILES[profiler_name]

    # Test that directory and corresponding files exist.
    assert os.path.isdir(python_stats_dir)

    for node_id in os.listdir(python_stats_dir):
        node_dir_path = os.path.join(python_stats_dir, node_id)
        stats_dirs = os.listdir(node_dir_path)
        assert len(stats_dirs) == expected_stats_dir_count

        for stats_dir in stats_dirs:
            # Validate that the expected files are in the stats dir
            stats_dir_path = os.path.join(node_dir_path, stats_dir)
            stats_files = os.listdir(stats_dir_path)
            assert set(stats_files) == set(allowed_files)

            # Validate the actual stats files
            for stats_file in stats_files:
                stats_path = os.path.join(stats_dir_path, stats_file)
                if stats_file == CPROFILE_STATS_FILENAME:
                    assert pstats.Stats(stats_path)
                elif stats_file == PYINSTRUMENT_JSON_FILENAME:
                    with open(stats_path, "r") as f:
                        assert json.load(f)
