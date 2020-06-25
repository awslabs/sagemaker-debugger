# Third Party
# First Party
from smdebug.profiler.utils import (
    get_node_id_from_system_profiler_filename,
    get_utctimestamp_us_since_epoch_from_system_profiler_file,
)


def test_get_node_id_from_system_profiler_filename():
    filename = "job-name/profiler-output/system/incremental/2020060500/1591160699.algo-1.json"
    node_id = get_node_id_from_system_profiler_filename(filename)
    assert node_id == "algo-1"


def test_get_utctimestamp_us_since_epoch_from_system_profiler_file():
    filename = "job-name/profiler-output/system/incremental/2020060500/1591160699.algo-1.json"
    timestamp = get_utctimestamp_us_since_epoch_from_system_profiler_file(filename)
    assert timestamp == 1591160699000000

    filename = "job-name/profiler-output/system/incremental/2020060500/1591160699.lgo-1.json"
    timestamp = get_utctimestamp_us_since_epoch_from_system_profiler_file(filename)
    assert timestamp is None
