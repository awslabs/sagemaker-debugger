# Third Party
# First Party
from smdebug.profiler.utils import (
    get_node_id_from_system_profiler_filename,
    get_utctimestamp_us_since_epoch_from_system_profiler_file,
    read_tf_profiler_metadata_file,
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


def test_tf_profiler_metadata_file_read():
    # (key, value) = (file_path : expected result from read_metadata)
    file_path_list = {
        "./tests/profiler/resources/tfprofiler_timeline_traces/framework/tensorflow/"
        "detailed_profiling/2020080513/000000000/plugins/profile/2020_08_05_13_37_44/"
        "8c859046be41.ant.amazon.com.trace.json.gz": (
            "80807-8c859046be41.ant.amazon.com",
            "1596659864545103",
            "1596659864854168",
        ),
        "./tests/profiler/resources/tfprofiler_local_missing_metadata/framework/tensorflow/"
        "detailed_profiling/2020080513/000000000/plugins/profile/2020_08_05_13_37_44/"
        "ip-172-31-19-241.trace.json.gz": ("", "0", "0"),
        "s3://smdebug-testing/resources/tf2_detailed_profile/profiler-output/framework/tensorflow/"
        "detailed_profiling/2020080523/000000002/plugins/profile/2020_08_05_23_00_25/"
        "ip-10-0-209-63.ec2.internal.trace.json.gz": (
            "151-algo-1",
            "1596668425766312",
            "1596668425896759",
        ),
        "s3://smdebug-testing/resources/missing_tf_profiler_metadata/framework/tensorflow/"
        "detailed_profiling/2020080513/000000000/plugins/profile/2020_08_05_13_37_44/"
        "8c859046be41.ant.amazon.com.trace.json.gz": ("", "0", "0"),
        "s3://smdebug-testing/empty-traces/framework/tensorflow/detailed_profiling/2020052817/"
        "ip-172-31-48-136.trace.json.gz": ("", "0", "0"),
        "s3://smdebug-testing/empty-traces/framework/tensorflow/detailed_profiling/2020052817/"
        "ip-172-31-48-136.pb": ("", "0", "0"),
        "s3://smdebug-testing/empty-traces/framework/pevents/2020052817/"
        "ip-172-31-48-136.trace.json.gz": ("", "0", "0"),
    }
    for filepath in file_path_list:
        print(f"Searching for metadata for {filepath}")
        (node_id, start, end) = read_tf_profiler_metadata_file(filepath)
        assert (node_id, start, end) == file_path_list[
            filepath
        ], f"Metadata not found for {filepath}"
