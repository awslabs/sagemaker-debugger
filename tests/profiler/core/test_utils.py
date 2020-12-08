# Third Party
# Standard Library
import json
import time
import uuid
from pathlib import Path

import pytest

# First Party
from smdebug.core.access_layer.s3handler import ListRequest, S3Handler
from smdebug.profiler.algorithm_metrics_reader import (
    LocalAlgorithmMetricsReader,
    S3AlgorithmMetricsReader,
)
from smdebug.profiler.analysis.utils.merge_timelines import MergedTimeline, MergeUnit
from smdebug.profiler.profiler_constants import (
    CONVERT_TO_MICROSECS,
    HOROVODTIMELINE_SUFFIX,
    MERGEDTIMELINE_SUFFIX,
    MODELTIMELINE_SUFFIX,
    PYTHONTIMELINE_SUFFIX,
    TENSORBOARDTIMELINE_SUFFIX,
)
from smdebug.profiler.tf_profiler_parser import SMProfilerEvents
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


@pytest.mark.parametrize("trace_location", ["local", "s3"])
def test_timeline_merge(out_dir, trace_location):
    if trace_location == "local":
        tracefolder = "./tests/profiler/resources/merge_traces"
        combined_timeline = MergedTimeline(tracefolder, output_directory=out_dir)
        start_time_us = 1594662618874598
        end_time_us = 1594662623418701
        combined_timeline.merge_timeline(start_time_us, end_time_us)
        algo_reader = LocalAlgorithmMetricsReader(tracefolder)
    elif trace_location == "s3":
        tracefolder = "s3://smdebug-testing/resources/tf2_detailed_profile/profiler-output"
        combined_timeline = MergedTimeline(tracefolder, output_directory=out_dir)
        start_time_us = 1596668400000000
        end_time_us = 1596668441010095
        combined_timeline.merge_timeline(start_time_us, end_time_us)
        algo_reader = S3AlgorithmMetricsReader(tracefolder)
    else:
        return

    files = []
    for path in Path(out_dir).rglob(f"*{MERGEDTIMELINE_SUFFIX}"):
        files.append(path)

    assert len(files) == 1

    with open(files[0], "r+") as merged_file:
        event_list = json.load(merged_file)

    # check if the events are sorted by start time
    for i in range(len(event_list) - 1):
        if "ts" in event_list[i] and "ts" in event_list[i + 1]:
            assert event_list[i]["ts"] <= event_list[i + 1]["ts"]

    total_events = 0
    # check if the number of events match individual files
    for suffix in [
        PYTHONTIMELINE_SUFFIX,
        MODELTIMELINE_SUFFIX,
        TENSORBOARDTIMELINE_SUFFIX,
        HOROVODTIMELINE_SUFFIX,
    ]:
        events = algo_reader.get_events(start_time_us, end_time_us, file_suffix_filter=[suffix])
        total_events += len(events)

    t_events = SMProfilerEvents()
    t_events.read_events_from_file(str(files[0]))
    all_trace_events = t_events.get_all_events()
    num_trace_events = len(all_trace_events)

    assert total_events == num_trace_events


@pytest.mark.parametrize("trace_location", ["local", "s3"])
def test_timeline_merge_by_step(out_dir, trace_location):
    start_step, end_step = 2, 4
    if trace_location == "local":
        tracefolder = "./tests/profiler/resources/merge_traces"
        combined_timeline = MergedTimeline(tracefolder, output_directory=out_dir)
        combined_timeline.merge_timeline(start_step, end_step, unit=MergeUnit.STEP)
        algo_reader = LocalAlgorithmMetricsReader(tracefolder)
    elif trace_location == "s3":
        tracefolder = "s3://smdebug-testing/resources/tf2_detailed_profile/profiler-output"
        combined_timeline = MergedTimeline(tracefolder, output_directory=out_dir)
        combined_timeline.merge_timeline(start_step, end_step, unit=MergeUnit.STEP)
        algo_reader = S3AlgorithmMetricsReader(tracefolder)
    else:
        return

    files = []
    for path in Path(out_dir).rglob(f"*{MERGEDTIMELINE_SUFFIX}"):
        files.append(path)

    assert len(files) == 1

    total_events = 0
    # check if the number of events match individual files
    for suffix in [
        PYTHONTIMELINE_SUFFIX,
        MODELTIMELINE_SUFFIX,
        TENSORBOARDTIMELINE_SUFFIX,
        HOROVODTIMELINE_SUFFIX,
    ]:
        events = algo_reader.get_events_by_step(start_step, end_step, file_suffix_filter=[suffix])
        total_events += len(events)

    t_events = SMProfilerEvents()
    t_events.read_events_from_file(str(files[0]))
    all_trace_events = t_events.get_all_events()
    num_trace_events = len(all_trace_events)

    assert total_events == num_trace_events


def test_merge_timeline_s3_write():
    bucket_name = "smdebug-testing"
    key_name = f"outputs/smprofiler-timeline-merge-test-{uuid.uuid4()}"
    location = "s3://{}/{}".format(bucket_name, key_name)

    tracefolder = "./tests/profiler/resources/merge_traces"
    combined_timeline = MergedTimeline(tracefolder, output_directory=location)
    combined_timeline.merge_timeline(0, time.time() * CONVERT_TO_MICROSECS, unit=MergeUnit.TIME)

    start_step, end_step = 2, 4
    tracefolder = "s3://smdebug-testing/resources/tf2_detailed_profile/profiler-output"
    combined_timeline = MergedTimeline(tracefolder, output_directory=location)
    combined_timeline.merge_timeline(start_step, end_step, unit=MergeUnit.STEP)

    request = ListRequest(bucket_name, key_name)
    files = S3Handler.list_prefixes([request])
    assert len(files) == 1
    assert len(files[0]) == 2


def test_timeline_merge_file_suffix_filter(out_dir):
    bucket_name = "s3://smdebug-testing/profiler-traces"
    combined_timeline = MergedTimeline(
        path=bucket_name, file_suffix_filter=[MODELTIMELINE_SUFFIX], output_directory=out_dir
    )
    combined_timeline.merge_timeline(0, time.time() * CONVERT_TO_MICROSECS)

    files = []
    for path in Path(out_dir).rglob(f"*{MERGEDTIMELINE_SUFFIX}"):
        files.append(path)

    assert len(files) == 1

    with open(files[0], "r+") as merged_file:
        event_list = json.load(merged_file)

    for i in range(len(event_list) - 1):
        if "ts" in event_list[i] and "ts" in event_list[i + 1]:
            assert event_list[i]["ts"] <= event_list[i + 1]["ts"]


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
