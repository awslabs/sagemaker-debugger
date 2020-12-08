# First Party
# Standard Library
import time

# Third Party
import pytest

from smdebug.profiler.algorithm_metrics_reader import (
    LocalAlgorithmMetricsReader,
    S3AlgorithmMetricsReader,
)
from smdebug.profiler.profiler_constants import CONVERT_TO_MICROSECS
from smdebug.profiler.utils import TimeUnits


@pytest.mark.parametrize("use_in_memory_cache", [True, False])
def test_S3MetricsReader(use_in_memory_cache):
    bucket_name = "s3://smdebug-testing/resources/model_timeline_traces"
    tt = S3AlgorithmMetricsReader(bucket_name, use_in_memory_cache=use_in_memory_cache)
    events = tt.get_events(1590461127873222, 1590461139949971)
    print(f"Number of events {len(events)}")


@pytest.mark.parametrize("use_in_memory_cache", [True, False])
def test_LocalMetricsReader(
    use_in_memory_cache, tracefolder="./tests/profiler/resources/test_traces"
):
    lt = LocalAlgorithmMetricsReader(tracefolder, use_in_memory_cache=use_in_memory_cache)
    events = lt.get_events(1589930980, 1589930995, unit=TimeUnits.SECONDS)
    print(f"Number of events {len(events)}")
    assert len(events) == 4


@pytest.mark.parametrize("use_in_memory_cache", [True, False])
def test_LocalMetricsReader_Model_timeline(
    use_in_memory_cache, tracefolder="./tests/profiler/resources/model_timeline_traces"
):
    lt = LocalAlgorithmMetricsReader(tracefolder, use_in_memory_cache=use_in_memory_cache)
    events = lt.get_events(1590461127873222, 1590461139949971)

    print(f"Number of events {len(events)}")
    assert len(events) == 54
    assert events[0].node_id == "0001"


@pytest.mark.parametrize("use_in_memory_cache", [True, False])
def test_LocalMetricsReader_Horovod_timeline(
    use_in_memory_cache, tracefolder="./tests/profiler/resources/horovod_timeline_traces"
):
    lt = LocalAlgorithmMetricsReader(tracefolder, use_in_memory_cache=use_in_memory_cache)
    events = lt.get_events(1593673051472800, 1593673051473100)

    print(f"Number of events {len(events)}")
    assert len(events) == 15


@pytest.mark.parametrize("use_in_memory_cache", [True, False])
@pytest.mark.parametrize("trace_location", ["local", "s3"])
def test_MetricsReader_TFProfiler_timeline(use_in_memory_cache, trace_location):
    if trace_location == "local":
        tracefolder = "./tests/profiler/resources/tfprofiler_timeline_traces"
        lt = LocalAlgorithmMetricsReader(tracefolder, use_in_memory_cache=use_in_memory_cache)
    elif trace_location == "s3":
        bucket_name = "s3://smdebug-testing/resources/tf2_detailed_profile/profiler-output"
        lt = S3AlgorithmMetricsReader(bucket_name, use_in_memory_cache=use_in_memory_cache)
    else:
        return
    events = lt.get_events(0, time.time() * CONVERT_TO_MICROSECS)

    print(f"Number of events {len(events)}")
    if trace_location == "local":
        assert len(events) == 798
    elif trace_location == "s3":
        assert len(events) >= 73000


@pytest.mark.parametrize("use_in_memory_cache", [True, False])
def test_MetricReader_all_files(use_in_memory_cache):
    bucket_name = "s3://smdebug-testing/resources/pytorch_traces_with_pyinstru/profiler-output"
    lt = S3AlgorithmMetricsReader(bucket_name, use_in_memory_cache=use_in_memory_cache)

    events = lt.get_events(0, time.time() * CONVERT_TO_MICROSECS)

    assert len(events) != 0
