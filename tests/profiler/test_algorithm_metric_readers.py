# First Party
# Third Party
import pytest

from smdebug.profiler.algorithm_metrics_reader import (
    LocalAlgorithmMetricsReader,
    S3AlgorithmMetricsReader,
)
from smdebug.profiler.utils import TimeUnits


@pytest.mark.parametrize("use_in_memory_cache", [True, False])
def test_S3MetricsReader(use_in_memory_cache):
    bucket_name = "s3://smdebug-testing/resources/model_timeline_traces"
    tt = S3AlgorithmMetricsReader(bucket_name, use_in_memory_cache=use_in_memory_cache)
    events = tt.get_events(1590461127873222, 1590461139949971)
    print(f"Number of events {len(events)}")


@pytest.mark.parametrize("use_in_memory_cache", [True, False])
def test_LocalMetricsReader(use_in_memory_cache, tracefolder="./tests/profiler/test_traces"):
    lt = LocalAlgorithmMetricsReader(tracefolder, use_in_memory_cache=use_in_memory_cache)
    events = lt.get_events(1589930980, 1589930995, unit=TimeUnits.SECONDS)
    print(f"Number of events {len(events)}")
    assert len(events) == 4


@pytest.mark.parametrize("use_in_memory_cache", [True, False])
def test_LocalMetricsReader_Model_timeline(
    use_in_memory_cache, tracefolder="./tests/profiler/model_timeline_traces"
):
    lt = LocalAlgorithmMetricsReader(tracefolder, use_in_memory_cache=use_in_memory_cache)
    events = lt.get_events(1590461127873222, 1590461139949971)

    print(f"Number of events {len(events)}")
    assert len(events) == 54
    assert events[0].node_id == "0001"
