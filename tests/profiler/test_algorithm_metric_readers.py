# First Party
from smdebug.profiler.AlgorithmMetricsReader import (
    LocalAlgorithmMetricsReader,
    S3AlgorithmMetricsReader,
)
from smdebug.profiler.utils import TimeUnits


def test_S3MetricsReader():
    bucket_name = "s3://smdebug-testing/resources/model_timeline_traces"
    tt = S3AlgorithmMetricsReader(bucket_name)
    events = tt.get_events(1590461127873222, 1590461139949971)
    print(f"Number of events {len(events)}")


def test_LocalMetricsReader(tracefolder="./tests/profiler/test_traces"):
    lt = LocalAlgorithmMetricsReader(tracefolder)
    events = lt.get_events(1589930980, 1589930995, unit=TimeUnits.SECONDS)
    print(f"Number of events {len(events)}")
    assert len(events) == 4


def test_LocalMetricsReader_Model_timeline(tracefolder="./tests/profiler/model_timeline_traces"):
    lt = LocalAlgorithmMetricsReader(tracefolder)
    events = lt.get_events(1590461127873222, 1590461139949971)

    print(f"Number of events {len(events)}")
    assert len(events) == 54
    assert events[0].node_id == "0001"
