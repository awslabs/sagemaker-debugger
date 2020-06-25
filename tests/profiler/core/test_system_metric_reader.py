# First Party
# Third Party
import pytest

from smdebug.profiler.system_metrics_reader import LocalSystemMetricsReader, S3SystemMetricsReader
from smdebug.profiler.utils import TimeUnits


@pytest.mark.parametrize("use_in_memory_cache", [True, False])
def test_SystemLocalMetricsReader(
    use_in_memory_cache, metricfolder="./tests/profiler/resources/test_traces"
):
    lt = LocalSystemMetricsReader(metricfolder, use_in_memory_cache=use_in_memory_cache)
    events = lt.get_events(1591100000, 1692300000, unit=TimeUnits.SECONDS)

    print(f"Number of events {len(events)}")
    assert len(events) == 78

    events = lt.get_events(1591100000, 1692300000, TimeUnits.SECONDS, "cpu")

    print(f"Number of cpu events {len(events)}")
    assert len(events) == 68

    events = lt.get_events(1591100000, 1591167699, TimeUnits.SECONDS)

    print(f"Number of events {len(events)}")
    assert len(events) == 10

    events = lt.get_events(1591748165, 1600000000, TimeUnits.SECONDS)

    print(f"Number of events {len(events)}")
    assert len(events) == 32


@pytest.mark.parametrize("use_in_memory_cache", [True, False])
def test_SystemS3MetricsReader(use_in_memory_cache):
    bucket_name = "s3://smdebug-testing/resources/trainingjob_name/profiler-output"
    tt = S3SystemMetricsReader(bucket_name, use_in_memory_cache=use_in_memory_cache)
    events = tt.get_events(1591100000, 1692300000, unit=TimeUnits.SECONDS)

    print(f"Number of events {len(events)}")
    assert len(events) == 14

    events = tt.get_events(1591100000, 1692300000, TimeUnits.SECONDS, "cpu")

    print(f"Number of cpu events {len(events)}")
    assert len(events) == 4

    events = tt.get_events(1591100000, 1591167699, TimeUnits.SECONDS)

    print(f"Number of events {len(events)}")
    assert len(events) == 10


@pytest.mark.parametrize("use_in_memory_cache", [True, False])
def test_SystemS3MetricsReader_2(use_in_memory_cache):
    bucket_name = "s3://smdebug-testing/resources/trainingjob_name2/profiler-output"
    tt = S3SystemMetricsReader(bucket_name, use_in_memory_cache=use_in_memory_cache)
    events = tt.get_events(1591748000, 1591749000, unit=TimeUnits.SECONDS)

    print(f"Number of events {len(events)}")
    assert len(events) == 96

    print(f"Start after prefix is {tt._startAfter_prefix}")
    assert (
        tt._startAfter_prefix
        == "resources/trainingjob_name2/profiler-output/system/incremental/1591748100.algo-1.json"
    )

    events = tt.get_events(1591748160, 1591749000, TimeUnits.SECONDS)

    print(f"Number of events {len(events)}")
    assert len(events) == 64

    events = tt.get_events(1591748170, 1591749000, TimeUnits.SECONDS)

    print(f"Number of events {len(events)}")
    assert len(events) == 32

    events = tt.get_events(1591748165, 1591749000, TimeUnits.SECONDS)

    print(f"Number of events {len(events)}")
    assert len(events) == 64


@pytest.mark.parametrize("use_in_memory_cache", [True, False])
def test_SystemS3MetricsReader_updateStartAfterPrefix(use_in_memory_cache):
    bucket_name = "s3://smdebug-testing/resources/trainingjob_name/profiler-output"
    tt = S3SystemMetricsReader(bucket_name, use_in_memory_cache=use_in_memory_cache)
    tt.get_events(1591100000, 1692300000, unit=TimeUnits.SECONDS)
    print(f"Start after prefix is {tt._startAfter_prefix}")
    assert (
        tt._startAfter_prefix
        == "resources/trainingjob_name/profiler-output/system/incremental/1591160699.algo-1.json"
    )
