# Standard Library
import json
import time

# Third Party
import psutil
import pytest

# First Party
from smdebug.profiler import HorovodProfilerEvents
from smdebug.profiler.hvd_trace_file_rotation import HvdTraceFileRotation
from smdebug.profiler.profiler_constants import CONVERT_TO_MICROSECS


def test_horovodprofiler_events(
    trace_file="./tests/profiler/resources/horovod_timeline_traces/framework/pevents/2020070206/1593673051473228_88359-8c859046be41.ant.amazon.com_horovod_timeline.json"
):
    trace_json_file = trace_file
    print(f"Reading the trace file {trace_json_file}")
    t_events = HorovodProfilerEvents()
    t_events.read_events_from_file(trace_json_file)

    all_trace_events = t_events.get_all_events()
    num_trace_events = len(all_trace_events)

    print(f"Number of events read = {num_trace_events}")
    assert num_trace_events == 19

    completed_event_list = t_events.get_events_within_time_range(
        1593673051473000, 1593673051473228
    )  # microseconds
    print(
        f"Number of events occurred between 1593673051473000 and 1593673051473228 are {len(completed_event_list)}"
    )
    assert len(completed_event_list) == 12

    start_time_sorted = t_events.get_events_start_time_sorted()
    start_time_for_first_event = start_time_sorted[0].start_time
    print(f"The first event started at {start_time_for_first_event}")
    assert start_time_for_first_event == 1592860696000713

    end_time_sorted = t_events.get_events_end_time_sorted()
    end_time_for_last_event = end_time_sorted[-1].end_time
    print(f"The last event ended at {end_time_for_last_event}")
    assert end_time_for_last_event == 1593673051473228


def test_steady_clock_to_epoch_time_conversion(
    simple_profiler_config_parser,
    monkeypatch,
    trace_file="./tests/profiler/resources/horovod_timeline_small.json",
):
    """
    This test checks if steady clock/ monotonic time to time since epoch conversion
    works as expected. This is being done, by converting timestamps from Horovod
    event files (that record timestamps according to monotonic clock) to epoch time
    and checking if it is within +/- 1 year/month/day of system boot time.
    If the conversion was incorrect, the time/year would be much earlier/later than current year.
    """
    assert simple_profiler_config_parser.profiling_enabled

    monkeypatch.setenv("HOROVOD_TIMELINE", trace_file)

    hvd_file_reader = HvdTraceFileRotation(simple_profiler_config_parser)

    assert hvd_file_reader.enabled

    boot_time = time.gmtime(psutil.boot_time())

    with open(trace_file) as hvd_file:
        events_dict = json.load(hvd_file)
        for e in events_dict:
            if "ts" in e:
                timestamp_in_us = hvd_file_reader._convert_monotonic_to_epoch_time(e["ts"])
                gmtime = time.gmtime(timestamp_in_us / CONVERT_TO_MICROSECS)

                # assuming that the test is being run the same day as system boot or
                # at the maximum one day after system boot.
                assert pytest.approx(boot_time.tm_year, 1) == gmtime.tm_year
                assert pytest.approx(boot_time.tm_mon, 1) == gmtime.tm_mon
                assert pytest.approx(boot_time.tm_mday, 1) == gmtime.tm_mday

    hvd_file_reader.close()
