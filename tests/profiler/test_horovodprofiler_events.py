# First Party
from smdebug.profiler import HorovodProfilerEvents


def test_horovodprofiler_events(trace_file="./tests/profiler/horovod_timeline_small.json"):
    trace_json_file = trace_file
    print(f"Reading the trace file {trace_json_file}")
    t_events = HorovodProfilerEvents()
    t_events.read_events_from_file(trace_json_file)

    all_trace_events = t_events.get_all_events()
    num_trace_events = len(all_trace_events)

    print(f"Number of events read = {num_trace_events}")
    assert num_trace_events == 306

    completed_event_list = t_events.get_events_within_time_range(0, 8990000)  # microseconds
    print(f"Number of events occurred between 0 and 8990000000 are {len(completed_event_list)}")
    assert len(completed_event_list) == 113

    start_time_sorted = t_events.get_events_start_time_sorted()
    start_time_for_first_event = start_time_sorted[0].start_time
    print(f"The first event started at {start_time_for_first_event}")
    assert start_time_for_first_event == 8440608000

    end_time_sorted = t_events.get_events_end_time_sorted()
    end_time_for_last_event = end_time_sorted[-1].end_time
    print(f"The first event started at {end_time_for_last_event}")
    assert end_time_for_last_event == 9296459000
