# First Party
from smdebug.profiler import TFProfilerEvents


def test_tfprofiler_events(trace_file="./tests/profiler/ip-172-31-19-241.trace.json"):
    trace_json_file = trace_file
    print(f"Reading the trace file {trace_json_file}")
    t_events = TFProfilerEvents(trace_json_file)

    all_trace_events = t_events.get_all_events()
    num_trace_events = len(all_trace_events)

    print(f"Number of events read = {num_trace_events}")
    assert num_trace_events == 1663

    event_list = t_events.get_events_at(15013686)  # nanoseconds
    print(f"Number of events at 15013686 are {len(event_list)}")
    assert len(event_list) == 21

    completed_event_list = t_events.get_events_within_range(0, 15013686)  # nanoseconds
    print(f"Number of events occurred between 0 and 15013686 are {len(completed_event_list)}")
    assert len(completed_event_list) == 1005

    start_time_sorted = t_events.get_events_start_time_sorted()
    start_time_for_first_event = start_time_sorted[0].start_time
    print(f"The first event started at {start_time_for_first_event}")
    assert start_time_for_first_event == 116457.0

    end_time_sorted = t_events.get_events_end_time_sorted()
    end_time_for_last_event = end_time_sorted[-1].end_time
    print(f"The first event started at {end_time_for_last_event}")
    assert end_time_for_last_event == 66478760.0
