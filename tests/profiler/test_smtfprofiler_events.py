# First Party
from smdebug.profiler import SMTFProfilerEvents


def test_smtfprofiler_events(trace_file="./tests/profiler/smtf_profiler_trace.json"):
    trace_json_file = trace_file
    print(f"Reading the trace file {trace_json_file}")
    t_events = SMTFProfilerEvents(trace_json_file)

    all_trace_events = t_events.get_all_events()
    num_trace_events = len(all_trace_events)

    print(f"Number of events read = {num_trace_events}")
    assert num_trace_events == 49

    event_list = t_events.get_events_at(1589314018458800000)  # nanoseconds
    print(f"Number of events at 15013686 are {len(event_list)}")
    assert len(event_list) == 1

    completed_event_list = t_events.get_events_within_range(0, 1589314018470000000)  # nanoseconds
    print(f"Number of events occurred between 0 and 15013686 are {len(completed_event_list)}")
    assert len(completed_event_list) == 34

    start_time_sorted = t_events.get_events_start_time_sorted()
    start_time_for_first_event = start_time_sorted[0].start_time
    print(f"The first event started at {start_time_for_first_event}")
    assert start_time_for_first_event == 1589314018458743000

    end_time_sorted = t_events.get_events_end_time_sorted()
    end_time_for_last_event = end_time_sorted[-1].end_time
    print(f"The first event started at {end_time_for_last_event}")
    assert end_time_for_last_event == 1589314018481947000
