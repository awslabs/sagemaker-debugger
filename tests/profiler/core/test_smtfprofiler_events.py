# First Party
# Standard Library
from datetime import datetime

from smdebug.profiler import SMProfilerEvents
from smdebug.profiler.utils import TimeUnits, get_node_id_from_tracefilename


def test_smprofiler_events(
    trace_file="./tests/profiler/resources/1589314018481947000_1234-testhost_model_timeline.json"
):
    trace_json_file = trace_file
    print(f"Reading the trace file {trace_json_file}")
    t_events = SMProfilerEvents()
    t_events.read_events_from_file(trace_json_file)

    all_trace_events = t_events.get_all_events()
    num_trace_events = len(all_trace_events)

    print(f"Number of events read = {num_trace_events}")
    assert num_trace_events == 49

    node_id_from_file = get_node_id_from_tracefilename(trace_json_file)
    node_id_from_event = all_trace_events[10].node_id
    assert node_id_from_event == node_id_from_file

    completed_event_list = t_events.get_events_within_time_range(
        0, 1589314018.4700, unit=TimeUnits.SECONDS
    )
    print(
        f"Number of events occurred between 0 and 1589314018.4700 are {len(completed_event_list)}"
    )
    assert len(completed_event_list) == 39

    start_dt = datetime.fromtimestamp(0)
    end_dt = datetime.fromtimestamp(1589314018.4700)
    completed_event_list = t_events.get_events_within_range(start_dt, end_dt)
    print(
        f"Number of events occurred between {start_dt} and {end_dt} are {len(completed_event_list)}"
    )
    assert len(completed_event_list) == 39

    start_time_sorted = t_events.get_events_start_time_sorted()
    start_time_for_first_event = start_time_sorted[0].start_time
    print(f"The first event started at {start_time_for_first_event}")
    assert start_time_for_first_event == 1589314018458743

    end_time_sorted = t_events.get_events_end_time_sorted()
    end_time_for_last_event = end_time_sorted[-1].end_time
    print(f"The first event started at {end_time_for_last_event}")
    assert end_time_for_last_event == 1589314018481947
