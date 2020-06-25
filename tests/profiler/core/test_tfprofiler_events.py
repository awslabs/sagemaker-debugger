# First Party
from smdebug.profiler import TensorboardProfilerEvents
from smdebug.profiler.utils import TimeUnits


def test_tensorboardprofiler_events(
    trace_file="./tests/profiler/resources/ip-172-31-19-241.trace.json"
):
    trace_json_file = trace_file
    print(f"Reading the trace file {trace_json_file}")
    t_events = TensorboardProfilerEvents()
    t_events.read_events_from_file(trace_json_file)

    all_trace_events = t_events.get_all_events()
    num_trace_events = len(all_trace_events)

    print(f"Number of events read = {num_trace_events}")
    assert num_trace_events == 256

    completed_event_list = t_events.get_events_within_time_range(
        0, 0.015013686, unit=TimeUnits.SECONDS
    )
    print(f"Number of events occurred between 0 and 15013686 are {len(completed_event_list)}")
    assert len(completed_event_list) == 256

    start_time_sorted = t_events.get_events_start_time_sorted()
    start_time_for_first_event = start_time_sorted[0].start_time
    print(f"The first event started at {start_time_for_first_event}")
    assert start_time_for_first_event == 116457.0

    end_time_sorted = t_events.get_events_end_time_sorted()
    end_time_for_last_event = end_time_sorted[-1].end_time
    pid_last_event = end_time_sorted[-1].pid
    tid_last_event = end_time_sorted[-1].tid
    print(f"The first event started at {end_time_for_last_event}")
    assert end_time_for_last_event == 64045679.0

    processes = t_events.get_processes()
    print(f"Number of processes = {len(processes)}")
    assert len(processes) == 9

    process_info = t_events.get_process_info(pid_last_event)
    print(f"Process Name = {process_info.name}  Process Id = {process_info.id}")

    thread_info = process_info.get_thread_info(tid_last_event)
    print(f"Thread name = {thread_info.thread_name} Thread id = {thread_info.tid}")
