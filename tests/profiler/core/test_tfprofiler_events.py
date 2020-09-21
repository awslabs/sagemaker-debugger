# First Party
# Standard Library
import os
import time

from smdebug.profiler import TensorboardProfilerEvents
from smdebug.profiler.utils import TimeUnits, read_tf_profiler_metadata_file


def test_tensorboardprofiler_events(
    trace_file="./tests/profiler/resources/tfprofiler_timeline_traces"
):
    trace_json_file = ""
    for dirpath, subdirs, files in os.walk(trace_file):
        for x in files:
            if x.endswith(".json.gz"):
                trace_json_file = os.path.join(dirpath, x)
                break
    if trace_json_file == "":
        assert False

    _, start_time_micros, end_time_micros = read_tf_profiler_metadata_file(trace_json_file)

    print(f"Reading the trace file {trace_json_file}")
    t_events = TensorboardProfilerEvents()
    t_events.read_events_from_file(trace_json_file)

    all_trace_events = t_events.get_all_events()
    num_trace_events = len(all_trace_events)

    print(f"Number of events read = {num_trace_events}")
    assert num_trace_events == 798

    completed_event_list = t_events.get_events_within_time_range(
        0, time.time(), unit=TimeUnits.SECONDS
    )
    print(f"Number of events occurred between 0 and {time.time()} are {len(completed_event_list)}")
    assert len(completed_event_list) == 798

    start_time_sorted = t_events.get_events_start_time_sorted()
    start_time_for_first_event = start_time_sorted[0].start_time
    relative_start_time = start_time_for_first_event - int(start_time_micros)
    print(f"The first event started at {relative_start_time}")
    assert relative_start_time == 21307.0

    end_time_sorted = t_events.get_events_end_time_sorted()
    end_time_for_last_event = end_time_sorted[-1].end_time
    pid_last_event = end_time_sorted[-1].pid
    tid_last_event = end_time_sorted[-1].tid
    relative_end_time = end_time_for_last_event - int(start_time_micros)
    print(f"The last event ended at {relative_end_time}")
    assert relative_end_time == 293205.0

    processes = t_events.get_processes()
    print(f"Number of processes = {len(processes)}")
    assert len(processes) == 2

    process_info = t_events.get_process_info(pid_last_event)
    print(f"Process Name = {process_info.name}  Process Id = {process_info.id}")

    thread_info = process_info.get_thread_info(tid_last_event)
    print(f"Thread name = {thread_info.thread_name} Thread id = {thread_info.tid}")
