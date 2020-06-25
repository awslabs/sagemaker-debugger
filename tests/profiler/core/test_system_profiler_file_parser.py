# First Party
from smdebug.profiler.system_profiler_file_parser import ProfilerSystemEvents
from smdebug.profiler.utils import TimeUnits


def test_profiler_system_events(file_path="./tests/profiler/resources/1591160699.algo-1.json"):
    event_json_file = file_path
    print(f"Reading the profiler system event file {event_json_file}")
    events = ProfilerSystemEvents()
    events.read_events_from_file(event_json_file)

    all_events = events.get_all_events()
    num_events = len(all_events)

    print(f"Number of events read = {num_events}")
    assert num_events == 14

    completed_event_list = events.get_events_within_time_range(
        0, 1591170700.4570, unit=TimeUnits.SECONDS
    )
    print(
        f"Number of events occurred between 0 and 1591170700.4570 are {len(completed_event_list)}"
    )
    assert len(completed_event_list) == 14

    completed_event_list = events.get_events_within_time_range(
        0, 1591170700.4570, TimeUnits.SECONDS, "cpu"
    )
    print(
        f"Number of cpu events occurred between 0 and 1591170700.4570 are {len(completed_event_list)}"
    )
    assert len(completed_event_list) == 4

    completed_event_list = events.get_events_within_time_range(
        0, 1591167699.9572, unit=TimeUnits.SECONDS
    )
    print(
        f"Number of events occurred between 0 and 1591167699.9572 are {len(completed_event_list)}"
    )
    assert len(completed_event_list) == 11

    completed_event_list = events.get_events_within_time_range(
        1591169699.4579, 2591167699.9572, unit=TimeUnits.SECONDS
    )
    print(
        f"Number of events occurred between 1591169699.4579 and 2591167699.9572 are {len(completed_event_list)}"
    )
    assert len(completed_event_list) == 2

    completed_event_list = events.get_events_within_time_range(
        1591169699.4579, 2591167699.9572, TimeUnits.SECONDS, "gpu"
    )

    print(
        f"Number of gpu events occurred between 1591169699.4579 and 2591167699.9572 are {len(completed_event_list)}"
    )
    assert len(completed_event_list) == 2

    completed_event_list = events.get_events_within_time_range(0, 0, TimeUnits.SECONDS, "gpu")

    print(f"Number of gpu events occurred between 0 and 0 are {len(completed_event_list)}")
    assert len(completed_event_list) == 0
