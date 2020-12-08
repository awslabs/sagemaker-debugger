# First Party
# Standard Library
import json

from smdebug.core.logger import get_logger
from smdebug.profiler.utils import TimeUnits, convert_utc_timestamp_to_nanoseconds


class ProfilerSystemEvent:
    def __init__(self, event_type, name, dimension, value, node_id, timestamp):
        """
        type must be one of the value in ["cpu", "gpu"]
        """
        self.type = event_type
        """
        name specifies the name of cpu/gpu core, it is optional as memory event doesn't have a type
        """
        self.name = name
        """
        dimension is a part of the event that is associated with the specified type, example: CPUUtilization
        """
        self.dimension = dimension
        self.value = value
        self.node_id = node_id
        """
        timestamp in seconds, example: 1591160699.4570894
        """
        self.timestamp = timestamp


class SystemProfilerEventParser:
    def __init__(self):
        self._events = []
        self.logger = get_logger()

    def _read_event(self, event):
        name = None
        if "Name" in event:
            name = event["Name"]
        parsed_event = ProfilerSystemEvent(
            event["Type"],
            name,
            event["Dimension"],
            event["Value"],
            event["NodeId"],
            event["Timestamp"],
        )
        self._events.append(parsed_event)

    def read_events_from_file(self, eventfile):
        try:
            with open(eventfile) as json_data:
                event_line = json_data.readline()
                while event_line:
                    json_event = json.loads(event_line)
                    event_line = json_data.readline()
                    self.read_event_from_dict(json_event)
        except Exception as e:
            self.logger.error(
                f"Can't open profiler system metric file {eventfile}: Exception {str(e)}"
            )
            raise ValueError(
                f"Can't open profiler system metric file {eventfile}: Exception {str(e)}"
            )

    def read_events_from_json_data(self, system_profiler_json_data):
        for event in system_profiler_json_data:
            self._read_event(event)

    def read_event_from_dict(self, event):
        self._read_event(event)

    def get_events_within_time_range(
        self, start_time, end_time, unit=TimeUnits.MICROSECONDS, event_type=None
    ):
        start_time_nanos = convert_utc_timestamp_to_nanoseconds(start_time, unit)
        end_time_nanos = convert_utc_timestamp_to_nanoseconds(end_time, unit)
        result_events = list(
            filter(
                lambda event: self._valid_event(
                    event, start_time_nanos, end_time_nanos, event_type
                ),
                self._events,
            )
        )
        return result_events

    @staticmethod
    def _valid_event(event, start_time_nanos, end_time_nanos, event_type):
        timestamp_in_nanos = convert_utc_timestamp_to_nanoseconds(
            event.timestamp, TimeUnits.SECONDS
        )
        if event_type is not None:
            return (
                event
                and event.type == event_type
                and start_time_nanos <= timestamp_in_nanos <= end_time_nanos
            )
        else:
            return event and start_time_nanos <= timestamp_in_nanos <= end_time_nanos

    def get_all_events(self):
        return self._events

    def clear_events(self):
        self._events = []


class ProfilerSystemEvents(SystemProfilerEventParser):
    def __init__(self):
        super().__init__()
