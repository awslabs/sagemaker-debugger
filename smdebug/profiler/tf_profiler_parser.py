# Standard Library
import json
from datetime import datetime

# First Party
from smdebug.profiler.trace_event_file_parser import TraceEventParser
from smdebug.profiler.utils import TimeUnits, convert_utc_datetime_to_nanoseconds


class SMProfilerEvents(TraceEventParser):
    def __init__(self):
        super().__init__()

    def _populate_start_time(self, event):
        event_args = event["args"] if "args" in event else None
        if self._start_time_known is False:
            if event_args is None:
                return
            if "start_time_since_epoch_in_micros" in event_args:
                self._start_timestamp = event_args["start_time_since_epoch_in_micros"]
                self._start_time_known = True
                self.logger.info(f"Start time for events in uSeconds = {self._start_timestamp}")

    """
    Return the events that are in progress at the specified timestamp.
    The timestamp can accept the datetime object.
    Performance of this function can be improved by implementing interval tree.
    """

    def get_events_at_time(self, timestamp_datetime: datetime):
        if timestamp_datetime.__class__ is datetime:
            timestamp_in_nanoseconds = convert_utc_datetime_to_nanoseconds(timestamp_datetime)
            return self.get_events_at_timestamp(
                timestamp_in_nanoseconds, unit=TimeUnits.NANOSECONDS
            )

    """
    Return the events that have started and completed within the given start and end time boundaries.
    The start and end time can be specified datetime objects.
    The events that are in progress during these boundaries are not included.
    """

    def get_events_within_range(self, start_time: datetime, end_time: datetime):
        if start_time.__class__ is datetime:
            start_time_nanoseconds = convert_utc_datetime_to_nanoseconds(start_time)
        if end_time.__class__ is datetime:
            end_time_nanoseconds = convert_utc_datetime_to_nanoseconds(end_time)
        return self.get_events_within_time_range(
            start_time_nanoseconds, end_time_nanoseconds, unit=TimeUnits.NANOSECONDS
        )


class TensorboardProfilerEvents(TraceEventParser):
    def __init__(self):
        super().__init__()

    def _populate_start_time(self, event):
        # TODO, not sure if we can implement this right now
        return

    def read_events_from_file(self, tracefile):
        try:
            with open(tracefile) as json_data:
                trace_json_data = json.load(json_data)
        except Exception as e:
            self.logger.error(f"Can't open TF trace file {tracefile}: Exception {str(e)} ")
            return
        if "traceEvents" not in trace_json_data:
            self.logger.error(f"The TF trace file {tracefile} does not contain traceEvents")
            return
        trace_events_json = trace_json_data["traceEvents"]

        for event in trace_events_json:
            self._read_event(event)


class HorovodProfilerEvents(TraceEventParser):
    def __init__(self):
        super().__init__()
        self._base_timestamp_initialized = False

    def _populate_start_time(self, event):
        # TODO, populate the self._start_timestamp when we make changes to horovod to record the unix epoch based
        #  timestamp at the start of tracing.
        return
