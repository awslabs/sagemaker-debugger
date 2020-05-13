# First Party
from smdebug.core.logger import get_logger


class ThreadInfo:
    def __init__(self, tid, thread_name):
        self.tid = tid
        self.thread_name = thread_name


class ProcessInfo:
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self._threads = dict()

    def add_thread(self, threadid, thread_name):
        self._threads[threadid] = ThreadInfo(threadid, thread_name)

    def get_thread_info(self, threadid):
        return self._threads[threadid]


class TraceEvent:
    def __init__(self, ts, name, dur, pid, tid, event_args):
        self.start_time = ts
        self.event_name = name
        self.duration = dur
        self.end_time = self.start_time + self.duration
        self.pid = pid
        self.tid = tid
        self.event_args = event_args


class TraceEventParser:
    def __init__(self):
        self._processes = dict()
        self._trace_events = list()
        self._start_timestamp = 0
        self._start_time_known = False
        # The timestamp in trace events are in micro seconds, we multiply by 1000 to convert to ns
        self._timescale_multiplier_for_ns = 1000
        self.logger = get_logger("smdebug-profiler")

    def read_trace_file(self):
        pass

    def _populate_process_info_for_metaevent(self, event):
        id = event["pid"]
        if event["name"] == "process_name":
            name = event["args"]["name"] if "name" in event["args"] else "Unknown"
            self._processes[id] = ProcessInfo(id, name)

    def _populate_thread_info_for_metaevent(self, event):
        pass

    def _populate_start_time(self, event):
        pass

    def _read_event(self, event):
        if "ph" not in event:
            self.logger.error(f"In correctly formatted trace file. The 'ph' field is not present")
            return
        phase_type = event["ph"]
        if phase_type == "M":
            self._populate_process_info_for_metaevent(event)
            self._populate_thread_info_for_metaevent(event)
            self._populate_start_time(event)

        if phase_type == "X":
            # In nano seconds
            start_time = (event["ts"] + self._start_timestamp) * self._timescale_multiplier_for_ns
            # In nano seconds
            dur = event["dur"] * self._timescale_multiplier_for_ns
            name = event["name"]
            id = event["pid"]
            tid = event["tid"] if "tid" in event else "0"
            event_args = event["args"] if "args" in event else None
            t_event = TraceEvent(start_time, name, dur, id, tid, event_args)
            self._trace_events.append(t_event)

    def get_all_events(self):
        return self._trace_events

    def get_events_start_time_sorted(self):
        return sorted(self._trace_events, key=lambda x: x.start_time)

    def get_events_end_time_sorted(self):
        return sorted(self._trace_events, key=lambda x: x.end_time)

    """
    Return the events that are in progress at the specified timestamp.
    Performance of this function can be improved by implementing interval tree.
    """

    def get_events_at(self, timestamp_in_nanoseconds):
        result_events = list()
        for x_event in self._trace_events:
            if x_event.start_time <= timestamp_in_nanoseconds <= x_event.end_time:
                result_events.append(x_event)
        return result_events

    """
    Return the events that have started and completed within the given start and end time boundaries.
    The events that are in progress during these boundaries are not included.
    """

    def get_events_within_range(self, start_time, end_time):
        result_events = list()
        for x_event in self._trace_events:
            if start_time <= x_event.start_time and end_time >= x_event.end_time:
                result_events.append(x_event)
        return result_events

    def get_process_info(self, process_id):
        return self._processes[process_id]

    def get_processes(self):
        return self._processes

    # TODO
    def get_events_for_process(self, pid, start_time, end_time):
        pass

    # TODO
    def get_events_for_thread(self, tid, start_time, end_time):
        pass
