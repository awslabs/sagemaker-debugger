# Standard Library
import json

# First Party
from smdebug.profiler.trace_event_file_parser import ProcessInfo, TraceEvent, TraceEventParser


class SMTFProfilerEvents(TraceEventParser):
    def __init__(self, trace_file):
        self._trace_json_file = trace_file
        super().__init__()
        self.read_trace_file()

    def _populate_start_time(self, event):
        event_args = event["args"] if "args" in event else None
        if self._start_time_known is False:
            if event_args is None:
                return
            if "start_time_since_epoch_in_micros" in event_args:
                self._start_timestamp = event_args["start_time_since_epoch_in_micros"]
                self._start_time_known = True
                self.logger.info(f"Start time for events in uSeconds = {self._start_timestamp}")

    # TODO implementation of below would be changed to support streaming file and incomplete json file
    def read_trace_file(self):
        try:
            with open(self._trace_json_file) as json_data:
                trace_json_data = json.load(json_data)
        except Exception as e:
            self.logger.error(
                f"Can't open TF trace file {self._trace_json_file}: Exception {str(e)}"
            )
            return

        for event in trace_json_data:
            self._read_event(event)


class TFProfilerEvents(TraceEventParser):
    def __init__(self, trace_file):
        self._trace_json_file = trace_file
        super().__init__()
        self.read_trace_file()

    def _populate_thread_info_for_metaevent(self, event):
        if event["name"] == "thread_name":
            name = event["args"]["name"]
            t_id = event["tid"]
            pid = event["pid"]
            if pid not in self._processes:
                self.logger.warn(
                    f"Did not find matching process for pid {pid}. Creating a process with name 'Unknown'"
                )
                self._processes[pid] = ProcessInfo(pid, "Unknown")
            self._processes[pid].add_thread(t_id, name)

    def _populate_start_time(self, event):
        # TODO, not sure if we can implement this right now
        return

    def read_trace_file(self):
        try:
            with open(self._trace_json_file) as json_data:
                trace_json_data = json.load(json_data)
        except Exception as e:
            self.logger.error(
                f"Can't open TF trace file {self._trace_json_file}: Exception {str(e)} "
            )
            return
        if "traceEvents" not in trace_json_data:
            self.logger.error(
                f"The TF trace file {self._trace_json_file} does not contain traceEvents"
            )
            return
        trace_events_json = trace_json_data["traceEvents"]

        for event in trace_events_json:
            self._read_event(event)


class HorovodProfilerEvents(TraceEventParser):
    def __init__(self, trace_file):
        self._trace_json_file = trace_file
        self._pid_stacks = dict()
        """
        In horovod trace, the 'ts' timestamps for events are relative to the first 'ts' timestamp included in the
        first event. We will consider this timestamp as base_timestamp and subtract it from the 'ts' values read
        from the subsequent events. It will give us 0-based timestamps for rest of the events. Please note that
        this base_timestamp is not related to unix epoch based timestamp. We would still have to add absolute start time
        (self._start_timestamp) to obtain the absolute start time of any event.
        """
        self._base_timestamp = 0
        self._base_timestamp_initialized = False
        super().__init__()
        self.read_trace_file()

    def _populate_thread_info_for_metaevent(self, event):
        if event["name"] == "thread_name":
            name = event["args"]["name"]
            t_id = event["tid"]
            pid = event["pid"]
            if pid not in self._processes:
                self.logger.warn(
                    f"Did not find matching process for pid {pid}. Creating a process with name 'Unknown'"
                )
                self._processes[pid] = ProcessInfo(pid, "Unknown")
            self._processes[pid].add_thread(t_id, name)

    def _populate_start_time(self, event):
        # TODO, populate the self._start_timestamp when we make changes to horovod to record the unix epoch based
        #  timestamp at the start of tracing.
        return

    def read_trace_file(self):
        try:
            with open(self._trace_json_file) as json_data:
                trace_json_data = json.load(json_data)
        except Exception as e:
            self.logger.error(
                f"Can't open Horovod trace file {self._trace_json_file}: Exception {str(e)}"
            )
            return

        for event in trace_json_data:
            self._read_event(event)

    def _read_event(self, event):
        if "ph" not in event:
            self.logger.error(f"In correctly formatted trace file. The 'ph' field is not present")
            return
        phase_type = event["ph"]
        if "ts" in event and not self._base_timestamp_initialized:
            self._base_timestamp = event["ts"]
            self.logger.info(
                f"The base timestamp in horovod trace file for future events is {self._base_timestamp}"
            )
            self._base_timestamp_initialized = True
        if phase_type == "M":
            self._populate_process_info_for_metaevent(event)
            self._populate_thread_info_for_metaevent(event)
            self._populate_start_time(event)
        if phase_type == "X":
            # In nano seconds
            start_time = (
                event["ts"] - self._base_timestamp + self._start_timestamp
            ) * self._timescale_multiplier_for_ns
            # In nano seconds
            dur = event["dur"] * self._timescale_multiplier_for_ns
            name = event["name"]
            id = event["pid"]
            tid = event["tid"] if "tid" in event else "0"
            event_args = event["args"] if "args" in event else None
            t_event = TraceEvent(start_time, name, dur, id, tid, event_args)
            self._trace_events.append(t_event)
        if phase_type == "B":
            pid = event["pid"]
            if pid not in self._pid_stacks:
                self._pid_stacks[pid] = []
            self._pid_stacks[pid].append(event)
        if phase_type == "E":
            pid = event["pid"]
            if pid not in self._pid_stacks:
                self.logger.error(f"Did not find the 'B' type event in the pid {pid}")
            else:
                b_event = self._pid_stacks[pid][-1]
                self._pid_stacks[pid].pop()
                start_time = (
                    b_event["ts"] - self._base_timestamp + self._start_timestamp
                ) * self._timescale_multiplier_for_ns
                end_time = (
                    event["ts"] - self._base_timestamp + self._start_timestamp
                ) * self._timescale_multiplier_for_ns
                duration = end_time - start_time
                if duration < 0:
                    self.logger.error(
                        f"Error in reading the events 'B' and 'E' or trace file is corrupt: pid = "
                        f"{pid}, start_time = {b_event['ts']} end_time = {event['ts']} name = "
                        f"{b_event['name']}"
                    )
                    return
                tid = b_event["tid"] if "tid" in event else "0"
                name = b_event["name"]
                event_args = event["args"] if "args" in event else None
                t_event = TraceEvent(start_time, name, duration, pid, tid, event_args)
                self._trace_events.append(t_event)
