# Standard Library
import json

# First Party
from smdebug.profiler.trace_event_file_parser import ProcessInfo, TraceEventParser


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
