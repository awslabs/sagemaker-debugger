# Standard Library
import argparse
import json

# First Party
from smdebug.profiler.trace_event_file_parser import (
    TIMESCALE_MULTIPLIER,
    ProcessInfo,
    TraceEventParser,
)


class SMTFProfilerEvents(TraceEventParser):
    def __init__(self, trace_file):
        self._trace_json_file = trace_file
        # SMTFProfile events are in micro seconds, we multiply by 1000 to convert to ns
        self._timescale_multiplier_for_ns = TIMESCALE_MULTIPLIER["ns"]
        super().__init__(timescale_multiplier_for_ns=TIMESCALE_MULTIPLIER["ns"])
        self.read_trace_file()

    def _populate_start_time(self, event):
        event_args = event["args"] if "args" in event else None
        if self._start_time_known is False:
            ## TODO Change parser
            if event_args and "value" in event_args:
                self._start_time_stamp = event_args["value"]
                self._start_time_known = True

    # TODO implementation of below would be changed to support streaming file and incomplete json file
    def read_trace_file(self):
        try:
            with open(self._trace_json_file) as json_data:
                trace_json_data = json.load(json_data)
        except Exception as e:
            # TODO log
            print(f"Can't open TF trace file {trace_json_file}: Exception {str(e)} ")
            return

        for event in trace_json_data:
            self._read_event(event)


class TFProfilerEvents(TraceEventParser):
    def __init__(self, trace_file):
        self._trace_json_file = trace_file
        # TFPrfoiler events in are in displayed in ns
        super().__init__(timescale_multiplier_for_ns=TIMESCALE_MULTIPLIER["ns"])
        self.read_trace_file()

    def _populate_thread_info_for_metaevent(self, event):
        if event["name"] == "thread_name":
            name = event["args"]["name"]
            t_id = event["tid"]
            pid = event["pid"]
            if pid not in self._processes:
                # log
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
            # TODO log
            print(f"Can't open TF trace file {trace_json_file}: Exception {str(e)} ")
            return
        if "traceEvents" not in trace_json_data:
            # TODO log
            return
        trace_events_json = trace_json_data["traceEvents"]

        if "displayTimeUnit" in trace_json_data:
            # TODO log, displayTimeUnit reset
            self._timescale_multiplier_for_ns = TIMESCALE_MULTIPLIER[
                trace_json_data["displayTimeUnit"]
            ]
        for event in trace_events_json:
            self._read_event(event)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_file", type=str, help="Config file for the request in JSON format")
    args = parser.parse_args()

    trace_json_file = args.trace_file
    t_events = TFProfilerEvents(trace_json_file)

    print("trace file Done")

    # Test
    event_list = t_events.get_events_at(15013686)  # nanoseconds
    print(f"{event_list}")

    completed_event_list = t_events.get_events_within_range(0, 15013686)  # nanoseconds
    print(f"{completed_event_list}")

    start_time_sorted = t_events.get_events_start_time_sorted()

    end_time_sorted = t_events.get_events_end_time_sorted()
