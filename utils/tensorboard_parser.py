# Standard Library
import argparse
import json


class threadId:
    def __init__(self, tid, thread_name):
        self.tid = tid
        self.thread_name = thread_name


class processId:
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self._threads = dict()

    def add_thread(self, threadid, thread_name):
        self._threads[threadid] = threadId(threadid, thread_name)

    def get_thread_info(self, threadid):
        return self._threads[threadid]


class XEvent:
    def __init__(self, ts, name, dur, pid, tid, event_args):
        self.start_time = ts
        self.event_name = name
        self.duration = dur
        self.end_time = self.start_time + self.duration
        self.pid = pid
        self.tid = tid
        self.event_args = event_args


class tensorboardEvents:
    def __init__(self, trace_file):
        self.trace_json_file = trace_file
        self.timescale = "ns"
        self._processes = dict()
        self._x_events = list()
        self._start_time_stamp = 0
        self._start_time_known = False

        try:
            with open(trace_json_file) as json_data:
                trace_json_data = json.load(json_data)
        except Exception as e:
            print(f"Error No metric file for {trace_json_file}: Exception {str(e)} ")
            status = False
            return

        trace_events = trace_json_data["traceEvents"]
        self.timescale = trace_json_data["displayTimeUnit"]

        for event in trace_events:
            if "ph" not in event:
                continue
            phase_type = event["ph"]
            if phase_type == "M":
                id = event["pid"]
                if event["name"] == "process_name":
                    name = event["args"]["name"]
                    self._processes[id] = processId(id, name)
                if event["name"] == "thread_name":
                    name = event["args"]["name"]
                    t_id = event["tid"]
                    self._processes[id].add_thread(t_id, name)
                event_args = event["args"] if "args" in event else None
                if self._start_time_known is False:
                    if event_args and "value" in event_args:
                        self._start_time_stamp = event_args["value"]
                        self._start_time_known = True

            if phase_type == "X":
                start_time = (event["ts"] * 1000) - self._start_time_stamp  # In nano seconds
                dur = event["dur"] * 1000  # In nano seconds
                name = event["name"]
                id = event["pid"]
                tid = event["tid"] if "tid" in event else "0"
                event_args = event["args"] if "args" in event else None
                t_event = XEvent(start_time, name, dur, id, tid, event_args)
                self._x_events.append(t_event)

    def get_all_events(self):
        return self._x_events

    def get_events_start_time_sorted(self):
        newlist = sorted(self._x_events, key=lambda x: x.start_time)
        return newlist

    def get_events_end_time_sorted(self):
        newlist = sorted(self._x_events, key=lambda x: x.end_time)
        return newlist

    """
    Return the events that are in progress at the specified timestamp.
    Performance of this function can be improved by implementing interval tree.
    """

    def get_events_at(self, timestamp_in_nanoseconds):
        result_events = list()
        for x_event in self._x_events:
            if x_event.start_time <= timestamp_in_nanoseconds <= x_event.end_time:
                result_events.append(x_event)
        return result_events

    """
    Return the events that have started and completed within the given start and end time boundaries.
    The events that are in progress during these boundaries are not included.
    """

    def get_events_within_range(self, start_time, end_time):
        result_events = list()
        for x_event in self._x_events:
            if start_time <= x_event.start_time and end_time >= x_event.end_time:
                result_events.append(x_event)
        return result_events

    def get_process_info(self, process_id):
        return self._processes[process_id]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_file", type=str, help="Config file for the request in JSON format")
    args = parser.parse_args()

    trace_json_file = args.trace_file
    t_events = tensorboardEvents(trace_json_file)

    print("trace file Done")

    # Test
    event_list = t_events.get_events_at(15013686)  # nanoseconds
    print(f"{event_list}")

    completed_event_list = t_events.get_events_within_range(0, 15013686)  # nanoseconds
    print(f"{completed_event_list}")

    start_time_sorted = t_events.get_events_start_time_sorted()

    end_time_sorted = t_events.get_events_end_time_sorted()
