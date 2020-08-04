# First Party
# Standard Library
import json
from builtins import Exception, dict, hash, sorted, str
from datetime import datetime

from smdebug.core.logger import get_logger
from smdebug.profiler.utils import (
    TimeUnits,
    convert_utc_datetime_to_nanoseconds,
    convert_utc_timestamp_to_nanoseconds,
    get_node_id_from_tracefilename,
)


class ThreadInfo:
    def __init__(self, tid, thread_name):
        self.tid = tid
        self.thread_name = thread_name


"""
Thid contains infomation about all the phases and list of
threads found for all these phase
This needs to have node-id
"""


class ProcessInfo:
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self._threads = dict()

    def add_thread(self, threadid, thread_name):
        self._threads[threadid] = ThreadInfo(threadid, thread_name)

    def get_thread_info(self, threadid):
        return self._threads[threadid]


# TODO merge all event files
# read all events of one file
# create metadata map ,
# read events of other file, update metadata map , update all events with new pids from new metadata map
# dump metadata json
# standardize time and update start and endtime ts
# dump all event metadata
# dump all events

# step timeline distributed
# for every phase, Threadids are under that pid

"""
Metadata pid
Step , step-node-i+pid+tid ………
Forward , forward-node-i + pid+tid
Backward
DataLoading

create unique pid for each phase(),
create unique threadname and thread id for each thread ( thread_name = (node_id+actual_pid+actual_thread_id+actual_thread_name))
"""

# each trace event parser has list of events
# has list of processes info , each process info has list of threads

# while merging
# new PRocess info =
#     go through each of exisiting process info
#     for each process name, create unique threadid_prefix (node_id + actual_pid)


# Device, name = (node - id, pid, thread_id)


class TraceEvent:
    def __init__(
        self, ts, name, dur, phase_pid, phase_tid, event_args, node_id, event_phase="", pid=0, tid=0
    ):
        self.start_time = ts
        self.event_name = name
        self.duration = dur
        self.end_time = self.start_time + self.duration
        # this pid points to a unique phase name
        self.phase_pid = phase_pid
        self.phase_tid = phase_tid
        self.pid = pid
        # this tid points to a unique tid under phase
        self.tid = tid
        self.event_args = event_args
        self.node_id = node_id
        self.event_phase = event_phase


class TraceEventParser:
    def __init__(self):
        # list of ProcessInfo found in this file
        self._processes = dict()
        # reverse mapping from name to id
        self._process_name_to_id = dict()
        self._trace_events = []
        """
        The _pid_stacks maintain the directory of stacks indexed using pid. The stack contains 'B' type events.
        The stack will be popped as we process the 'E' events for the same pid.
        """
        self._pid_stacks = dict()
        self._start_timestamp = 0
        self._start_time_known = False
        # The timestamp in trace events are in micro seconds, we multiply by 1000 to convert to ns
        self._timescale_multiplier_for_ns = 1000
        self.logger = get_logger("smdebug-profiler")

    def read_trace_file(self):
        pass

    def type(self):
        pass

    """
    if metadata event has process name,
    we need to handle multiple nodes with same process name
    _processes is mapping from pid to name for rest of the file
    For multiple nodes, pid will be different/same for each node and name will also be same for each ndoe

    """

    def _populate_process_info_for_metaevent(self, event):
        id = event["pid"]
        if event["name"] == "process_name":
            process_name = event["args"]["name"] if "name" in event["args"] else "Unknown"
            # we will check if this name has already been seen, possibly for previous node
            # if not seen, we will create an entry from id to name and reverse entry from name to id
            # otherwise, we will point the id to existing id.
            if process_name not in self._process_name_to_id:
                self._processes[id] = ProcessInfo(id, process_name)
                self._process_name_to_id[process_name] = id
            else:
                existing_id = self._process_name_to_id[process_name]
                self._processes[id] = ProcessInfo(existing_id, process_name)

    def _populate_thread_info_for_metaevent(self, event, node_id="", phase_tid_default=None):
        if event["name"] == "thread_name":
            name = event["args"]["name"] + "_node:" + node_id
            t_id = event["tid"]

        elif phase_tid_default is not None:
            # there is no thread mentioned and this is unique thread for pahse and node
            # We will be generating a unique tid here and return this tid to be populated in event
            name = str(phase_tid_default) + "_node:" + node_id
            t_id = hash(name)
        else:
            self.logger.info(
                f"Event:{event} doesn't have thread_name nor phase_tid_default. Returning"
            )
            return
        pid = event["pid"]
        if pid not in self._processes:
            self.logger.warn(
                f"Did not find matching process for pid {pid}. Creating a process with name 'Unknown'"
            )
            self._processes[pid] = ProcessInfo(pid, "Unknown")
        self._processes[pid].add_thread(t_id, name)
        return t_id

    def _populate_start_time(self, event):
        event_args = event["args"] if "args" in event else None
        if self._start_time_known is False:
            if event_args is None:
                return
            if "start_time_since_epoch_in_micros" in event_args:
                self._start_timestamp = event_args["start_time_since_epoch_in_micros"]
                self._start_time_known = True
                self.logger.info(f"Start time for events in uSeconds = {self._start_timestamp}")

    def _read_event(self, event, node_id=""):
        if "ph" not in event:
            return
        phase_type = event["ph"]
        if phase_type == "M":
            self._populate_process_info_for_metaevent(event)
            self._populate_thread_info_for_metaevent(event, node_id)
            self._populate_start_time(event)
        if phase_type == "X":
            # In nano seconds
            start_time = (event["ts"] + self._start_timestamp) * self._timescale_multiplier_for_ns
            # In nano seconds
            dur = event["dur"] * self._timescale_multiplier_for_ns
            name = event["name"]
            pid = phase_pid = event["pid"]  # this is phase pid
            if "tid" in event:
                phase_tid = event["tid"]
            else:
                # we will generate unique tid which is hash of 0 + node_id
                phase_tid = self._populate_thread_info_for_metaevent(
                    event, node_id=node_id, phase_tid_default=0
                )

            event_args = event["args"] if "args" in event else None
            tid = phase_tid
            if event_args:
                # Tf Detailed metrics emits pid and thread_id
                if "pid" in event_args:
                    pid = event_args["pid"]
                if "thread_id" in event_args:
                    tid = event_args["thread_id"]

            # TODO get actual pid of process
            # TODO get actual thread id of processes. depending on file type actual pid and tid may be into args
            phase_name = "Unknown"
            if phase_pid in self._processes:
                phase_name = self._processes[phase_pid].name

            t_event = TraceEvent(
                start_time,
                name,
                dur,
                phase_pid,
                phase_tid,
                event_args,
                node_id,
                phase_name,
                pid,
                tid,
            )
            self._trace_events.append(t_event)
        # TOD ignoring B and E events for now.
        # need to handle it
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
                    b_event["ts"] + self._start_timestamp
                ) * self._timescale_multiplier_for_ns
                end_time = (event["ts"] + self._start_timestamp) * self._timescale_multiplier_for_ns
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
                t_event = TraceEvent(start_time, name, duration, pid, tid, event_args, node_id)
                self._trace_events.append(t_event)

    def get_all_events(self):
        return self._trace_events

    def get_events_start_time_sorted(self):
        return sorted(self._trace_events, key=lambda x: x.start_time)

    def get_events_end_time_sorted(self):
        return sorted(self._trace_events, key=lambda x: x.end_time)

    """
    Return the events that
    1. are active or running at the start or end timestamps
    2. started and completed within the given range.
    In other words, the function will not return the events that have completed before the 'start' timestamp and
    started after the 'end' timestamp.
    The start and end time, by default, are assumed to be in microseconds.
    For tracefiles generated by smdebug, the start and end timestamps need to be seconds elapsed
    from epoch ( January 1, 1970 12:00:00 AM)
    For horovod and tensorboard generated tracefiles, the start and end timestamps will be interpreted as
    seconds elapsed from the first recorded event.
    """

    def get_events_within_time_range(self, start_time, end_time, unit=TimeUnits.MICROSECONDS):
        start_time_nanoseconds = convert_utc_timestamp_to_nanoseconds(start_time, unit)
        end_time_nanoseconds = convert_utc_timestamp_to_nanoseconds(end_time, unit)
        result_events = list()
        for x_event in self._trace_events:
            # event finished before start time or event started after the end time
            if (
                x_event.end_time < start_time_nanoseconds
                or end_time_nanoseconds < x_event.start_time
            ):
                continue
            result_events.append(x_event)

        return result_events

    """
    The TraceEvent class can not support retrieving events based on given datetime objects.
    This is because only smdebug based tracefile store the timestamps based on unix epoch timestamp.
    """

    def get_events_within_range(self, start_time: datetime, end_time: datetime):
        """
        Return the events that have started and completed within the given start and end time boundaries.
        The start and end time can be specified datetime objects.
        The events that are in progress during these boundaries are not included.
        """
        start_time_nanoseconds = end_time_nanoseconds = 0
        if start_time.__class__ is datetime:
            start_time_nanoseconds = convert_utc_datetime_to_nanoseconds(start_time)
        if end_time.__class__ is datetime:
            end_time_nanoseconds = convert_utc_datetime_to_nanoseconds(end_time)
        return self.get_events_within_time_range(
            start_time_nanoseconds, end_time_nanoseconds, unit=TimeUnits.NANOSECONDS
        )

    def get_process_info(self, process_id):
        return self._processes[process_id]

    def get_processes(self):
        return self._processes

    # TODO implementation of below would be changed to support streaming file and incomplete json file
    def read_events_from_file(self, tracefile):
        try:
            with open(tracefile) as json_data:
                trace_json_data = json.load(json_data)
        except Exception as e:
            self.logger.error(f"Can't open trace file {tracefile}: Exception {str(e)}")
            return
        node_id = get_node_id_from_tracefilename(tracefile)
        self.read_events_from_json_data(trace_json_data, node_id)

    def read_events_from_json_data(self, trace_json_data, node_id):
        for event in trace_json_data:
            self._read_event(event, node_id)

    # TODO
    def get_events_for_process(self, pid, start_time, end_time):
        pass

    # TODO
    def get_events_for_thread(self, tid, start_time, end_time):
        pass

    def clear_events(self):
        self._trace_events = []

    # TODO this will merge
    # def merge(otherTraceEventParser):
    #    if not isinstance(otherTraceEventParser, TraceEventParser):
    #        raise Exception(f"otherTraceEventParser is of type: {instance(otherTraceEventParser)}. Expected type:TraceEventParser"

    # def dumpEventsToTimelineJson():
    #    #TODO implement
