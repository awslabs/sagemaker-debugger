# First Party
# Standard Library
import json
import re
from builtins import Exception, dict, hash, sorted, str
from datetime import datetime

from smdebug.core.logger import get_logger
from smdebug.profiler.utils import (
    TimeUnits,
    convert_utc_datetime_to_microseconds,
    convert_utc_datetime_to_nanoseconds,
    convert_utc_timestamp_to_microseconds,
    convert_utc_timestamp_to_nanoseconds,
    get_node_id_from_tracefilename,
)


class ThreadInfo:
    def __init__(self, tid, thread_name):
        self.tid = tid
        self.thread_name = thread_name

    def __repr__(self):
        return f"tid:{self.tid} name:{self.thread_name}"


"""
This contains infomation about all the phases and list of
threads found for all these phase.
For mutiple worker scenarios, there will be a phase and different workers will be treated as thread for the phase
"""


class ProcessInfo:
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self._threads = dict()

    def add_thread(self, threadid, thread_name):
        if threadid not in self._threads:
            self._threads[threadid] = ThreadInfo(threadid, thread_name)

    def get_thread_info(self, threadid):
        return self._threads[threadid]

    def __repr__(self):
        return f"id:{self.id} name:{self.name} threads:{self._threads}"


class TraceEvent:
    def __init__(
        self,
        ts,
        name,
        dur,
        phase_pid,
        phase_tid,
        event_args,
        node_id,
        phase,
        file_type,
        event_phase="",
        pid=0,
        tid=0,
        process_info=None,
    ):
        self.start_time = ts
        self.event_name = name
        self.duration = dur
        self.end_time = self.start_time + self.duration
        # this pid points to a unique phase name, the left most column of timeline file
        self.phase_pid = phase_pid
        # different threads and workers will be under phase, each worker will have its unique phase_tid
        self.phase_tid = phase_tid
        # Actual process id
        self.pid = pid
        # Actual thread id tid
        self.tid = tid
        self.event_args = event_args
        self.node_id = node_id
        self.phase = phase
        # the phase name this event belongs to, this can also come from self._processes[phase_pid].name
        self.event_phase = event_phase
        self.process_info = process_info
        self.file_type = file_type

    def to_json(self):
        json_dict = {
            "name": self.event_name,
            "pid": self.pid,
            "tid": self.tid,
            "ph": self.phase,
            "ts": self.start_time,
        }

        # handle Instant event
        if self.phase == "i":
            if self.event_args:
                # Instant events have a field unique to them called scope.
                # scope can be "g" - global, "p" - process, "t" - thread.
                # parsing this value that is being passed as args.
                s = self.event_args["s"] if "s" in self.event_args else "t"
                json_dict.update({"s": s})
                if "s" in self.event_args:
                    self.event_args.pop("s")
        elif self.phase == "X":
            json_dict.update({"dur": self.duration})

        if self.event_args:
            json_dict["args"] = self.event_args

        return json.dumps(json_dict)


class TraceEventParser:
    def __init__(self, type=""):
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
        self.type = type
        self.logger = get_logger()

    def read_trace_file(self):
        pass

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
                # this looks like multinode scenario, where same phase is coming from different node
                # we will tie this id to existing phase, any thread for this id would be related to this phase
                existing_id = self._process_name_to_id[process_name]
                self._processes[id] = ProcessInfo(existing_id, process_name)

    def _populate_thread_info_for_metaevent(
        self, event, node_id="", phase_tid_default=None, phase_name=""
    ):
        # The `E` event in horovod and SMDataParallel does not have a `name` entity. Add an empty `name` in an event dictionary if it is absent.
        if self.type in ["SMDataParallelMetrics", "HorovodMetrics"]:
            if "name" not in event and "ph" in event and event["ph"] == "E":
                event["name"] = ""
        if event["name"] == "thread_name":
            name = node_id + "_" + event["args"]["name"]
            t_id = event["tid"]
        elif event["name"] == "thread_sort_index":
            name = "Unknown"
            t_id = event["tid"]
        elif phase_tid_default is not None:
            # there is no thread mentioned and this is unique thread for phase and node
            # We will be generating a unique tid here and return this tid to be populated in event
            name = node_id + "_" + str(phase_tid_default)
            t_id = hash(name + str(phase_name))
        else:
            self.logger.debug(
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
        if event_args is None:
            return
        # the start time since micros is set for each trace event parser type.
        # in cases such as multiprocessing, files of the same type may have
        # different start times. updating start time here if it is different from
        # previous file that was read.
        if "start_time_since_epoch_in_micros" in event_args:
            start_time_in_us = event_args["start_time_since_epoch_in_micros"]
            if self._start_timestamp == start_time_in_us:
                return
            self._start_timestamp = start_time_in_us
            self.logger.info(f"Start time for events in uSeconds = {self._start_timestamp}")

    def _read_event(self, event, node_id=""):
        if "ph" not in event:
            return
        # if node-id in for of pid-algo-i , interchange to algo-i-pid so that we can have good sorted order
        # if we view this in timeline
        found_nodeids_parts = re.match("(.*)-(algo.*)", node_id) if node_id is not None else None
        if found_nodeids_parts is not None and len(found_nodeids_parts.groups()) == 2:
            node_id = found_nodeids_parts[2] + "-" + found_nodeids_parts[1]

        phase_type = event["ph"]
        if phase_type == "M":
            self._populate_process_info_for_metaevent(event)
            self._populate_thread_info_for_metaevent(event, node_id)
            self._populate_start_time(event)
        if phase_type == "X":
            # In nano seconds
            start_time = event["ts"] + self._start_timestamp
            # In nano seconds
            dur = event["dur"]
            name = event["name"]
            pid = phase_pid = event["pid"]  # this is phase pid
            phase_tid = event["tid"] if "tid" in event else 0

            phase_name = "Unknown"
            if phase_pid in self._processes:
                phase_name = self._processes[phase_pid].name

            if (
                "tid" in event
                and phase_pid in self._processes
                and event["tid"] in self._processes[phase_pid]._threads
            ):
                phase_tid = event["tid"]
            else:
                # we will generate unique tid which is hash of 0 + node_id
                phase_tid = self._populate_thread_info_for_metaevent(
                    event, node_id=node_id, phase_tid_default=phase_tid, phase_name=phase_name
                )

            event_args = event["args"] if "args" in event else None
            tid = phase_tid
            if event_args:
                # Tf Detailed metrics emits pid and thread_id
                if "pid" in event_args:
                    pid = event_args["pid"]
                if "thread_id" in event_args:
                    tid = event_args["thread_id"]

            t_event = TraceEvent(
                start_time,
                name,
                dur,
                phase_pid,
                phase_tid,
                event_args,
                node_id,
                phase_type,
                file_type=self.type,
                event_phase=phase_name,
                pid=pid,
                tid=tid,
                process_info=self._processes[phase_pid],
            )
            self._trace_events.append(t_event)
        # TODO ignoring B and E events for now. Need to check to handle it
        if phase_type == "B":
            pid = event["pid"]
            if pid not in self._pid_stacks:
                self._pid_stacks[pid] = []
            self._pid_stacks[pid].append(event)
        if phase_type == "E":
            pid = phase_pid = event["pid"]
            if pid not in self._pid_stacks:
                self.logger.info(
                    f"Did not find the 'B' type event in the pid {pid} . Skipping event: {event}"
                )
            else:
                b_event = self._pid_stacks[pid][-1]
                self._pid_stacks[pid].pop()
                start_time = b_event["ts"] + self._start_timestamp
                end_time = event["ts"] + self._start_timestamp
                duration = end_time - start_time
                if duration < 0:
                    self.logger.error(
                        f"Error in reading the events 'B' and 'E' or trace file is corrupt: pid = "
                        f"{pid}, start_time = {b_event['ts']} end_time = {event['ts']} name = "
                        f"{b_event['name']}"
                    )
                    return

                name = b_event["name"]
                event_args = event["args"] if "args" in event else None
                phase_tid = tid = b_event["tid"] if "tid" in event else "0"
                phase_name = "Unknown"
                if phase_pid in self._processes:
                    phase_name = self._processes[phase_pid].name

                if (
                    "tid" in event
                    and phase_pid in self._processes
                    and tid in self._processes[phase_pid]._threads
                ):
                    phase_tid = self._processes[phase_pid]._threads[tid]
                else:
                    # we will generate unique tid which is hash of 0 + node_id
                    phase_tid = self._populate_thread_info_for_metaevent(
                        event, node_id=node_id, phase_tid_default=phase_tid, phase_name=phase_name
                    )
                tid = phase_tid
                if event_args:
                    # Tf Detailed metrics emits pid and thread_id
                    # get actual pid of process
                    # get actual thread id of processes. depending on file type actual pid and tid may be into args
                    if "pid" in event_args:
                        pid = event_args["pid"]
                    if "thread_id" in event_args:
                        tid = event_args["thread_id"]

                t_event = TraceEvent(
                    start_time,
                    name,
                    duration,
                    phase_pid,
                    phase_tid,
                    event_args,
                    node_id,
                    "X",
                    file_type=self.type,
                    event_phase=phase_name,
                    pid=pid,
                    tid=tid,
                    process_info=self._processes[phase_pid],
                )
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
        start_time_microseconds = convert_utc_timestamp_to_microseconds(start_time, unit)
        end_time_microseconds = convert_utc_timestamp_to_microseconds(end_time, unit)
        result_events = list()
        for x_event in self._trace_events:
            # event finished before start time or event started after the end time
            if (
                x_event.end_time < start_time_microseconds
                or end_time_microseconds < x_event.start_time
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
        start_time_microseconds = end_time_microseconds = 0
        if start_time.__class__ is datetime:
            start_time_microseconds = convert_utc_datetime_to_microseconds(start_time)
        if end_time.__class__ is datetime:
            end_time_microseconds = convert_utc_datetime_to_microseconds(end_time)
        return self.get_events_within_time_range(
            start_time_microseconds, end_time_microseconds, unit=TimeUnits.MICROSECONDS
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
