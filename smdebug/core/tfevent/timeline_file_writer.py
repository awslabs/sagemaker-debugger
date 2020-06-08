# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Writes trace events to disk in trial dir or user-specified dir."""

# Standard Library
import collections
import json
import os
import threading
import time
from datetime import datetime

# Third Party
import six

# First Party
from smdebug.core.access_layer.file import SMDEBUG_TEMP_PATH_SUFFIX
from smdebug.core.locations import TraceFileLocation
from smdebug.core.logger import get_logger
from smdebug.core.utils import ensure_dir, get_node_id
from smdebug.profiler.profiler_constants import CONVERT_TO_MICROSECS

logger = get_logger()


def _get_sentinel_event(base_start_time):
    """Generate a sentinel trace event for terminating worker."""
    return TimelineRecord(timestamp=time.time(), base_start_time=base_start_time)


"""
TimelineRecord represents one trace event that ill be written into a trace event JSON file.
"""


class TimelineRecord:
    def __init__(
        self,
        timestamp,
        base_start_time,
        training_phase="",
        phase="X",
        operator_name="",
        args=None,
        duration=0,
    ):
        """
        :param timestamp: Mandatory field. start_time for the event
        :param training_phase: strings like, train_step_time , test_step_time etc
        :param phase: Trace event phases. Example "X", "B", "E", "M"
        :param operator_name: more details about phase like whether dataset or iterator
        :param args: other information to be added as args
        :param duration: any duration manually computed (in seconds)
        """
        self.training_phase = training_phase
        self.phase = phase
        self.op_name = operator_name
        self.args = args
        self.base_start_time = base_start_time
        abs_ts_micros = int(timestamp * CONVERT_TO_MICROSECS)
        self.rel_ts_micros = abs_ts_micros - self.base_start_time
        self.duration = (
            duration if duration else int(round(time.time() * CONVERT_TO_MICROSECS) - abs_ts_micros)
        )
        self.event_end_ts_micros = abs_ts_micros + self.duration
        self.pid = 0
        self.tid = 0

    def to_json(self):
        json_dict = {
            "name": self.op_name,
            "pid": self.pid,
            "ph": self.phase,
            "ts": self.rel_ts_micros,
        }
        if self.phase == "X":
            json_dict.update({"dur": self.duration})

        if self.args:
            json_dict["args"] = self.args

        return json.dumps(json_dict)


class TimelineFileWriter:
    """This class is adapted from EventFileWriter in Tensorflow:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/summary/writer/event_file_writer.py
    Writes TimelineRecord to a JSON trace file.
    The `TimelineFileWriter` class creates a timeline JSON file in the specified directory,
    and asynchronously writes TimelineRecord to the file.
    """

    def __init__(self, profiler_config_parser, max_queue=100):
        """Creates a `TimelineFileWriter` and a trace event file to write to.
        This event file will contain TimelineRecord as JSON strings, which are written to
        disk via the write_record method.
        If the profiler is not enabled, trace events will not be written to the file.
        The other arguments to the constructor control the asynchronous writes to
        the event file:
        """
        self.start_time_since_epoch_in_micros = int(round(time.time() * CONVERT_TO_MICROSECS))
        self._profiler_config_parser = profiler_config_parser
        self._event_queue = six.moves.queue.Queue(max_queue)
        self._sentinel_event = _get_sentinel_event(self.start_time_since_epoch_in_micros)
        self._worker = _TimelineLoggerThread(
            queue=self._event_queue,
            sentinel_event=self._sentinel_event,
            base_start_time=self.start_time_since_epoch_in_micros,
            profiler_config_parser=self._profiler_config_parser,
        )
        self._worker.start()

    def write_trace_events(
        self, timestamp, training_phase="", op_name="", phase="X", duration=0, **kwargs
    ):
        if not self._worker._healthy or not self._profiler_config_parser.enabled:
            return
        duration_in_us = int(duration * CONVERT_TO_MICROSECS)  # convert to micro seconds
        args = {**kwargs}
        event = TimelineRecord(
            training_phase=training_phase,
            operator_name=op_name,
            phase=phase,
            timestamp=timestamp,
            args=args,
            duration=duration_in_us,
            base_start_time=self.start_time_since_epoch_in_micros,
        )
        self.write_event(event)

    def write_event(self, event):
        """Adds a trace event to the JSON file."""
        self._event_queue.put(event)

    def flush(self):
        """Flushes the trace event file to disk.
        Call this method to make sure that all pending events have been written to disk.
        """
        self._event_queue.join()
        self._worker.flush()

    def close(self):
        """Flushes the trace event file to disk and close the file.
        """
        self.write_event(self._sentinel_event)
        self._worker.join()
        self._worker.close()


class _TimelineLoggerThread(threading.Thread):
    """Thread that logs events. Copied from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/summary/writer/event_file_writer.py#L133"""

    def __init__(
        self, queue, sentinel_event, base_start_time, profiler_config_parser, verbose=False
    ):
        """Creates a _TimelineLoggerThread."""
        threading.Thread.__init__(self)
        self.daemon = True
        self._queue = queue
        self._sentinel_event = sentinel_event
        self._num_outstanding_events = 0
        self._writer = None
        self.verbose = verbose
        self.tensor_table = collections.defaultdict(int)
        self.continuous_fail_count = 0
        self.is_first = True
        self.last_event_end_time = int(round(base_start_time / CONVERT_TO_MICROSECS))
        self.last_file_close_time = self.last_event_end_time
        self.cur_hour = datetime.utcfromtimestamp(self.last_file_close_time).hour
        self._healthy = True
        self._profiler_config_parser = profiler_config_parser
        self.node_id = get_node_id()

    def run(self):
        while True:
            # if there is long interval between 2 events, just keep checking if
            # the file is still open. if it is open for too long and a new event
            # has not occurred, close the open file based on rotation policy.
            if self._writer and self._should_rotate_now(time.time()):
                self.close()

            event = self._queue.get()

            if (
                not self._healthy
                or not self._profiler_config_parser.enabled
                or event is self._sentinel_event
            ):
                self._queue.task_done()
                break

            try:
                # write event
                _ = self.write_event(event)
            finally:
                self._queue.task_done()
            time.sleep(0)

    def open(self, path, cur_event_end_time):
        """
        Open the trace event file either from init or when closing and opening a file based on rotation policy
        """
        try:
            ensure_dir(path)
            self._writer = open(path, "a+")
        except (OSError, IOError) as err:
            logger.debug(f"Sagemaker-Debugger: failed to open {path}: {str(err)}")
            self.continuous_fail_count += 1
            return False
        self.tensor_table = collections.defaultdict(int)
        self.is_first = True
        self._writer.write("[\n")
        self._healthy = True
        self.cur_hour = datetime.utcfromtimestamp(cur_event_end_time).hour
        return True

    def _get_rotation_info(self, now):
        file_size = self.file_size()

        # find the difference between the now and last file closed time (in seconds)
        diff_in_seconds = int(round(now - self.last_file_close_time))

        now_datehour = datetime.utcfromtimestamp(now)

        # check if the flush is going to happen in the next hour, if so,
        # close the file, create a new directory for the next hour and write to file there
        diff_in_hours = abs(now_datehour.hour - self.cur_hour)

        return file_size, diff_in_seconds, diff_in_hours

    def _should_rotate_now(self, now):
        file_size, diff_in_seconds, diff_in_hours = self._get_rotation_info(now)
        rotation_policy = self._profiler_config_parser.config.trace_file.rotation_policy

        if diff_in_hours != 0:
            return True

        if diff_in_seconds > rotation_policy.file_close_interval:
            return True

        if file_size > rotation_policy.file_max_size:
            return True

        return False

    def write_event(self, record):
        """Appends trace event to the file."""
        # Check if event is of type TimelineRecord.
        if not isinstance(record, TimelineRecord):
            raise TypeError("expected a TimelineRecord, " " but got %s" % type(record))
        self._num_outstanding_events += 1

        """
        Rotation policy:
        Close file if file size exceeds $ENV_MAX_FILE_SIZE or folder was created more than
        $ENV_CLOSE_FILE_INTERVAL time duration.
        """
        end_time_for_event = (
            record.event_end_ts_micros / CONVERT_TO_MICROSECS
        )  # convert back to secs

        # check if any of the rotation policies have been satisfied. close the existing
        # trace file and open a new one
        # policy 1: if file size exceeds specified max_size
        # policy 2: if same file has been written to for close_interval time
        # policy 3: if a write is being made in the next hour, create a new directory
        if self._writer and self._should_rotate_now(end_time_for_event):
            self.close()

        #  if file has not been created yet, create now
        if not self._writer:
            file_opened = self.open(path=self.name(), cur_event_end_time=end_time_for_event)
            if not file_opened:
                file_open_fail_threshold = (
                    self._profiler_config_parser.config.trace_file.file_open_fail_threshold
                )
                if self.continuous_fail_count >= file_open_fail_threshold:
                    logger.warning(
                        "Encountered {} number of continuous failures while trying to open the file. "
                        "Marking the writer unhealthy. All future events will be dropped.".format(
                            str(file_open_fail_threshold)
                        )
                    )
                    self._healthy = False
                return

        if self.tensor_table[record.training_phase] == 0:
            tensor_idx = len(self.tensor_table)
            self.tensor_table[record.training_phase] = tensor_idx

            # First writing a metadata event
            if self.is_first:
                args = {"start_time_since_epoch_in_micros": record.base_start_time}
                json_dict = {"name": "process_name", "ph": "M", "pid": 0, "args": args}
                self._writer.write(json.dumps(json_dict) + ",\n")

                args = {"sort_index": 0}
                json_dict = {"name": "process_sort_index", "ph": "M", "pid": 0, "args": args}
                self._writer.write(json.dumps(json_dict) + ",\n")

            args = {"name": record.training_phase}
            json_dict = {"name": "process_name", "ph": "M", "pid": tensor_idx, "args": args}
            self._writer.write(json.dumps(json_dict) + ",\n")

            args = {"sort_index": tensor_idx}
            json_dict = {"name": "process_sort_index", "ph": "M", "pid": tensor_idx, "args": args}
            self._writer.write(json.dumps(json_dict) + ",\n")

            self.is_first = False

        record.pid = self.tensor_table[record.training_phase]

        # write the trace event record
        position_and_length_of_record = self._writer.write(record.to_json() + ",\n")
        self.flush()
        self.last_event_end_time = int(round(record.event_end_ts_micros / CONVERT_TO_MICROSECS))
        return position_and_length_of_record

    def flush(self):
        """Flushes the trace event file to disk."""
        if self._num_outstanding_events == 0:
            return
        if self._writer is not None:
            self._writer.flush()
            if self.verbose and logger is not None:
                logger.debug(
                    "wrote %d %s to disk",
                    self._num_outstanding_events,
                    "event" if self._num_outstanding_events == 1 else "events",
                )
            self._num_outstanding_events = 0

    def close(self):
        """Flushes the pending events and closes the writer after it is done."""
        if self._writer is not None:
            # seeking the last ',' and replacing with ']' to mark EOF
            file_seek_pos = self._writer.tell()
            self._writer.seek(file_seek_pos - 2)
            self._writer.truncate()

            if file_seek_pos > 2:
                self._writer.write("\n]")

            self.flush()
            self._writer.close()

            if self._profiler_config_parser.enabled:
                # ensure that there's a directory for the new file name
                new_file_name = TraceFileLocation().get_file_location(
                    base_dir=self._profiler_config_parser.config.local_path,
                    timestamp=self.last_event_end_time,
                )
                ensure_dir(new_file_name)
                os.rename(self.name(), new_file_name)

            self._writer = None
            self.last_file_close_time = time.time()

    def name(self):
        return (
            self._profiler_config_parser.config.local_path
            + "/"
            + self.node_id
            + SMDEBUG_TEMP_PATH_SUFFIX
        )

    def file_size(self):
        return os.path.getsize(self.name())  # in bytes
