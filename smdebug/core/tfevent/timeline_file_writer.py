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
from smdebug.core.config_constants import (
    CONVERT_TO_MICROSECS,
    ENV_CLOSE_FILE_INTERVAL_DEFAULT,
    ENV_MAX_FILE_SIZE_DEFAULT,
    FILE_OPEN_FAIL_THRESHOLD_DEFAULT,
)
from smdebug.core.locations import TraceFileLocation
from smdebug.core.logger import get_logger
from smdebug.core.utils import ensure_dir

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
        self.abs_ts_micros = int(timestamp * CONVERT_TO_MICROSECS)
        self.rel_ts_micros = self.abs_ts_micros - self.base_start_time
        self.duration = (
            duration
            if duration
            else int(round(time.time() * CONVERT_TO_MICROSECS) - self.abs_ts_micros)
        )
        self.event_end_ts_micros = self.abs_ts_micros + self.duration
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

    def __init__(self, path, max_queue=100):
        """Creates a `TimelineFileWriter` and a trace event file to write to.
        This event file will contain TimelineRecord as JSON strings, which are written to
        disk via the write_record method.
        The other arguments to the constructor control the asynchronous writes to
        the event file:
        """
        self.start_time_since_epoch_in_micros = int(round(time.time() * CONVERT_TO_MICROSECS))
        self._path = path
        self._event_queue = six.moves.queue.Queue(max_queue)
        self._sentinel_event = _get_sentinel_event(self.start_time_since_epoch_in_micros)
        self._worker = _TimelineLoggerThread(
            queue=self._event_queue,
            sentinel_event=self._sentinel_event,
            path=path,
            base_start_time=self.start_time_since_epoch_in_micros,
        )
        self._worker.start()

    def write_trace_events(
        self, timestamp, training_phase="", op_name="", phase="X", duration=0, args=None
    ):
        if not self._worker._healthy:
            return
        duration_in_us = int(duration * CONVERT_TO_MICROSECS)  # convert to micro seconds
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

    def __init__(self, queue, sentinel_event, path, base_start_time, verbose=False):
        """Creates a _TimelineLoggerThread."""
        threading.Thread.__init__(self)
        self.daemon = False
        self._queue = queue
        self._sentinel_event = sentinel_event
        self.base_dir = path
        self._num_outstanding_events = 0
        self._writer = None
        self.verbose = verbose
        self.tensor_table = collections.defaultdict(int)
        self.continuous_fail_count = 0
        self.is_first = True
        self.last_event_end_time = int(round(base_start_time / CONVERT_TO_MICROSECS))
        self.last_file_close_time = self.last_event_end_time
        self.cur_hour = datetime.utcfromtimestamp(self.last_file_close_time).hour
        self._filename = TraceFileLocation().get_file_location(
            base_dir=self.base_dir, timestamp=self.last_event_end_time
        )
        self._healthy = True

        self.file_close_interval = float(
            os.getenv("ENV_CLOSE_FILE_INTERVAL", ENV_CLOSE_FILE_INTERVAL_DEFAULT)
        )
        self.file_max_size = float(os.getenv("ENV_MAX_FILE_SIZE", ENV_MAX_FILE_SIZE_DEFAULT))
        self.file_open_fail_threshold = int(
            os.getenv("FILE_OPEN_FAIL_THRESHOLD", FILE_OPEN_FAIL_THRESHOLD_DEFAULT)
        )

    def run(self):
        while True:
            event = self._queue.get()

            if not self._healthy or event is self._sentinel_event:
                self._queue.task_done()
                break

            try:
                # write event
                _ = self.write_event(event)
            finally:
                self._queue.task_done()
            time.sleep(0)

    def open(self, path):
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
        # self._filename = path
        self._healthy = True
        return True

    def _get_rotation_info(self, now):
        file_size = self.file_size()

        # find the difference between the 2 times (in seconds)
        diff_in_seconds = int(round(now - self.last_file_close_time))

        now_datehour = datetime.utcfromtimestamp(now)

        # check if the flush is going to happen in the next hour, if so,
        # close the file, create a new directory for the next hour and write to file there
        diff_in_hours = abs(now_datehour.hour - self.cur_hour)

        return file_size, diff_in_seconds, diff_in_hours

    def _should_rotate_now(self, now):
        file_size, diff_in_seconds, diff_in_hours = self._get_rotation_info(now)

        if diff_in_hours != 0:
            self.cur_hour = datetime.utcfromtimestamp(now).hour
            return True

        if diff_in_seconds > self.file_close_interval:
            return True

        if file_size > self.file_max_size:
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
            self.last_file_close_time = self.last_event_end_time

        #  if file has not been created yet, create now
        if not self._writer:
            file_opened = self.open(path=self._filename)
            if not file_opened:
                if self.continuous_fail_count >= self.file_open_fail_threshold:
                    logger.warning(
                        "Encountered {} number of continuous failures while trying to open the file. "
                        "Marking the writer unhealthy. All future events will be dropped.".format(
                            str(self.file_open_fail_threshold)
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
        self.last_event_end_time = int(round(end_time_for_event))
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
            os.rename(
                self._filename,
                TraceFileLocation().get_file_location(
                    base_dir=self.base_dir, timestamp=self.last_event_end_time
                ),
            )
            self._writer = None

    def name(self):
        return self._filename

    def file_size(self):
        return os.path.getsize(self._filename)  # in bytes
