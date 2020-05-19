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
import os
import threading
import time
from datetime import datetime

# Third Party
import six

# First Party
from smdebug.core.config_constants import SM_PROFILER_TRACE_FILE_PATH_CONST_STR
from smdebug.core.locations import TraceFileLocation
from smdebug.core.tfevent.timeline_writer import TimelineRecord, TimelineWriter
from smdebug.core.utils import is_s3


def _get_sentinel_event():
    """Generate a sentinel trace event for terminating worker."""
    return TimelineRecord()


def _get_size_and_timestamp(file_path, ev_writer):
    s3, bucket_name, key_name = is_s3(file_path)
    path = file_path.split(SM_PROFILER_TRACE_FILE_PATH_CONST_STR)
    # get the timestamp of the current file's folder
    fpath = path[1].split("/")[1]
    file_timestamp = int(fpath.split("_")[0])
    if s3:
        file_size = ev_writer.tlrecord_writer._writer.get_file_size()
    else:
        file_size = os.path.getsize(file_path + ".tmp")  # in bytes

    return file_size, file_timestamp


class TimelineFileWriter:
    """This class is adapted from EventFileWriter in Tensorflow:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/summary/writer/event_file_writer.py
    Writes TimelineRecord to a JSON trace file.
    The `TimelineFileWriter` class creates a timeline JSON file in the specified directory,
    and asynchronously writes TimelineRecord to the file.
    """

    def __init__(self, path, max_queue=10, flush_secs=120):
        """Creates a `TimelineFileWriter` and a trace event file to write to.
        This event file will contain TimelineRecord as JSON strings, which are written to
        disk via the write_record method.
        The other arguments to the constructor control the asynchronous writes to
        the event file:
        """
        self._path = path
        self._event_queue = six.moves.queue.Queue(max_queue)
        self._ev_writer = TimelineWriter(path=self._path)
        self._ev_writer.init()
        self._flush_secs = flush_secs
        self._sentinel_event = _get_sentinel_event()
        self._worker = _TimelineLoggerThread(
            queue=self._event_queue,
            ev_writer=self._ev_writer,
            flush_secs=self._flush_secs,
            sentinel_event=self._sentinel_event,
        )
        self._worker.start()

    def write_trace_events(
        self, training_phase="", op_name="", phase="X", timestamp=None, duration=1, args=None
    ):
        duration_in_us = int(duration * 1000000)  # convert to micro seconds
        event = TimelineRecord(
            training_phase=training_phase,
            operator_name=op_name,
            phase=phase,
            timestamp=timestamp,
            args=args,
            duration=duration_in_us,
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
        self._ev_writer.flush()

    def close(self):
        """Flushes the trace event file to disk and close the file.
        """
        self.write_event(self._sentinel_event)
        self._worker.join()
        self._ev_writer.close()

    def name(self):
        return self._ev_writer.name()


class _TimelineLoggerThread(threading.Thread):
    """Thread that logs events. Copied from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/summary/writer/event_file_writer.py#L133"""

    def __init__(self, queue, ev_writer, flush_secs, sentinel_event):
        """Creates a _TimelineLoggerThread."""
        threading.Thread.__init__(self)
        self.daemon = True
        self._queue = queue
        self._ev_writer = ev_writer
        self._flush_secs = flush_secs
        # The first event will be flushed immediately.
        self._next_event_flush_time = 0
        self._sentinel_event = sentinel_event

    def run(self):
        while True:
            event = self._queue.get()

            if event is self._sentinel_event:
                self._queue.task_done()
                break

            try:
                # write event
                _ = self._ev_writer.write_event(event)

                # Flush the event writer every so often.
                now = time.time()
                if now > self._next_event_flush_time:
                    """
                    Rotation policy:
                    Close file if file size exceeds $ENV_MAX_FILE_SIZE or folder was created more than
                    $ENV_CLOSE_FILE_INTERVAL time duration.
                    """
                    file_name = self._ev_writer.name()
                    path = file_name.split(SM_PROFILER_TRACE_FILE_PATH_CONST_STR)

                    # get the file size of the current directory
                    file_size, file_timestamp = _get_size_and_timestamp(
                        file_name, ev_writer=self._ev_writer
                    )

                    # find the difference between the 2 times (in seconds)
                    diff_in_seconds = int(round(now - file_timestamp))

                    current_file_datehour = datetime.fromtimestamp(file_timestamp)
                    now_datehour = datetime.fromtimestamp(now)

                    # check if the flush is going to happen in the next hour, if so,
                    # close the file, create a new directory for the next hour and write to file there
                    diff_in_hours = abs(now_datehour.hour - current_file_datehour.hour)

                    # check if any of the rotation policies have been satisfied. close the existing
                    # trace file and open a new one
                    # policy 1: if file size exceeds specified max_size
                    # policy 2: if same file has been written to for close_interval time
                    # policy 3: if a write is being made in the next hour, create a new directory
                    # TODO: what is the default max_file_size and close_file_interval?
                    if (
                        file_size > int(os.getenv("ENV_MAX_FILE_SIZE", file_size))
                        or diff_in_seconds > int(os.getenv("ENV_CLOSE_FILE_INTERVAL", 3600))
                        or diff_in_hours != 0
                    ):
                        self._ev_writer.close()
                        el = TraceFileLocation()
                        new_file_path = el.get_file_location(base_dir=path[0])
                        self._ev_writer.open(path=new_file_path)
                    self._ev_writer.flush()
                    # Do it again in _flush_secs time.
                    self._next_event_flush_time = now + self._flush_secs
            finally:
                self._queue.task_done()
