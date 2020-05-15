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

"""Writes events to disk in a trial dir."""

# Standard Library
import threading
import time

# Third Party
import six

# First Party
from smdebug.core.tfevent.timeline_writer import TimelineRecord, TimelineWriter


def size_and_shape(t):
    if type(t) == bytes or type(t) == str:
        return (len(t), [len(t)])
    return (t.nbytes, t.shape)


def _get_sentinel_event():
    """Generate a sentinel event for terminating worker."""
    return TimelineRecord()


class TimelineFileWriter:
    """This class is adapted from EventFileWriter in Tensorflow:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/summary/writer/event_file_writer.py
    Writes `Event` protocol buffers to an event file.
    The `EventFileWriter` class creates an event file in the specified directory,
    and asynchronously writes Event protocol buffers to the file. The Event file
    is encoded using the tfrecord format, which is similar to RecordIO.
    """

    def __init__(
        self,
        path,
        max_queue=10,
        flush_secs=120,
        index_writer=None,
        verbose=False,
        write_checksum=False,
    ):
        """Creates a `EventFileWriter` and an event file to write to.
        On construction the summary writer creates a new event file in `logdir`.
        This event file will contain `Event` protocol buffers, which are written to
        disk via the add_event method.
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
        self, tensor_name="", op_name="", phase="X", timestamp=None, duration=1, worker=0, args=None
    ):
        duration_in_us = int(duration * 100000)
        event = TimelineRecord(
            tensor_name=tensor_name,
            operator_name=op_name,
            phase=phase,
            timestamp=timestamp,
            args=args,
            duration=duration_in_us,
        )
        self.write_event(event)

    def write_event(self, event):
        """Adds an event to the event file."""
        self._event_queue.put(event)

    def flush(self):
        """Flushes the event file to disk.
        Call this method to make sure that all pending events have been written to disk.
        """
        self._event_queue.join()
        self._ev_writer.flush()

    def close(self):
        """Flushes the event file to disk and close the file.
        Call this method when you do not need the summary writer anymore.
        """
        self.write_event(self._sentinel_event)
        self.flush()
        self._worker.join()
        self._ev_writer.close()

    def name(self):
        return self._ev_writer.name()


class _TimelineLoggerThread(threading.Thread):
    """Thread that logs events. Copied from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/summary/writer/event_file_writer.py#L133"""

    def __init__(self, queue, ev_writer, flush_secs, sentinel_event):
        """Creates an _EventLoggerThread."""
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
                    self._ev_writer.flush()
                    # Do it again in two minutes.
                    self._next_event_flush_time = now + self._flush_secs
            finally:
                self._queue.task_done()
