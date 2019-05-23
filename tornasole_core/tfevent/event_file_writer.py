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

"""Writes events to disk in a logdir."""

import logging
import numpy as np
import os.path
import socket
import threading
import time

import six

from .event_pb2 import Event
from .summary_pb2 import Summary, SummaryMetadata
from tornasole_core.tfrecord.record_writer import RecordWriter
from .util import make_tensor_proto

logging.basicConfig()

def size_and_shape( t ):
    if type(t) == bytes or type(t) == str:
        return (len(t), [len(t)])
    return (t.nbytes, t.shape)

def make_numpy_array( x ):
    if isinstance(x, np.ndarray):
        return x
    elif np.isscalar(x):
        return np.array([x])
    elif isinstance(x, tuple):
        return np.asarray(x, dtype=x.dtype)
    else:
        raise TypeError('_make_numpy_array only accepts input types of numpy.ndarray, scalar,'
                            ' while received type {}'.format(str(type(x))))


class EventsWriter(object):
    """Writes `Event` protocol buffers to an event file. This class is ported from
    EventsWriter defined in
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/events_writer.cc"""
    def __init__(self, file_prefix, verbose=True):
        """
        Events files have a name of the form
        '/file/path/events.out.tfevents.[timestamp].[hostname][file_suffix]'
        """
        self._file_prefix = file_prefix
        self._file_suffix = ''
        self._filename = None
        self.tfrecord_writer = None
        self._num_outstanding_events = 0
        self._logger = None
        if verbose:
            self._logger = logging.getLogger(__name__)
            self._logger.setLevel(logging.INFO)

    def __del__(self):
        self.close()

    def _init_if_needed(self):
        if self.tfrecord_writer is not None:
            return
        self._filename = self._file_prefix + ".out.tfevents." + str(time.time())[:10]\
                         + "." + socket.gethostname() + self._file_suffix
        self.tfrecord_writer = RecordWriter(self._filename)
        if self._logger is not None:
                ('successfully opened events file: %s', self._filename)
        #event = Event()
        #event.wall_time = time.time()
        #self.write_event(event)
        #self.flush()  # flush the first event

    def init_with_suffix(self, file_suffix):
        """Initializes the events writer with file_suffix"""
        self._file_suffix = file_suffix
        self._init_if_needed()

    def write_event(self, event):
        """Appends event to the file."""
        # Check if event is of type event_pb2.Event proto.
        if not isinstance(event, Event):
            raise TypeError("expected an event_pb2.Event proto, "
                            " but got %s" % type(event))
        return self._write_serialized_event(event.SerializeToString())

    def _write_serialized_event(self, event_str):
        if self.tfrecord_writer is None:
            self._init_if_needed()
        self._num_outstanding_events += 1
        self.tfrecord_writer.write_record(event_str)

    def flush(self):
        """Flushes the event file to disk."""
        if self._num_outstanding_events == 0 or self.tfrecord_writer is None:
            return
        self.tfrecord_writer.flush()
        if self._logger is not None:
            self._logger.info('wrote %d %s to disk', self._num_outstanding_events,
                              'event' if self._num_outstanding_events == 1 else 'events')
        self._num_outstanding_events = 0

    def close(self):
        """Flushes the pending events and closes the writer after it is done."""
        self.flush()
        if self.tfrecord_writer is not None:
            self.tfrecord_writer.close()
            self.tfrecord_writer = None

    def name(self):
        return self._filename


def _get_sentinel_event():
    """Generate a sentinel event for terminating worker."""
    return Event()


class EventFileWriter():
    """This class is adapted from EventFileWriter in Tensorflow:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/summary/writer/event_file_writer.py
    Writes `Event` protocol buffers to an event file.
    The `EventFileWriter` class creates an event file in the specified directory,
    and asynchronously writes Event protocol buffers to the file. The Event file
    is encoded using the tfrecord format, which is similar to RecordIO.
    """

    def __init__(self, logdir, max_queue=10, flush_secs=120, filename_suffix='', verbose=True):
        """Creates a `EventFileWriter` and an event file to write to.
        On construction the summary writer creates a new event file in `logdir`.
        This event file will contain `Event` protocol buffers, which are written to
        disk via the add_event method.
        The other arguments to the constructor control the asynchronous writes to
        the event file:
        """
        self._logdir = logdir
        if not os.path.exists(self._logdir):
            os.makedirs(self._logdir)
        self._event_queue = six.moves.queue.Queue(max_queue)
        self._ev_writer = EventsWriter(os.path.join(self._logdir, "events"), verbose=verbose)
        self._ev_writer.init_with_suffix(filename_suffix)
        self._flush_secs = flush_secs
        self._sentinel_event = _get_sentinel_event()
        #if filename_suffix is not None:
        #     self._ev_writer.init_with_suffix(filename_suffix)        
        self._closed = False
        self._worker = _EventLoggerThread(self._event_queue, self._ev_writer,
                                          self._flush_secs, self._sentinel_event)

        self._worker.start()

    def get_logdir(self):
        """Returns the directory where event file will be written."""
        return self._logdir

    def reopen(self):
        """Reopens the EventFileWriter.
        Can be called after `close()` to add more events in the same directory.
        The events will go into a new events file.
        Does nothing if the `EventFileWriter` was not closed.
        """
        if self._closed:
            self._worker = _EventLoggerThread(self._event_queue, self._ev_writer,
                                              self._flush_secs, self._sentinel_event)
            self._worker.start()
            self._closed = False

    def write_graph(self, graph):
        """Adds a `Graph` protocol buffer to the event file."""
        event = Event(graph_def=graph.SerializeToString())
        self.write_event(event)


    def write_tensor(self, tdata, tname, trial, step, worker):
        plugin_data = [SummaryMetadata.PluginData(plugin_name='tensor')]
        smd = SummaryMetadata(plugin_data=plugin_data)
        value = make_numpy_array(tdata)
        tag = tname
        tensor_proto = make_tensor_proto(nparray_data=value, tag=tag)
        s = Summary(value=[Summary.Value(tag=tag, metadata=smd, tensor=tensor_proto)])
        self.write_summary(s, step)

    def write_summary(self, summary, step):
        event = Event(summary=summary)
        event.wall_time = time.time()
        event.step = step
        self.write_event(event)

    def write_event(self, event):
        """Adds an event to the event file."""
        if not self._closed:
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
        if not self._closed:
            print("Emitting sentinel")
            self.write_event(self._sentinel_event)
            self.flush()
            self._worker.join()
            self._ev_writer.close()
            self._closed = True

    def name(self):
        return self._ev_writer.name()


class _EventLoggerThread(threading.Thread):
    """Thread that logs events. Copied from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/summary/writer/event_file_writer.py#L133"""

    def __init__(self, queue, ev_writer, flush_secs, sentinel_event):
        """Creates an _EventLoggerThread."""
        threading.Thread.__init__(self)
        print( "THREAD")
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
                print("Retrieving Sentinel")
                self._queue.task_done()
                break
            try:
                self._ev_writer.write_event(event)
                # Flush the event writer every so often.
                now = time.time()
                if now > self._next_event_flush_time:
                    self._ev_writer.flush()
                    # Do it again in two minutes.
                    self._next_event_flush_time = now + self._flush_secs
            finally:
                print("FINAL")
                self._queue.task_done()
