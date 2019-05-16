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

"""APIs for logging data in the event file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from .proto import event_pb2
from .proto import summary_pb2
from .event_file_writer import EventFileWriter


class SummaryToEventTransformer(object):
    """This class is adapted with minor modifications from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/summary/writer/writer.py#L125
    Users should not use this class directly for logging MXNet data.
    This class abstractly implements the SummaryWriter API: add_summary.
    The endpoint generates an event protobuf from the Summary object, and passes
    the event protobuf to _event_writer, which is of type EventFileWriter, for logging.
    """
    def __init__(self, event_writer):
        """Initializes the _event_writer with the passed-in value.

        Parameters
        ----------
          event_writer: EventFileWriter
              An event file writer writing events to the files in the path `logdir`.
        """
        self._event_writer = event_writer
        # This set contains tags of Summary Values that have been encountered
        # already. The motivation here is that the SummaryWriter only keeps the
        # metadata property (which is a SummaryMetadata proto) of the first Summary
        # Value encountered for each tag. The SummaryWriter strips away the
        # SummaryMetadata for all subsequent Summary Values with tags seen
        # previously. This saves space.
        self._seen_summary_tags = set()

    def add_summary(self, summary, global_step=None):
        """Adds a `Summary` protocol buffer to the event file.
        This method wraps the provided summary in an `Event` protocol buffer and adds it
        to the event file.

        Parameters
        ----------
          summary : A `Summary` protocol buffer
              Optionally serialized as a string.
          global_step: Number
              Optional global step value to record with the summary.
        """
        if isinstance(summary, bytes):
            summ = summary_pb2.Summary()
            summ.ParseFromString(summary)
            summary = summ

        # We strip metadata from values with tags that we have seen before in order
        # to save space - we just store the metadata on the first value with a
        # specific tag.
        for value in summary.value:
            if not value.metadata:
                continue

            if value.tag in self._seen_summary_tags:
                # This tag has been encountered before. Strip the metadata.
                value.ClearField("metadata")
                continue

            # We encounter a value with a tag we have not encountered previously. And
            # it has metadata. Remember to strip metadata from future values with this
            # tag string.
            self._seen_summary_tags.add(value.tag)

        event = event_pb2.Event(summary=summary)
        self._add_event(event, global_step)

    def add_graph(self, graph):
        """Adds a `Graph` protocol buffer to the event file."""
        event = event_pb2.Event(graph_def=graph.SerializeToString())
        self._add_event(event, None)

    def _add_event(self, event, step):
        event.wall_time = time.time()
        if step is not None:
            event.step = int(step)
        self._event_writer.add_event(event)


class FileWriter(SummaryToEventTransformer):
    """This class is adapted from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/summary/writer/writer.py.
    Even though this class provides user-level APIs in TensorFlow, it is recommended to use the
    interfaces defined in the class `SummaryWriter` (see below) for logging in MXNet as they are
    directly compatible with the MXNet NDArray type.
    This class writes `Summary` protocol buffers to event files. The `FileWriter` class provides
    a mechanism to create an event file in a given directory and add summaries and events to it.
    The class updates the file contents asynchronously.
    """
    def __init__(self, logdir, max_queue=10, flush_secs=120, filename_suffix=None, verbose=True):
        """Creates a `FileWriter` and an event file.
        On construction the summary writer creates a new event file in `logdir`.
        This event file will contain `Event` protocol buffers constructed when you
        call one of the following functions: `add_summary()`, or `add_event()`.

        Parameters
        ----------
            logdir : str
                Directory where event file will be written.
            max_queue : int
                Size of the queue for pending events and summaries.
            flush_secs: Number
                How often, in seconds, to flush the pending events and summaries to disk.
            filename_suffix : str
                Every event file's name is suffixed with `filename_suffix` if provided.
            verbose : bool
                Determines whether to print logging messages.
        """
        event_writer = EventFileWriter(logdir, max_queue, flush_secs, filename_suffix, verbose)
        super(FileWriter, self).__init__(event_writer)

    def __enter__(self):
        """Make usable with "with" statement."""
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        """Make usable with "with" statement."""
        self.close()

    def get_logdir(self):
        """Returns the directory where event file will be written."""
        return self._event_writer.get_logdir()

    def add_event(self, event):
        """Adds an event to the event file.

        Parameters
        ----------
            event : An `Event` protocol buffer.
        """
        self._event_writer.add_event(event)

    def flush(self):
        """Flushes the event file to disk.
        Call this method to make sure that all pending events have been written to disk.
        """
        self._event_writer.flush()

    def close(self):
        """Flushes the event file to disk and close the file.
        Call this method when you do not need the summary writer anymore.
        """
        self._event_writer.close()

    def reopen(self):
        """Reopens the EventFileWriter.
        Can be called after `close()` to add more events in the same directory.
        The events will go into a new events file. Does nothing if the EventFileWriter
        was not closed.
        """
        self._event_writer.reopen()