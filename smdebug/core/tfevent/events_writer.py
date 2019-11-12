# First Party
from smdebug.core.logger import get_logger
from smdebug.core.tfevent.proto.event_pb2 import Event
from smdebug.core.tfrecord.record_writer import RecordWriter


class EventsWriter:
    """Writes `Event` protocol buffers to an event file. This class is ported from
    EventsWriter defined in
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/events_writer.cc"""

    def __init__(self, path, index_writer=None, verbose=False, write_checksum=False):
        self._filename = path
        self.tfrecord_writer = None
        self.verbose = verbose
        self._num_outstanding_events = 0
        self._logger = get_logger()
        self.write_checksum = write_checksum
        self.index_writer = index_writer

    def __del__(self):
        self.close()

    def _init_if_needed(self):
        if self.tfrecord_writer is not None:
            return
        self.tfrecord_writer = RecordWriter(self._filename, self.write_checksum)

    def init(self):
        self._init_if_needed()

    def write_event(self, event):
        """Appends event to the file."""
        # Check if event is of type event_pb2.Event proto.
        if not isinstance(event, Event):
            raise TypeError("expected an event_pb2.Event proto, " " but got %s" % type(event))
        return self._write_serialized_event(event.SerializeToString())

    def _write_serialized_event(self, event_str):
        if self.tfrecord_writer is None:
            self._init_if_needed()
        self._num_outstanding_events += 1
        position_and_length_of_record = self.tfrecord_writer.write_record(event_str)
        return position_and_length_of_record

    def flush(self):
        """Flushes the event file to disk."""
        if self._num_outstanding_events == 0 or self.tfrecord_writer is None:
            return
        self.tfrecord_writer.flush()
        if self.verbose and self._logger is not None:
            self._logger.debug(
                "wrote %d %s to disk",
                self._num_outstanding_events,
                "event" if self._num_outstanding_events == 1 else "events",
            )
        self._num_outstanding_events = 0

    def close(self):
        """Flushes the pending events and closes the writer after it is done."""
        self.flush()
        if self.tfrecord_writer is not None:
            self.tfrecord_writer.close()
            self.tfrecord_writer = None

    def name(self):
        return self._filename
