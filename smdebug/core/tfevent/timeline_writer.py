# Standard Library
import collections
import json
import time

# First Party
from smdebug.core.access_layer.file import TSAccessFile
from smdebug.core.access_layer.s3 import TSAccessS3
from smdebug.core.logger import get_logger
from smdebug.core.utils import is_s3


"""
TimelineRecord represents one trace event that ill be written into a trace event JSON file.
"""


class TimelineRecord:
    def __init__(
        self,
        training_phase="",
        phase="X",
        operator_name="",
        args=None,
        timestamp=0,
        duration=0,
        start_since_epoch=0,
    ):
        """

        :param training_phase: strings like, data_iterating, forward, backward, operations etc
        :param phase: Trace event phases. Example "X", "B", "E", "M"
        :param operator_name: more details about phase like whether dataset or iterator
        :param args: other information to be added as args
        :param timestamp: start_time for the event
        :param duration: any duration manually computed (in seconds)
        :param start_since_epoch: start time to be added to file header as metadata. All event times are relative to this.
        """
        self.training_phase = training_phase
        self.phase = phase
        self.op_name = operator_name
        self.args = args
        self.ts_micros = timestamp
        self.duration = duration
        self.pid = 0
        self.tid = 0
        self.start_since_epoch_in_us = start_since_epoch

    def to_json(self):
        json_dict = {
            "name": self.op_name,
            "pid": self.pid,
            "ph": self.phase,
            "ts": self.ts_micros - self.start_since_epoch_in_us
            if self.ts_micros
            else int(round(time.time() * 1000000)) - self.start_since_epoch_in_us,
        }
        if self.phase == "X":
            json_dict.update({"dur": self.duration})

        if self.args:
            json_dict["args"] = self.args

        return json.dumps(json_dict)


class TimelineRecordWriter:
    """Write records in the following format for a single TimelineRecord"""

    def __init__(self, path, write_checksum, start_time_in_us):
        self.write_checksum = write_checksum
        self._writer = None
        self.start_time_in_us = start_time_in_us
        self.tensor_table = collections.defaultdict(int)
        self.is_first = True
        self.open(path, start_time_in_us)

    def __del__(self):
        self.close()

    def open(self, path, start_time_in_us):
        """
        Open the trace event file either from init or when closing and opening a file based on rotation policy
        """
        s3, bucket_name, key_name = is_s3(path)
        try:
            if s3:
                self._writer = TSAccessS3(bucket_name, key_name)
            else:
                self._writer = TSAccessFile(path, "a+")
        except (OSError, IOError) as err:
            raise ValueError("failed to open {}: {}".format(path, str(err)))
        self.start_time_in_us = start_time_in_us
        self.tensor_table = collections.defaultdict(int)
        self.is_first = True
        self._writer.write("[\n")

    def write_record(self, timeline_record):
        """Writes a TimelineRecord to file."""
        if self.tensor_table[timeline_record.training_phase] == 0:
            tensor_idx = len(self.tensor_table)
            self.tensor_table[timeline_record.training_phase] = tensor_idx

            # First writing a metadata event
            if self.is_first:
                args = {"name": "start_time_since_epoch_in_micros", "value": self.start_time_in_us}
                json_dict = {"name": "process_name", "ph": "M", "pid": 0, "args": args}
                self._writer.write(json.dumps(json_dict) + ",\n")

                args = {"sort_index": 0}
                json_dict = {"name": "process_sort_index", "ph": "M", "pid": 0, "args": args}
                self._writer.write(json.dumps(json_dict) + ",\n")

            args = {"name": timeline_record.training_phase}
            json_dict = {"name": "process_name", "ph": "M", "pid": tensor_idx, "args": args}
            self._writer.write(json.dumps(json_dict) + ",\n")

            args = {"sort_index": tensor_idx}
            json_dict = {"name": "process_sort_index", "ph": "M", "pid": tensor_idx, "args": args}
            self._writer.write(json.dumps(json_dict) + ",\n")

            self.is_first = False

        timeline_record.start_since_epoch_in_us = self.start_time_in_us
        timeline_record.pid = self.tensor_table[timeline_record.training_phase]

        # write the trace event record
        return self._writer.write(timeline_record.to_json() + ",\n")

    def flush(self):
        """Flushes the TimelineRecord string to file."""
        assert self._writer is not None
        self._writer.flush()

    def close(self):
        """Closes the record writer."""
        if self._writer is not None:
            # seeking the last ',' and replacing with ']' to mark EOF
            file_seek_pos = self._writer._accessor.tell()
            self._writer._accessor.seek(file_seek_pos - 2)
            self._writer._accessor.truncate()
            if file_seek_pos > 2:
                self._writer.write("\n]")

            self.flush()
            self._writer.close()
            self._writer = None


class TimelineWriter:
    """Writes TimelineRecord to a trace event file. This class is ported from
    EventsWriter defined in
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/events_writer.cc"""

    def __init__(self, path, index_writer=None, verbose=False, write_checksum=False):
        self._filename = path
        self.tlrecord_writer = None
        self.verbose = verbose
        self._num_outstanding_events = 0
        self._logger = get_logger()
        self.write_checksum = write_checksum
        self.index_writer = index_writer
        self.start_time_since_epoch_in_micros = int(round(time.time() * 1000000))

    def __del__(self):
        self.close()

    def open(self, path):
        self.start_time_since_epoch_in_micros = int(round(time.time() * 1000000))
        self.tlrecord_writer = TimelineRecordWriter(
            path, self.write_checksum, self.start_time_since_epoch_in_micros
        )
        self._filename = path

    def _init_if_needed(self):
        if self.tlrecord_writer is not None:
            return
        self.tlrecord_writer = TimelineRecordWriter(
            self._filename, self.write_checksum, self.start_time_since_epoch_in_micros
        )

    def init(self):
        self._init_if_needed()

    def write_event(self, record):
        """Appends trace event to the file."""
        # Check if event is of type TimelineRecord.
        if not isinstance(record, TimelineRecord):
            raise TypeError("expected a TimelineRecord, " " but got %s" % type(record))
        if self.tlrecord_writer is None:
            self._init_if_needed()
        self._num_outstanding_events += 1
        position_and_length_of_record = self.tlrecord_writer.write_record(record)
        return position_and_length_of_record

    def flush(self):
        """Flushes the trace event file to disk."""
        if self._num_outstanding_events == 0 or self.tlrecord_writer is None:
            return
        self.tlrecord_writer.flush()
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
        if self.tlrecord_writer is not None:
            self.tlrecord_writer.close()
            self.tlrecord_writer = None

    def name(self):
        return self._filename
