# Standard Library
import collections
import json
import os
import time

# First Party
from smdebug.core.access_layer.file import TSAccessFile
from smdebug.core.access_layer.s3 import TSAccessS3
from smdebug.core.config_constants import (
    CONVERT_TO_MICROSECS,
    SM_PROFILER_TRACE_FILE_PATH_CONST_STR,
)
from smdebug.core.logger import get_logger
from smdebug.core.utils import is_s3


"""
TimelineRecord represents one trace event that ill be written into a trace event JSON file.
"""


class TimelineRecord:
    def __init__(
        self, timestamp, training_phase="", phase="X", operator_name="", args=None, duration=0
    ):
        """
        :param timestamp: Mandatory field. start_time for the event
        :param training_phase: strings like, data_iterating, forward, backward, operations etc
        :param phase: Trace event phases. Example "X", "B", "E", "M"
        :param operator_name: more details about phase like whether dataset or iterator
        :param args: other information to be added as args
        :param duration: any duration manually computed (in seconds)
        """
        self.training_phase = training_phase
        self.phase = phase
        self.op_name = operator_name
        self.args = args
        self.ts_micros = int(timestamp * CONVERT_TO_MICROSECS)
        self.duration = (
            duration
            if duration
            else int(round(time.time() * CONVERT_TO_MICROSECS) - self.ts_micros)
        )
        self.pid = 0
        self.tid = 0

    def to_json(self):
        json_dict = {"name": self.op_name, "pid": self.pid, "ph": self.phase, "ts": self.ts_micros}
        if self.phase == "X":
            json_dict.update({"dur": self.duration})

        if self.args:
            json_dict["args"] = self.args

        return json.dumps(json_dict)


class TimelineWriter:
    """Writes TimelineRecord to a trace event file. This class is ported from
    EventsWriter defined in
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/events_writer.cc"""

    def __init__(self, path, verbose=False):
        self._filename = path
        self.verbose = verbose
        self._num_outstanding_events = 0
        self._logger = get_logger()
        self.start_time_since_epoch_in_micros = int(round(time.time() * CONVERT_TO_MICROSECS))
        self._writer = None
        self.tensor_table = collections.defaultdict(int)
        self.is_first = True
        self.open(path)

    def __del__(self):
        self.close()

    def open(self, path):
        """
        Open the trace event file either from init or when closing and opening a file based on rotation policy
        """
        s3, bucket_name, key_name = is_s3(path)
        try:
            if s3:
                self._writer = TSAccessS3(bucket_name, key_name, binary=False)
            else:
                self._writer = TSAccessFile(path, "a+")
        except (OSError, IOError) as err:
            raise ValueError("failed to open {}: {}".format(path, str(err)))
        self.tensor_table = collections.defaultdict(int)
        self.is_first = True
        self._writer.write("[\n")
        self._filename = path

    def write_event(self, record):
        """Appends trace event to the file."""
        # Check if event is of type TimelineRecord.
        if not isinstance(record, TimelineRecord):
            raise TypeError("expected a TimelineRecord, " " but got %s" % type(record))
        self._num_outstanding_events += 1

        if self.tensor_table[record.training_phase] == 0:
            tensor_idx = len(self.tensor_table)
            self.tensor_table[record.training_phase] = tensor_idx

            # First writing a metadata event
            if self.is_first:
                args = {"start_time_since_epoch_in_micros": self.start_time_since_epoch_in_micros}
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

        record.ts_micros -= self.start_time_since_epoch_in_micros
        record.pid = self.tensor_table[record.training_phase]

        # write the trace event record
        position_and_length_of_record = self._writer.write(record.to_json() + ",\n")
        return position_and_length_of_record

    def flush(self):
        """Flushes the trace event file to disk."""
        if self._num_outstanding_events == 0:
            return
        assert self._writer is not None
        self._writer.flush()
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
        if self._writer is not None:
            # seeking the last ',' and replacing with ']' to mark EOF
            if isinstance(self._writer, TSAccessFile):
                file_seek_pos = self._writer._accessor.tell()
                self._writer._accessor.seek(file_seek_pos - 2)
                self._writer._accessor.truncate()
            else:
                file_seek_pos = len(self._writer.data)
                self._writer.data = self._writer.data[:-2]

            if file_seek_pos > 2:
                self._writer.write("\n]")

            self.flush()
            self._writer.close(delete_if_empty=True)
            self._writer = None

    def name(self):
        return self._filename

    def file_size(self):
        if isinstance(self._writer, TSAccessFile):
            return os.path.getsize(self._filename + ".tmp")  # in bytes
        else:
            return len(self._writer.data)
