# Standard Library
import collections
import json
import os
import time
from datetime import datetime

# First Party
from smdebug.core.access_layer.file import TSAccessFile
from smdebug.core.access_layer.s3 import TSAccessS3
from smdebug.core.config_constants import (
    CONVERT_TO_MICROSECS,
    ENV_CLOSE_FILE_INTERVAL_DEFAULT,
    ENV_MAX_FILE_SIZE_DEFAULT,
    FILE_OPEN_FAIL_THRESHOLD_DEFAULT,
    SM_PROFILER_TRACE_FILE_PATH_CONST_STR,
)
from smdebug.core.locations import TraceFileLocation
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
        self.rel_ts_micros = int(timestamp * CONVERT_TO_MICROSECS)
        self.abs_ts_micros = int(timestamp * CONVERT_TO_MICROSECS)
        self.duration = (
            duration
            if duration
            else int(round(time.time() * CONVERT_TO_MICROSECS) - self.abs_ts_micros)
        )
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


class TimelineWriter:
    """Writes TimelineRecord to a trace event file. This class is ported from
    EventsWriter defined in
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/events_writer.cc"""

    def __init__(self, path, verbose=False):
        self._filename = None
        self.base_dir = path
        self.verbose = verbose
        self._num_outstanding_events = 0
        self._logger = get_logger()
        self._writer = None
        self.start_time_since_epoch_in_micros = int(round(time.time()))
        self.tensor_table = collections.defaultdict(int)
        self.continuous_fail_count = 0
        self.is_first = True
        self.last_timestamp = 0
        # self.open(path)

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
            self._logger.debug(f"Sagemaker-Debugger: failed to open {path}: {str(err)}")
            return
        self.tensor_table = collections.defaultdict(int)
        self.is_first = True
        self._writer.write("[\n")
        self._filename = path

    def _get_file_timestamp(self):
        file_path = self.name()
        path = file_path.split(SM_PROFILER_TRACE_FILE_PATH_CONST_STR)
        # get the timestamp of the current file
        fpath = path[1].split("/")[1]
        file_timestamp = int(fpath.split("_")[0])

        # returning timestamp from file name
        return file_timestamp

    def _get_rotation_info(self, now):
        # get the file size and timestamp of the current file
        file_timestamp = self._get_file_timestamp()

        file_size = self.file_size()

        # find the difference between the 2 times (in seconds)
        diff_in_seconds = int(round(now - file_timestamp))

        current_file_datehour = datetime.utcfromtimestamp(file_timestamp)
        now_datehour = datetime.utcfromtimestamp(now)

        # check if the flush is going to happen in the next hour, if so,
        # close the file, create a new directory for the next hour and write to file there
        diff_in_hours = abs(now_datehour.hour - current_file_datehour.hour)

        return file_size, diff_in_seconds, diff_in_hours

    def _should_rotate_now(self, now):
        file_size, diff_in_seconds, diff_in_hours = self._get_rotation_info(now)

        if diff_in_hours != 0:
            return True

        if diff_in_seconds > float(
            os.getenv("ENV_CLOSE_FILE_INTERVAL", ENV_CLOSE_FILE_INTERVAL_DEFAULT)
        ):
            return True

        if file_size > float(os.getenv("ENV_MAX_FILE_SIZE", ENV_MAX_FILE_SIZE_DEFAULT)):
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
        now = (
            record.abs_ts_micros + record.duration
        ) / CONVERT_TO_MICROSECS  # convert back to secs

        # check if any of the rotation policies have been satisfied. close the existing
        # trace file and open a new one
        # policy 1: if file size exceeds specified max_size
        # policy 2: if same file has been written to for close_interval time
        # policy 3: if a write is being made in the next hour, create a new directory
        if self._writer and self._should_rotate_now(now):
            self.close()
            self.last_timestamp = now

        el = TraceFileLocation()
        new_file_path = el.get_file_location(base_dir=self.base_dir, timestamp=now)

        #  if file has not been created yet, create now
        if not self._writer:
            self.open(path=new_file_path)
            if not self._writer:
                self.continuous_fail_count += 1

                if self.continuous_fail_count >= int(
                    os.getenv("FILE_OPEN_FAIL_THRESHOLD", FILE_OPEN_FAIL_THRESHOLD_DEFAULT)
                ):
                    self._logger.warning(
                        "Encountered {} number of continuous failures while trying to open the file. "
                        "Empty the record queue and mark the writer unhealthy.".format(
                            str(
                                os.getenv(
                                    "FILE_OPEN_FAIL_THRESHOLD", FILE_OPEN_FAIL_THRESHOLD_DEFAULT
                                )
                            )
                        )
                    )
                    return

        if self._writer and now > self.last_timestamp:
            if isinstance(self._writer, TSAccessFile):
                os.rename(self._filename + ".tmp", new_file_path + ".tmp")
            elif isinstance(self._writer, TSAccessS3):
                s3, bucket_name, key_name = is_s3(new_file_path)
                self._writer.rename(key_name=key_name)
            self.last_timestamp = now

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

        record.rel_ts_micros -= self.start_time_since_epoch_in_micros
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
            self._writer.close()
            self._writer = None

    def name(self):
        return self._filename

    def file_size(self):
        if isinstance(self._writer, TSAccessFile):
            return os.path.getsize(self._filename + ".tmp")  # in bytes
        else:
            return len(self._writer.data)
