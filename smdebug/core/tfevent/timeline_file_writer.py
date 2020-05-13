# Standard Library
import collections
import json
import time

# First Party
from smdebug.core.access_layer.file import TSAccessFile
from smdebug.core.access_layer.s3 import TSAccessS3
from smdebug.core.utils import is_s3


class TimelineWriter(object):
    def __init__(self, file_path):
        """ Writer is initialized upon adding the first index. """
        self.file_path = file_path
        self.event_payload = []
        self.writer = None
        self.is_first = True
        self.tensor_table = collections.defaultdict(int)
        self.start_time_since_epoch_in_micros = int(round(time.time() * 1000000))
        self._init_writer()
        self.writer.write("[\n")

    def __exit__(self):
        self.close()

    def _init_writer(self):
        s3, bucket_name, key_name = is_s3(self.file_path)
        if s3:
            self.writer = TSAccessS3(bucket_name, key_name, binary=False)
        else:
            self.writer = TSAccessFile(self.file_path, "a+")

    def write_trace_events(
        self, tensor_name="", op_name="", timestamp=None, duration=1, worker=0, args=None
    ):
        # args = {
        #     # "start_timestamp": timestamp - duration if timestamp else time.time() - duration,
        #     # "end_timestamp": timestamp if timestamp else time.time(),
        #     "step number": step_num
        # }
        # args["start_timestamp"] = int(args["start_timestamp"] * 100000)
        # args["end_timestamp"] = int(args["end_timestamp"] * 100000)
        duration_in_us = int(duration * 100000)
        event = Event(
            tensor_name=tensor_name,
            op_name=op_name,
            timestamp=timestamp,
            args=args,
            duration=duration_in_us,
            start_time_since_epoch=self.start_time_since_epoch_in_micros,
        )
        self.add_event(event)

    def add_event(self, event):
        if not self.writer:
            self._init_writer()
        self.event_payload.append(event)

    def flush(self):
        """Flushes the event string to file."""
        if not self.writer:
            return

        for event in self.event_payload:
            if self.tensor_table[event.tensor_name] == 0:
                tensor_idx = len(self.tensor_table)
                self.tensor_table[event.tensor_name] = tensor_idx
                if self.is_first:
                    args = {
                        "name": "start_time_since_epoch_in_micros",
                        "value": self.start_time_since_epoch_in_micros,
                    }
                    metadata = MetaData(name="process_name", args=args, tensor_idx=0)
                    self.writer.write(metadata.to_json() + ",\n")

                    args = {"sort_index": 0}
                    metadata = MetaData(name="process_sort_index", args=args, tensor_idx=0)
                    self.writer.write(metadata.to_json() + ",\n")

                args = {"name": event.tensor_name}
                metadata = MetaData(name="process_name", args=args, tensor_idx=tensor_idx)
                self.writer.write(metadata.to_json() + ",\n")

                args = {"sort_index": tensor_idx}
                metadata = MetaData(name="process_sort_index", args=args, tensor_idx=tensor_idx)
                self.writer.write(metadata.to_json() + ",\n")

                self.is_first = False

            event.pid = self.tensor_table[event.tensor_name]
            self.writer.write(event.to_json() + ",\n")
        self.writer.flush()
        self.event_payload = []

    def close(self):
        """Closes the timeline writer."""
        if self.writer is not None:
            self.writer._accessor.seek(self.writer._accessor.tell() - 2)
            self.writer._accessor.truncate()
            self.writer.write("\n]")
            self.flush()
            self.writer.close()
            self.writer = None


class Event:
    def __init__(
        self,
        tensor_name="",
        op_name="",
        phase="X",
        args=None,
        timestamp=None,
        duration=1,
        start_time_since_epoch=0,
    ):
        self.tensor_name = tensor_name
        self.op_name = op_name
        self.phase = phase
        self.pid = 0
        self.args = args
        self.timestamp = timestamp
        self.duration = duration
        self.start_time_since_epoch_in_micros = start_time_since_epoch

    def to_json(self):
        json_dict = {
            "name": self.op_name,
            "ph": self.phase,
            "ts": self.timestamp - self.start_time_since_epoch_in_micros
            if self.timestamp
            else int(round(time.time() * 1000000)) - self.start_time_since_epoch_in_micros,
            "pid": self.pid,
            "dur": self.duration,
        }
        if self.args:
            json_dict["args"] = self.args

        return json.dumps(json_dict)


class MetaData:
    def __init__(self, name="", tensor_idx=0, args=None):
        self.name = name
        self.tensor_idx = tensor_idx
        self.args = args

    def to_json(self):
        json_dict = {"name": self.name, "ph": "M", "pid": self.tensor_idx}
        if self.args:
            json_dict["args"] = self.args

        return json.dumps(json_dict)
