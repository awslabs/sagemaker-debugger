# Standard Library
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
        self.start_time_since_epoch_in_micros = int(round(time.time() * 1000000))
        self._init_writer()

    def __exit__(self):
        self.close()

    def _init_writer(self):
        s3, bucket_name, key_name = is_s3(self.file_path)
        if s3:
            self.writer = TSAccessS3(bucket_name, key_name, binary=False)
        else:
            self.writer = TSAccessFile(self.file_path, "a+")
        self.writer.write("[")

        # TODO: kannanva: probably have a write_marker()
        args = {
            "name": "start_time_since_epoch_in_micros",
            "value": self.start_time_since_epoch_in_micros,
        }
        marker = Marker(name="StepTime", args=args, worker="0")
        self.writer.write(marker.to_json() + ",\n")

    def write_trace_events(self, tensor_name="", timestamp=None, duration=1, worker=0, args=None):
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
            timestamp=timestamp,
            args=args,
            duration=duration_in_us,
            worker=worker,
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
            self.writer.write(event.to_json() + ",\n")
        self.writer.flush()
        self.event_payload = []

    def close(self):
        """Closes the record writer."""
        if self.writer is not None:
            self.flush()
            self.writer.close()
            self.writer = None


class Event:
    def __init__(
        self,
        tensor_name="",
        phase="X",
        worker="",
        args=None,
        timestamp=None,
        duration=1,
        start_time_since_epoch=0,
    ):
        self.tensor_name = tensor_name
        self.phase = phase
        self.worker = worker
        self.args = args
        self.timestamp = timestamp
        self.duration = duration
        self.start_time_since_epoch_in_micros = start_time_since_epoch

    def to_json(self):
        json_dict = {
            "name": self.tensor_name,
            "ph": self.phase,
            "ts": self.timestamp - self.start_time_since_epoch_in_micros
            if self.timestamp
            else int(round(time.time() * 1000000)) - self.start_time_since_epoch_in_micros,
            "pid": self.worker,  # TODO: kannanva: pid should be tensor index. For now, it is worker name.
            "dur": self.duration,
        }
        if self.args:
            json_dict["args"] = self.args

        return json.dumps(json_dict)


class Marker:
    def __init__(self, name="", worker="", args=None):
        self.name = name
        self.worker = worker
        self.args = args

    def to_json(self):
        json_dict = {
            "name": self.name,
            "ph": "M",
            "pid": self.worker,  # TODO: kannanva: pid should be tensor index. For now, it is worker name.
        }
        if self.args:
            json_dict["args"] = self.args

        return json.dumps(json_dict)
