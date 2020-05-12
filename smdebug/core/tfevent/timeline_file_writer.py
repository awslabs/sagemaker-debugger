# Standard Library
import json
import os
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

    def __exit__(self):
        self.close()

    def _init_writer(self):
        # TODO: kannanva: file path remains as .tmp. Need to check this.
        s3, bucket_name, key_name = is_s3(self.file_path)
        if s3:
            self.writer = TSAccessS3(bucket_name, key_name, binary=False)
        else:
            self.writer = TSAccessFile(self.file_path, "a+")
        self.writer.write("[")

    def write_trace_event(self, tensor_name="", step_num=0, timestamp=None, duration=1, worker=0):
        args = {
            # "start_timestamp": timestamp - duration if timestamp else time.time() - duration,
            # "end_timestamp": timestamp if timestamp else time.time(),
            "step number": step_num,
        }
        # args["start_timestamp"] = int(args["start_timestamp"] * 100000)
        # args["end_timestamp"] = int(args["end_timestamp"] * 100000)
        duration_in_us = int(duration * 100000)
        event = Event(
            tensor_name=tensor_name, timestamp=timestamp, args=args, duration=duration_in_us, worker=worker
        )
        self.add_event(event)

    def add_event(self, event):
        if not self.writer:
            self._init_writer()
        self.event_payload.append(event)

    def flush(self):
        """Flushes the event string to file."""
        if not self.writer:
            raise ValueError(f"Cannot flush because self.writer={self.writer}")
        # if not self.event_payload:
        #     raise ValueError(
        #         f"Cannot write empty event={self.event_payload} to file {self.file_path}"
        #     )

        # TODO: kannanva: Add marker event indicating start of step?
        for event in self.event_payload:
            self.writer.write(event.to_json() + ",\n")
        self.writer.flush()
        self.event_payload = []

    def close(self):
        """Closes the record writer."""
        if self.writer is not None:
            if self.event_payload:
                self.flush()
                self.writer.close()
                self.writer = None


class Event:
    def __init__(self, tensor_name="", phase="X", worker="", args=None, timestamp=None, duration=1):
        self.tensor_name = tensor_name
        self.phase = phase
        self.worker = worker
        self.args = args
        self.timestamp = timestamp
        self.duration = duration

    def to_json(self):
        json_dict = {
            "name": self.tensor_name,
            "ph": self.phase,
            "ts": self.timestamp if self.timestamp else int(round(time.time() * 1000000)),
            "pid": self.worker, # TODO: kannanva: pid should be tensor index. For now, it is worker name.
            "dur": self.duration,
        }
        if self.args:
            json_dict["args"] = self.args

        return json.dumps(json_dict)
