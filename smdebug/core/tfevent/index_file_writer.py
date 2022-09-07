# Standard Library
import json

# First Party
from smdebug.core.access_layer.file import TSAccessFile
from smdebug.core.access_layer.s3 import TSAccessS3
from smdebug.core.utils import is_s3
from smdebug.exceptions import SMDebugValueError


class IndexWriter(object):
    def __init__(self, file_path):
        """ Writer is initialized upon adding the first index. """
        self.file_path = file_path
        self.index_payload = []
        self.index_meta = {}
        self.shape_payload = []
        self.writer = None

    def __exit__(self):
        self.close()

    def _init_writer(self):
        s3, bucket_name, key_name = is_s3(self.file_path)
        if s3:
            self.writer = TSAccessS3(bucket_name, key_name, binary=False)
        else:
            self.writer = TSAccessFile(self.file_path, "a+")

    def add_index(self, tensorlocation):
        if not self.writer:
            self._init_writer()
        if not self.index_meta or not "event_file_name" in self.index_meta:
            self.index_meta = {
                "mode": tensorlocation.mode,
                "mode_step": tensorlocation.mode_step,
                "event_file_name": tensorlocation.event_file_name,
            }
        self.index_payload.append(tensorlocation.to_dict())

    def add_shape(self, tensorshape):
        if not self.writer:
            self._init_writer()
        if not self.index_meta:
            self.index_meta = {"mode": tensorshape.mode, "mode_step": tensorshape.mode_step}
        self.shape_payload.append(tensorshape.to_dict())

    def flush(self):
        """Flushes the event string to file."""
        if not self.writer:
            raise SMDebugValueError(f"Cannot flush because self.writer={self.writer}")
        if not self.index_meta:
            raise SMDebugValueError(
                f"Cannot write empty index_meta={self.index_meta} to file {self.file_path}"
            )
        if not self.index_payload and not self.shape_payload:
            raise SMDebugValueError(
                f"Cannot write empty payload: index_payload={self.index_payload}, shape_payload={self.shape_payload} to file {self.file_path}"
            )

        index = Index(
            meta=self.index_meta,
            tensor_payload=self.index_payload,
            shape_payload=self.shape_payload,
        )
        self.writer.write(index.to_json())
        self.writer.flush()
        self.index_meta = {}
        self.index_payload = []
        self.shape_payload = []

    def close(self):
        """Closes the record writer."""
        if self.writer is not None:
            if self.index_meta and (self.index_payload or self.shape_payload):
                self.flush()
                self.writer.close()
                self.writer = None


class Index:
    def __init__(self, meta=None, tensor_payload=None, shape_payload=None):
        self.meta = meta
        self.tensor_payload = tensor_payload
        self.shape_payload = shape_payload

    def to_json(self):
        return json.dumps(
            {
                "meta": self.meta,
                "tensor_payload": self.tensor_payload,
                "shape_payload": self.shape_payload,
            }
        )


class EventWithIndex(object):
    def __init__(self, event, tensorname, mode, mode_step):
        self.event = event
        self.tensorname = tensorname
        self.mode = mode
        self.mode_step = mode_step

    def get_mode(self):
        return str(self.mode).split(".")[-1]
