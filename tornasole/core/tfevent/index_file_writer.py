import json
from tornasole.core.access_layer.file import TSAccessFile
from tornasole.core.access_layer.s3 import TSAccessS3
from tornasole.core.utils import is_s3


class IndexWriter(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.writer = self._init_writer()
        self.index_payload = []
        self.index_meta = {}

    def __exit__(self):
        self.close()

    def _init_writer(self):
        s3, bucket_name, key_name = is_s3(self.file_path)
        if s3:
            writer = TSAccessS3(bucket_name, key_name, binary=False)
        else:
            writer = TSAccessFile(self.file_path, "a+")
        return writer

    def add_index(self, tensorlocation):
        if not self.index_meta:
            self.index_meta = {
                "mode": tensorlocation.mode,
                "mode_step": tensorlocation.mode_step,
                "event_file_name": tensorlocation.event_file_name,
            }
        self.index_payload.append(tensorlocation.to_dict())

    def flush(self):
        """Flushes the event string to file."""
        assert self.writer is not None
        index = Index(meta=self.index_meta, tensor_payload=self.index_payload)
        self.writer.write(index.to_json())
        self.writer.flush()
        self.index_meta = {}
        self.index_payload = {}

    def close(self):
        """Closes the record writer."""
        if self.writer is not None:
            self.flush()
            self.writer.close()
            self.writer = None


class Index:
    def __init__(self, meta=None, tensor_payload=None):
        self.meta = meta
        self.tensor_payload = tensor_payload

    def to_json(self):
        return json.dumps(self.__dict__)


class EventWithIndex(object):
    def __init__(self, event, tensorname, mode, mode_step):
        self.event = event
        self.tensorname = tensorname
        self.mode = mode
        self.mode_step = mode_step

    def get_mode(self):
        return str(self.mode).split(".")[-1]
