from tornasole.core.access_layer.file import TSAccessFile
from tornasole.core.access_layer.s3 import TSAccessS3
from tornasole.core.utils import is_s3


class IndexWriter(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.writer = None
        s3, bucket_name, key_name = is_s3(self.file_path)
        if s3:

            self.writer = TSAccessS3(bucket_name, key_name, binary=False)
        else:
            self.writer = TSAccessFile(self.file_path, 'a+')

    def __exit__(self):
        self.close()

    def add_index(self, tensorlocation):
        if self.writer is None:
            s3, bucket_name, key_name = is_s3(self.file_path)
            if s3:
                self.writer = TSAccessS3(bucket_name, key_name, binary=False)
            else:
                self.writer = TSAccessFile(self.file_path, 'a+')

        self.writer.write(tensorlocation.serialize() + "\n")

    def flush(self):
        """Flushes the event string to file."""
        assert self.writer is not None
        self.writer.flush()

    def close(self):
        """Closes the record writer."""
        if self.writer is not None:
            self.flush()
            self.writer.close()
            self.writer = None


class IndexArgs(object):
    def __init__(self, event, tensorname, mode, mode_step):
        self.event = event
        self.tensorname = tensorname
        self.mode = mode
        self.mode_step = mode_step

    def get_event(self):
        return self.event

    def get_tensorname(self):
        return self.tensorname

    def get_mode(self):
        return str(self.mode).split('.')[-1]

    def get_mode_step(self):
        return self.mode_step
