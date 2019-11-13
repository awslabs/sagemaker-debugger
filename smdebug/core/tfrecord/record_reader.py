# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Standard Library
import struct

# First Party
from smdebug.core.access_layer.file import TSAccessFile
from smdebug.core.access_layer.s3 import TSAccessS3
from smdebug.core.tfrecord.record_writer import CHECKSUM_MAGIC_BYTES
from smdebug.core.utils import is_s3

# Local
from ._crc32c import crc32c


class RecordReader:
    """Read records in the following format for a single record event_str:
    uint64 len(event_str)
    uint32 masked crc of len(event_str)
    byte event_str
    uint32 masked crc of event_str
    The implementation is ported from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/io/record_writer.cc
    Here we simply define a byte string _dest to buffer the record to be written to files.
    The flush and close mechanism is totally controlled in this class.
    In TensorFlow, _dest is a object instance of ZlibOutputBuffer (C++) which has its own flush
    and close mechanism defined."""

    def __init__(self, path):
        s3, bucket_name, key_name = is_s3(path)
        try:
            if s3:
                self._reader = TSAccessS3(bucket_name, key_name)
            else:
                self._reader = TSAccessFile(path, "rb")
        except (OSError, IOError) as err:
            raise ValueError("failed to open {}: {}".format(path, str(err)))
        except:
            raise
        self._reader.ingest_all()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def has_data(self):
        has = self._reader.has_data()
        # print("HASDATA=", has)
        return has

    def read_record(self, check="minimal"):
        strlen_bytes = self._reader.read(8)
        strlen = struct.unpack("Q", strlen_bytes)[0]
        saved_len_crc = struct.unpack("I", self._reader.read(4))[0]

        if check in ["minimal", "full"]:
            computed_len_crc = masked_crc32c(strlen_bytes)
            assert saved_len_crc == computed_len_crc

        payload = self._reader.read(strlen)
        saved_payload_crc = struct.unpack("I", self._reader.read(4))[0]
        if check == "full":
            computed_payload_crc = masked_crc32c(payload)
            assert saved_payload_crc == computed_payload_crc
        elif check == "minimal":
            computed_payload_crc = masked_crc32c(CHECKSUM_MAGIC_BYTES)
            assert saved_payload_crc == computed_payload_crc
        else:
            # no check
            pass
        return payload

    def flush(self):
        assert False

    def close(self):
        """Closes the record reader."""
        if self._reader is not None:
            self._reader.close()
            self._reader = None


def masked_crc32c(data):
    """Copied from
    https://github.com/TeamHG-Memex/tensorboard_logger/blob/master/tensorboard_logger/tensorboard_logger.py"""
    x = u32(crc32c(data))  # pylint: disable=invalid-name
    return u32(((x >> 15) | u32(x << 17)) + 0xA282EAD8)


def u32(x):  # pylint: disable=invalid-name
    """Copied from
    https://github.com/TeamHG-Memex/tensorboard_logger/blob/master/tensorboard_logger/tensorboard_logger.py"""
    return x & 0xFFFFFFFF
