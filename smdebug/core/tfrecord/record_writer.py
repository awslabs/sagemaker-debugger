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

"""Writer for writing events to the event file."""

# Standard Library
import struct

# First Party
from smdebug.core.access_layer.file import TSAccessFile
from smdebug.core.access_layer.s3 import TSAccessS3
from smdebug.core.utils import is_s3

# Local
from ._crc32c import crc32c

CHECKSUM_MAGIC_BYTES = b"0x12345678"


class RecordWriter:
    """Write records in the following format for a single record event_str:
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

    def __init__(self, path, write_checksum):
        self.write_checksum = write_checksum
        s3, bucket_name, key_name = is_s3(path)
        try:
            if s3:
                self._writer = TSAccessS3(bucket_name, key_name)
            else:
                self._writer = TSAccessFile(path, "wb")
        except (OSError, IOError) as err:
            raise ValueError("failed to open {}: {}".format(path, str(err)))

    def __del__(self):
        self.close()

    def write_record(self, event_str):
        """Writes a serialized event to file."""
        header = struct.pack("Q", len(event_str))
        header += struct.pack("I", masked_crc32c(header))
        if self.write_checksum:
            footer = struct.pack("I", masked_crc32c(event_str))
        else:
            footer = struct.pack("I", masked_crc32c(CHECKSUM_MAGIC_BYTES))
        position_and_length_of_record = self._writer.write(header + event_str + footer)
        return position_and_length_of_record

    def flush(self):
        """Flushes the event string to file."""
        assert self._writer is not None
        self._writer.flush()

    def close(self):
        """Closes the record writer."""
        if self._writer is not None:
            self.flush()
            self._writer.close()
            self._writer = None


def masked_crc32c(data):
    """Copied from
    https://github.com/TeamHG-Memex/tensorboard_logger/blob/master/tensorboard_logger/tensorboard_logger.py"""
    x = u32(crc32c(data))  # pylint: disable=invalid-name
    return u32(((x >> 15) | u32(x << 17)) + 0xA282EAD8)


def u32(x):  # pylint: disable=invalid-name
    """Copied from
    https://github.com/TeamHG-Memex/tensorboard_logger/blob/master/tensorboard_logger/tensorboard_logger.py"""
    return x & 0xFFFFFFFF
