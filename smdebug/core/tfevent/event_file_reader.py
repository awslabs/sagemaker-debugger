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

"""Reads events from disk."""

# Third Party
import numpy as np

# First Party
import smdebug.core.tfevent.proto.types_pb2 as types_pb2
from smdebug.core.modes import MODE_PLUGIN_NAME, MODE_STEP_PLUGIN_NAME, ModeKeys
from smdebug.core.tfrecord.record_reader import RecordReader

# Local
from .proto.event_pb2 import Event


def as_dtype(t):
    _INTERN_TABLE = {
        types_pb2.DT_HALF: np.float16,
        types_pb2.DT_FLOAT: np.float32,
        types_pb2.DT_DOUBLE: np.float64,
        types_pb2.DT_INT32: np.int32,
        types_pb2.DT_INT64: np.int64,
        types_pb2.DT_STRING: np.str,
        types_pb2.DT_BOOL: np.bool,
    }
    return _INTERN_TABLE[t]


def get_tensor_data(tensor):
    shape = [d.size for d in tensor.tensor_shape.dim]
    # num_elements = np.prod(shape, dtype=np.int64)
    if tensor.dtype == 0 and tensor.string_val:
        assert len(shape) == 1
        res = []
        for i in range(shape[0]):
            r = tensor.string_val[i]
            r = r.decode("utf-8")
            res.append(r)
        return np.array(res)

    dtype = as_dtype(tensor.dtype)
    # dtype = tensor_dtype.as_numpy_dtype
    # dtype = np.float32
    # print("FOO=", tensor)
    # print("FOOTYPE=", tensor.dtype)
    if tensor.tensor_content:
        return np.frombuffer(tensor.tensor_content, dtype=dtype).copy().reshape(shape)
    elif dtype == np.int32:
        if len(tensor.int_val) > 0:
            return np.int32(tensor.int_val)
        else:
            return None
    elif dtype == np.int64:
        if len(tensor.int64_val) > 0:
            return np.int64(tensor.int64_val)
        else:
            return None
    elif dtype == np.float32:
        assert len(tensor.float_val) > 0, tensor
        return np.float32(tensor.float_val)
    elif dtype == np.bool:
        assert len(tensor.bool_val) > 0
        return np.bool(tensor.bool_val)
    else:
        raise Exception(f"Unknown type for Tensor={tensor}")


class EventsReader(object):
    """Writes `Event` protocol buffers to an event file. This class is ported from
    EventsReader defined in
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/events_writer.cc"""

    def __init__(self, filename):
        self._filename = filename
        self._tfrecord_reader = RecordReader(self._filename)

    def __exit__(self, exc_type, exc_value, traceback):
        self._tfrecord_reader.__exit__(exc_type, exc_value, traceback)

    def read_events(self, check="minimal"):
        while self._tfrecord_reader.has_data():
            rec = self._tfrecord_reader.read_record(check=check)
            event = Event()
            event.ParseFromString(rec)
            yield event


class EventFileReader:
    """This class is adapted from EventFileWriter in Tensorflow:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/summary/writer/event_file_writer.py
    Writes `Event` protocol buffers to an event file.
    The `EventFileWriter` class creates an event file in the specified directory,
    and asynchronously writes Event protocol buffers to the file. The Event file
    is encoded using the tfrecord format, which is similar to RecordIO.
    """

    def __init__(self, fname):
        """Creates a `EventFileWriter` and an event file to write to.
        On construction the summary writer creates a new event file in `logdir`.
        This event file will contain `Event` protocol buffers, which are written to
        disk via the add_event method.
        The other arguments to the constructor control the asynchronous writes to
        the event file:
        """
        self._filename = fname
        self._ev_reader = EventsReader(self._filename)

    def __exit__(self, exc_type, exc_value, traceback):
        self._ev_reader.__exit__(exc_type, exc_value, traceback)

    def _get_mode_modestep(self, step, plugin_data):
        mode_step = step
        mode = ModeKeys.GLOBAL
        for metadata in plugin_data:
            if metadata.plugin_name == MODE_STEP_PLUGIN_NAME:
                mode_step = int(metadata.content)
            if metadata.plugin_name == MODE_PLUGIN_NAME:
                mode = ModeKeys(int(metadata.content))
        return mode, mode_step

    def read_tensors(self, check="minimal"):
        for step, summ in self.read_summaries(check=check):
            for v in summ.value:
                val = v.WhichOneof("value")
                assert val == "tensor"
                tensor_name = v.tag
                # We have found the right tensor at the right step
                tensor_data = get_tensor_data(v.tensor)
                mode, mode_step = self._get_mode_modestep(step, v.metadata.plugin_data)
                yield (tensor_name, step, tensor_data, mode, mode_step)

    def read_summaries(self, check="minimal"):
        for ev in self.read_events(check=check):
            # graph gets bypassed here
            if ev.HasField("summary"):
                yield (ev.step, ev.summary)

    def read_events(self, check="minimal"):
        return self._ev_reader.read_events(check=check)
