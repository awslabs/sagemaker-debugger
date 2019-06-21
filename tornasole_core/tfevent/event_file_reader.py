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

import tornasole_core.tfevent.types_pb2 as types_pb2
import logging
import numpy as np
import os.path
import time


from .event_pb2 import Event
from .summary_pb2 import Summary, SummaryMetadata

from tornasole_core.tfrecord.record_reader import RecordReader

#todo: remove this logger perhaps
logging.basicConfig()

def as_dtype(t):
    _INTERN_TABLE = {
        types_pb2.DT_HALF: np.float16,
        types_pb2.DT_FLOAT: np.float32,
        types_pb2.DT_DOUBLE: np.float64,
        types_pb2.DT_INT32: np.int32,
        types_pb2.DT_INT64: np.int64,
        types_pb2.DT_STRING: np.str,
        types_pb2.DT_BOOL: np.bool
    }
    return _INTERN_TABLE[t]



def get_tensor_data(tensor):
    shape = [d.size for d in tensor.tensor_shape.dim]
    # num_elements = np.prod(shape, dtype=np.int64)
    if tensor.dtype == 0 and tensor.string_val:
        assert len(shape)==1
        res = []
        for i in range(shape[0]):
            r = tensor.string_val[i]
            r = r.decode('utf-8')
            res.append(r)
        return res

    dtype = as_dtype(tensor.dtype)
    #dtype = tensor_dtype.as_numpy_dtype
    #dtype = np.float32
    #print("FOO=", tensor)
    #print("FOOTYPE=", tensor.dtype)
    if tensor.tensor_content:
        return (np.frombuffer(tensor.tensor_content, dtype=dtype).copy().reshape(shape))
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
        raise Exception(f'Unknown type for Tensor={tensor}')


class EventsReader(object):
    """Writes `Event` protocol buffers to an event file. This class is ported from
    EventsReader defined in
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/events_writer.cc"""
    def __init__(self, filename, verbose=True):
        """
        Events files have a name of the form
        '/file/path/events.out.tfevents.[timestamp].[hostname][file_suffix]'
        """
        self._filename = filename
        self._tfrecord_reader = RecordReader(self._filename)
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

    def __exit__(self, exc_type, exc_value, traceback):
        self._tfrecord_reader.__exit__(exc_type, exc_value, traceback)

    #def has_data(self):
    #    return self._tfrecord_reader.has_data()

    def read_events(self, check=True):
        while self._tfrecord_reader.has_data():
            rec = self._tfrecord_reader.read_record(check=check)
            event = Event()
            event.ParseFromString(rec)
            yield event
        


class EventFileReader():
    """This class is adapted from EventFileWriter in Tensorflow:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/summary/writer/event_file_writer.py
    Writes `Event` protocol buffers to an event file.
    The `EventFileWriter` class creates an event file in the specified directory,
    and asynchronously writes Event protocol buffers to the file. The Event file
    is encoded using the tfrecord format, which is similar to RecordIO.
    """

    def __init__(self, fname, verbose=True):
        """Creates a `EventFileWriter` and an event file to write to.
        On construction the summary writer creates a new event file in `logdir`.
        This event file will contain `Event` protocol buffers, which are written to
        disk via the add_event method.
        The other arguments to the constructor control the asynchronous writes to
        the event file:
        """
        self._filename = fname
        self._ev_reader = EventsReader(self._filename, verbose=verbose)

    def __exit__(self,exc_type, exc_value, traceback):
        self._ev_reader.__exit__(exc_type, exc_value, traceback)

    def read_tensors(self, read_data=False, check=False):
        for (step,summ) in self.read_summaries(check=check):
            for v in summ.value:
                assert v.WhichOneof('value') == 'tensor'
                tensor_name = v.tag
                # We have found the right tensor at the right step
                if read_data:
                    tensor_data = get_tensor_data(v.tensor)
                else:
                    tensor_data = None
                yield (tensor_name, step, tensor_data)

    def read_summaries(self, check=True):
        for ev in self.read_events(check=check):
            #assert ev.HasField('step')
            if not ev.HasField('summary'):
                continue
            assert ev.HasField('summary')
            yield (ev.step, ev.summary)

    def read_events(self,check=True):
        return self._ev_reader.read_events(check=check)

