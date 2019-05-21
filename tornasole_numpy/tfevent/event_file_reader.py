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

import logging
import numpy as np
import os.path
import time


from .event_pb2 import Event
from .summary_pb2 import Summary, SummaryMetadata
from tornasole_numpy.tfrecord.record_reader import RecordReader

logging.basicConfig()

def get_tensor_data(tensor):
    shape = [d.size for d in tensor.tensor_shape.dim]
    # num_elements = np.prod(shape, dtype=np.int64)
    #tensor_dtype = dtypes.as_dtype(tensor.dtype)
    #dtype = tensor_dtype.as_numpy_dtype
    dtype = np.float32
    if tensor.tensor_content:
        return (np.frombuffer(tensor.tensor_content, dtype=dtype).copy().reshape(shape))
    elif dtype == np.int32:
        assert len(tensor.int_val) > 0
        return np.int32(tensor.int_val)
    elif dtype == np.float32:
        assert len(tensor.float_val) > 0
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

    def __del__(self):
        self._tfrecord_reader.__del__()

    def has_data(self):
        return self._tfrecord_reader.has_data()

    def read_event(self):
        if not self._tfrecord_reader.has_data():
            return None
        rec = self._tfrecord_reader.read_record()
        event = Event()
        event.ParseFromString(rec)
        return event
        


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

    def __del__(self):
        self._ev_reader.__del__()

    def read_tensors(self):
        summary = self.read_summary()
        if summary is None:
            return None
        res = []
        for v in summary.value:
            assert v.WhichOneof('value') == 'tensor'
            tensor_name = v.tag
            
            # We have found the right tensor at the right step
            tensor_data = get_tensor_data(v.tensor)
            res.append((tensor_name, tensor_data))
        return res

    def read_summary(self):
        event = self.read_event()
        if event is None:
            return None
        assert event.HasField('summary')
        return event.summary

    def read_event(self):
        return self._ev_reader.read_event()

