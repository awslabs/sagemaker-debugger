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

"""APIs for logging data in the event file."""
# First Party
from smdebug.core.modes import MODE_PLUGIN_NAME, MODE_STEP_PLUGIN_NAME
from smdebug.core.tfevent.event_file_writer import EventFileWriter
from smdebug.core.tfevent.index_file_writer import IndexWriter
from smdebug.core.tfevent.proto.event_pb2 import Event, TaggedRunMetadata
from smdebug.core.tfevent.proto.summary_pb2 import Summary, SummaryMetadata
from smdebug.core.tfevent.summary import (
    _get_default_bins,
    histogram_summary,
    make_numpy_array,
    scalar_summary,
)
from smdebug.core.tfevent.util import make_tensor_proto

# Local
from .locations import IndexFileLocationUtils, TensorboardFileLocation, TensorFileLocation
from .logger import get_logger
from .modes import ModeKeys

logger = get_logger()


class FileWriter:
    def __init__(
        self,
        trial_dir,
        worker,
        step=0,
        wtype="events",
        mode=ModeKeys.GLOBAL,
        max_queue=10,
        flush_secs=120,
        verbose=False,
        write_checksum=False,
    ):
        """Creates a `FileWriter` and an  file.
        On construction the summary writer creates a new event file in `trial_dir`.

        Parameters
        ----------
            trial_dir : str
                Directory where event file will be written.
            worker: str
                Worker name
            step: int
                Global step number
            wtype: str
                Used to denote what sort of data we are writing
            max_queue : int
                Size of the queue for pending events and summaries.
            flush_secs: Number
                How often, in seconds, to flush the pending events and summaries to disk.
            verbose : bool
                Determines whether to print logging messages.
        """
        self.trial_dir = trial_dir
        self.step = step
        self.worker = worker
        if worker is None:
            assert False, "Worker should not be none. Check worker name initialization"
        self.mode = mode
        if wtype == "events":
            el = TensorFileLocation(step_num=self.step, worker_name=self.worker)
            event_file_path = el.get_file_location(trial_dir=self.trial_dir)
            index_file_path = IndexFileLocationUtils.get_index_key_for_step(
                self.trial_dir, self.step, self.worker
            )
            self.index_writer = IndexWriter(index_file_path)
        elif wtype == "tensorboard":
            el = TensorboardFileLocation(
                step_num=self.step, worker_name=self.worker, mode=self.mode
            )
            event_file_path = el.get_file_location(base_dir=self.trial_dir)
            self.index_writer = None
        else:
            assert False, "Writer type not supported: {}".format(wtype)

        self._writer = EventFileWriter(
            path=event_file_path,
            index_writer=self.index_writer,
            max_queue=max_queue,
            flush_secs=flush_secs,
            verbose=verbose,
            write_checksum=write_checksum,
        )
        self._default_bins = _get_default_bins()

    def __enter__(self):
        """Make usable with "with" statement."""
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        """Make usable with "with" statement."""
        self.close()

    @staticmethod
    def _get_metadata(mode, mode_step):
        sm2 = SummaryMetadata.PluginData(plugin_name=MODE_STEP_PLUGIN_NAME, content=str(mode_step))
        sm3 = SummaryMetadata.PluginData(plugin_name=MODE_PLUGIN_NAME, content=str(mode.value))
        plugin_data = [sm2, sm3]
        smd = SummaryMetadata(plugin_data=plugin_data)
        return smd

    def write_tensor(
        self, tdata, tname, write_index=True, mode=ModeKeys.GLOBAL, mode_step=None, timestamp=None
    ):
        mode, mode_step = self._check_mode_step(mode, mode_step, self.step)
        smd = self._get_metadata(mode, mode_step)
        value = make_numpy_array(tdata)
        tag = tname
        tensor_proto = make_tensor_proto(nparray_data=value, tag=tag)
        s = Summary(value=[Summary.Value(tag=tag, metadata=smd, tensor=tensor_proto)])
        if write_index:
            self._writer.write_summary_with_index(
                s, self.step, tname, mode, mode_step, timestamp=timestamp
            )
        else:
            self._writer.write_summary(s, self.step, timestamp)

    def write_graph(self, graph):
        self._writer.write_graph(graph)

    def write_pytorch_graph(self, graph_profile):
        # https://github.com/pytorch/pytorch/blob/c749be9e9f8dd3db8b3582e93f917bd47e8e9e20/torch/utils/tensorboard/writer.py # L99
        # graph_profile = pytorch_graph.graph(self.model)
        graph = graph_profile[0]
        stepstats = graph_profile[1]
        event = Event(graph_def=graph.SerializeToString())
        self._writer.write_event(event)
        trm = TaggedRunMetadata(tag="step1", run_metadata=stepstats.SerializeToString())
        event = Event(tagged_run_metadata=trm)
        self._writer.write_event(event)

    def write_summary(self, summ, global_step, timestamp: float = None):
        self._writer.write_summary(summ, global_step, timestamp=timestamp)

    def write_histogram_summary(self, tdata, tname, global_step, bins="default"):
        """Add histogram data to the event file.
        Parameters
        ----------
        tname : str
            Name for the `values`.
        tdata: `numpy.ndarray`
            Values for building histogram.
        global_step : int
            Global step value to record.
        bins : int or sequence of scalars or str
            If `bins` is an int, it defines the number equal-width bins in the range
            `(values.min(), values.max())`.
            If `bins` is a sequence, it defines the bin edges, including the rightmost edge,
            allowing for non-uniform bin width.
            If `bins` is a str equal to 'default', it will use the bin distribution
            defined in TensorFlow for building histogram.
            Ref: https://www.tensorflow.org/programmers_guide/tensorboard_histograms
            The rest of supported strings for `bins` are 'auto', 'fd', 'doane', 'scott',
            'rice', 'sturges', and 'sqrt'. etc. See the documentation of `numpy.histogram`
            for detailed definitions of those strings.
            https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
        """
        if bins == "default":
            bins = self._default_bins
        try:
            s = histogram_summary(tname, tdata, bins)
            self._writer.write_summary(s, global_step)
        except ValueError as e:
            logger.warning(f"Unable to write histogram {tname} at {global_step}: {e}")

    def write_scalar_summary(self, name, value, global_step, timestamp: float = None):
        s = scalar_summary(name, value)
        self._writer.write_summary(s, global_step, timestamp=timestamp)

    def flush(self):
        """Flushes the event file to disk.
        Call this method to make sure that all pending events have been written to disk.
        """
        self._writer.flush()
        # don't flush index writer as we only want to flush on close

    def close(self):
        """Flushes the event file to disk and close the file.
        Call this method when you do not need the summary writer anymore.
        """
        self._writer.close()
        if self.index_writer is not None:
            self.index_writer.close()

    def name(self):
        return self._writer.name()

    @staticmethod
    def _check_mode_step(mode, mode_step, global_step):
        if mode_step is None:
            mode_step = global_step
        if mode is None:
            mode = ModeKeys.GLOBAL
        if not isinstance(mode, ModeKeys):
            mode_keys = ["ModeKeys." + x.name for x in ModeKeys]
            ex_str = "mode can be one of " + ", ".join(mode_keys)
            raise ValueError(ex_str)
        return mode, mode_step
