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
from .locations import EventFileLocation, IndexFileLocationUtils
from tornasole.core.tfevent.event_file_writer import EventFileWriter
from tornasole.core.tfevent.index_file_writer import IndexWriter

import socket

from .modes import ModeKeys


class FileWriter:
    def __init__(self, trial_dir, step=0, worker=None,
                 wtype='tensor',
                 max_queue=10, flush_secs=120,
                 verbose=False, write_checksum=False):
        """Creates a `FileWriter` and an  file.
        On construction the summary writer creates a new event file in `trial_dir`.
 
        Parameters
        ----------
            trial_dir : str
                Directory where event file will be written.
            step: int
                Global step number
            worker: str
                Worker name
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
            self.worker = socket.gethostname()

        index_file_path = IndexFileLocationUtils.get_index_key_for_step(
                self.trial_dir, self.step, self.worker)
        self.index_writer = IndexWriter(index_file_path)

        if wtype == 'tensor':
            el = EventFileLocation(step_num=self.step, worker_name=self.worker)
            event_file_path = el.get_location(trial_dir=self.trial_dir)
        else:
            assert False, 'Writer type not supported: {}'.format(wtype)

        self._writer = EventFileWriter(
                path=event_file_path, index_writer=self.index_writer,
                max_queue=max_queue, flush_secs=flush_secs,
                verbose=verbose, write_checksum=write_checksum
        )

    def __enter__(self):
        """Make usable with "with" statement."""
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        """Make usable with "with" statement."""
        self.close()

    def write_tensor(self, tdata, tname, write_index=True,
                     mode=ModeKeys.GLOBAL, mode_step=None):
        mode, mode_step = self._check_mode_step(mode, mode_step, self.step)
        self._writer.write_tensor(tdata, tname, write_index,
                                  global_step=self.step,
                                  mode=mode, mode_step=mode_step)

    def write_summary(self, summ, tname, global_step, write_index=True,
                      mode=ModeKeys.GLOBAL, mode_step=None):
        mode, mode_step = self._check_mode_step(mode, mode_step, global_step)
        if write_index:
            self._writer.write_summary_with_index(
                    summ, global_step, tname, mode, mode_step)
        else:
            self._writer.write_summary(summ, global_step)

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

