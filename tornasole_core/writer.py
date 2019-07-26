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

import time
from tornasole_core.tfevent.event_file_writer import EventFileWriter
import socket
from tornasole_core.modes import ModeKeys

class FileWriter():
    def __init__(self, logdir, trial, step, worker=None, rank=0, part=0,
                 wtype='tfevent',
                 max_queue=10, flush_secs=120,
                 filename_suffix='', verbose=False, write_checksum=False):
        """Creates a `FileWriter` and an  file.
        On construction the summary writer creates a new event file in `logdir`.
 
        Parameters
        ----------
            logdir : str
                Directory where event file will be written.
            max_queue : int
                Size of the queue for pending events and summaries.
            flush_secs: Number
                How often, in seconds, to flush the pending events and summaries to disk.
            filename_suffix : str
                Every event file's name is suffixed with `filename_suffix` if provided.
            verbose : bool
                Determines whether to print logging messages.
        """
        self.trial = trial
        self.step = step
        self.worker = worker
        if worker is None:
            self.worker = socket.gethostname()

        if wtype == 'tfevent':
            self._writer = EventFileWriter(logdir=logdir, trial=self.trial, worker=self.worker,
                                           step=self.step, part=part,
                                           max_queue=max_queue, flush_secs=flush_secs,
                                           filename_suffix=filename_suffix,
                                           verbose=verbose, write_checksum=write_checksum)

        else:
            assert False, 'Writer type not supported: {}'.format(wtype)

    def __enter__(self):
        """Make usable with "with" statement."""
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        """Make usable with "with" statement."""
        self.close()

    def write_tensor(self, tdata, tname, write_index=True,
                     mode=ModeKeys.GLOBAL, mode_step=None):
        self._writer.write_tensor(tdata, tname, write_index,
                                  mode, mode_step)

    def flush(self):
        """Flushes the event file to disk.
        Call this method to make sure that all pending events have been written to disk.
        """
        self._writer.flush()

    def close(self):
        """Flushes the event file to disk and close the file.
        Call this method when you do not need the summary writer anymore.
        """
        self._writer.close()

    def reopen(self):
        """Reopens the EventFileWriter.
        Can be called after `close()` to add more events in the same directory.
        The events will go into a new events file. Does nothing if the EventFileWriter
        was not closed.
        """
        self._writer.reopen()

    def name(self):
        return self._writer.name()
