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
from tornasole_numpy.tfevent.event_file_writer import EventFileWriter

class FileWriter():
    def __init__(self, logdir, trial=None, step=None, worker=None,
                    wtype='tfevent', 
                    max_queue=10, flush_secs=120, 
                    filename_suffix='', verbose=True):
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

        if wtype == 'tfevent':
            self._writer = EventFileWriter(logdir, max_queue, flush_secs, filename_suffix, verbose)
        else:
            assert False, 'Writer type not supported: {}'.format(wtype)
        

    def __enter__(self):
        """Make usable with "with" statement."""
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        """Make usable with "with" statement."""
        self.close()

    def write_tensor(self, tdata, tname, trial=None, step=None, worker=None):
        if trial is None:
            assert self.trial is not None
            trial = self.trial
        if step is None:
            assert self.step is not None
            step = self.step
        if worker is None:
            assert self.worker is not None
            worker = self.worker
        self._writer.write_tensor(tdata, tname, trial, step, worker)

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