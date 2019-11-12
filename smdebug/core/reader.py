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
from smdebug.core.tfevent.event_file_reader import EventFileReader


class FileReader:
    def __init__(self, fname, wtype="tfevent"):
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
        """
        if wtype == "tfevent":
            self._reader = EventFileReader(fname=fname)
        else:
            assert False

    def __enter__(self):
        """Make usable with "with" statement."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Make usable with "with" statement."""
        self._reader.__exit__(exc_type, exc_value, traceback)

    def read_tensors(self, check="minimal"):
        if check is True:
            check = "minimal"
        return self._reader.read_tensors(check=check)
