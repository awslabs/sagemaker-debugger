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
from smdebug.core.tfevent.event_file_reader import EventFileReader, get_tensor_data

# Local
from .utils import match_inc


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

    def read_events(self, check="minimal", regex_list=None):
        """

        Args:
            check: default value = 'minimal'
            regex_list: default value = None
            When check is 'minimal' the crc checksum of the read payload is compared with CHECKSUM_MAGIC_BYTES.
        Returns: List of scalar events. Each scalar event is a dictionary containing following keys:
        scalar_event{
        "timestamp"
        "step"
        "name"
        "value"
        }

        """
        if check.lower() == "minimal":
            check = "minimal"
        tf_events = self._reader.read_events(check=check)
        scalar_events = list()
        for tf_event in tf_events:
            for v in tf_event.summary.value:
                event_name = v.tag
                if regex_list is None or match_inc(event_name, regex_list):
                    scalar_events.append(
                        {
                            "timestamp": tf_event.wall_time,
                            "step": tf_event.step,
                            "name": event_name,
                            "value": get_tensor_data(v.tensor),
                        }
                    )
        return scalar_events
