# Standard Library

import json
import os
from datetime import datetime

# First Party
from smdebug.core.access_layer.s3handler import ListRequest, ReadObjectRequest, S3Handler
from smdebug.core.logger import get_logger
from smdebug.core.utils import list_files_in_directory
from smdebug.profiler.profiler_constants import (
    DEFAULT_PREFIX,
    HOROVODTIMELINE_PREFIX,
    MODELTIMELINE_PREFIX,
    PYTHONTIMELINE_PREFIX,
    TENSORBOARDTIMELINE_PREFIX,
)
from smdebug.profiler.tf_profiler_parser import (
    HorovodProfilerEvents,
    SMProfilerEvents,
    TensorboardProfilerEvents,
)
from smdebug.profiler.utils import TimeUnits, convert_utc_timestamp_to_microseconds


class MetricsReader:
    def __init__(self):
        self.prefix = DEFAULT_PREFIX
        self.hr_to_event_map = dict()
        self.logger = get_logger("smdebug-profiler")
        self._SMEventsParser = SMProfilerEvents()
        self._TBEventsParser = TensorboardProfilerEvents()
        self._HorovordEventsParser = HorovodProfilerEvents()
        # This is a set of parsed event files. The entry is made into this file only if the complete file is read.
        self._parsed_files = set()

    def _get_start_timestamp_from_file(self, filename):
        return filename.split("_")[0]

    """
    Get the tracefiles in the given hour directory that would contain the events corresponding to start and ent
    timestamps in microseconds
    """

    def _get_event_files(self, hr_directory, start_time_microseconds, end_time_microseconds):
        files = []
        for event_file in self.hr_to_event_map[hr_directory]:
            event_file_stamp = int(self._get_start_timestamp_from_file(event_file.split("/")[-1]))
            if start_time_microseconds <= event_file_stamp <= end_time_microseconds:
                files.append(event_file)
        return files

    def _get_trace_files_in_the_range(self, start_time_microseconds, end_time_microseconds):
        start_dt = datetime.utcfromtimestamp(start_time_microseconds / 1000000)
        end_dt = datetime.utcfromtimestamp(end_time_microseconds / 1000000)
        start_dt_dir = int(start_dt.strftime("%y%m%d%H"))
        end_dt_dir = int(end_dt.strftime("%y%m%d%H"))

        # Get the event files for the range of time
        event_files = list()
        for hr in sorted(self.hr_to_event_map.keys()):
            if start_dt_dir <= hr <= end_dt_dir:
                event_files.extend(
                    self._get_event_files(hr, start_time_microseconds, end_time_microseconds)
                )
        return event_files

    """
    The function returns the right event parser for given file name
    1. For Filename containing 'pythontimeline.json'  -> SMEventsParser
    2. For Filename containing 'model_timeline.json'  -> SMEventsParser
    3. For Filename containing 'tensorboard' (TBD) -> TensorboardProfilerEvents
    4. For Filename containing 'Horovod' (TBD) -> 'HorovodProfilerEvents
    """

    def _get_event_parser(self, filename):
        if PYTHONTIMELINE_PREFIX in filename:
            return self._SMEventsParser
        if MODELTIMELINE_PREFIX in filename:
            return self._SMEventsParser
        if TENSORBOARDTIMELINE_PREFIX in filename:
            return self._TBEventsParser
        if HorovodProfilerEvents in filename:
            return self._HorovordEventsParser

    """
    Create a map of timestamp to filename
    """

    def load_event_file_list(self):
        pass

    """
    The function returns the events that have recorded within the given time range.
    This is currently non-blocking call. The function will download (or parse) the tracefiles that are available
    for the given time range. It is possible that events are recorded during training but are not available for
    download.
    TODO: Implement caching to avoid repeat download
    TODO: Implement blocking call to wait for files to be available for download.
    """

    def get_events(self, start_time, end_time, unit=TimeUnits.MICROSECONDS):
        self.load_event_file_list()
        start_time = convert_utc_timestamp_to_microseconds(start_time, unit)
        end_time = convert_utc_timestamp_to_microseconds(end_time, unit)
        event_files = self._get_trace_files_in_the_range(start_time, end_time)

        # Download files and parse the events
        self.parse_event_files(event_files)

        """
        We might have recorded events from different sources within this timerange.
        we will get the events from the relevant event parsers and merge them before returning.
        """
        result = []
        for eventParser in [self._SMEventsParser, self._TBEventsParser, self._HorovordEventsParser]:
            range_events = eventParser.get_events_within_time_range(
                start_time, end_time, unit=TimeUnits.MICROSECONDS
            )
            result.extend(range_events)

        return result

    def parse_event_files(self, event_files):
        pass


class LocalMetricsReader(MetricsReader):
    def __init__(self, trace_root_folder):
        self.trace_root_folder = trace_root_folder
        super().__init__()

    """
    Create a map of timestamp to filename
    """

    def load_event_file_list(self):
        path = os.path.expanduser(self.trace_root_folder)
        event_dir = os.path.join(path, DEFAULT_PREFIX, "")
        event_regex = r"(.+)\.(json|csv)$"
        event_files = list_files_in_directory(event_dir, file_regex=event_regex)
        for event_file in event_files:
            dirs = event_file.split("/")
            dirs.pop()
            hr = int(dirs.pop())
            if hr not in self.hr_to_event_map:
                self.hr_to_event_map[hr] = set()
            self.hr_to_event_map[hr].add(event_file)

    """
    The function opens and reads the event files if they are not already parsed.
    For local metrics reader, we are currently assuming that the downloaded event file is a complete file.
    """

    def parse_event_files(self, event_files):
        for event_file in event_files:
            if event_file not in self._parsed_files:
                self._get_event_parser(event_file).read_events_from_file(event_file)
                self._parsed_files.add(event_file)


class S3MetricsReader(MetricsReader):
    def __init__(self, bucket_name):
        super().__init__()
        self.bucket_name = bucket_name

    """
    The function opens and reads the event files if they are not already parsed.
    For S3 metrics reader, we are currently downloading the entire event file. Currently, we will add this file to a
    _parsed_file set assuming that the file is complete and it won't be updated on S3 later. However it is
    possible that the event file has not reached it's maximum size and will be updated later.
    TODO: Check the downloaded size and add the file to _parsed_files only if the file with maximum size is
    downloaded and parsed.
    """

    def parse_event_files(self, event_files):
        for event_file in event_files:
            if event_file not in self._parsed_files:
                file_read_request = ReadObjectRequest(path=event_file)
                event_data = S3Handler.get_object(file_read_request)
                event_string = event_data.decode("utf-8")
                json_data = json.loads(event_string)
                self._get_event_parser(event_file).read_events_from_json_data(json_data)
                self._parsed_files.add(event_file)

    """
    Create a map of timestamp to filename
    """

    def load_event_file_list(self):
        list_dir = ListRequest(Bucket=self.bucket_name, Prefix=self.prefix, StartAfter=self.prefix)

        event_files = [x for x in S3Handler.list_prefix(list_dir) if "json" in x]
        for event_file in event_files:
            dirs = event_file.split("/")
            dirs.pop()
            hr = int(dirs.pop())
            if hr not in self.hr_to_event_map:
                self.hr_to_event_map[hr] = set()
            self.hr_to_event_map[hr].add(f"s3://{self.bucket_name}/{event_file}")
