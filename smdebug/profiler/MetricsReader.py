# Standard Library

import json
import os
from datetime import datetime

# First Party
from smdebug.core.access_layer.s3handler import ListRequest, ReadObjectRequest, S3Handler
from smdebug.core.logger import get_logger
from smdebug.core.utils import list_files_in_directory
from smdebug.profiler.profiler_constants import DEFAULT_PREFIX
from smdebug.profiler.tf_profiler_parser import SMTFProfilerEvents


class MetricsReader:
    def __init__(self):
        self.prefix = DEFAULT_PREFIX
        self.hr_to_event_map = dict()
        self.logger = get_logger("smdebug-profiler")

    def get_start_timestamp_from_file(self, filename):
        return filename.split("_")[0]

    """
    Get the tracefiles in the given hour directory that would contain the events corresponding to start and ent
    timestamps
    """

    def get_event_files(self, hr_directory, start_time_seconds, end_time_seconds):
        files = []
        for event_file in self.hr_to_event_map[hr_directory]:
            event_file_stamp = int(self.get_start_timestamp_from_file(event_file.split("/")[-1]))
            if start_time_seconds <= event_file_stamp <= end_time_seconds:
                files.append(event_file)
        return files

    """
    Create a map of timestamp to filename
    """

    def load_event_file_list(self):
        pass

    def get_events(self, start_time_seconds, end_time_seconds):
        self.load_event_file_list()
        event_files = self.get_trace_files_in_the_range(start_time_seconds, end_time_seconds)

        # Download files and parse the events
        traceParser = self.parse_event_files(event_files)

        range_events = traceParser.get_events_within_time_range(
            start_time_seconds, end_time_seconds
        )
        return range_events

    def parse_event_files(self, event_files):
        pass

    def get_trace_files_in_the_range(self, start_time_seconds, end_time_seconds):
        start_dt = datetime.utcfromtimestamp(start_time_seconds)
        end_dt = datetime.utcfromtimestamp(end_time_seconds)
        start_dt_dir = int(start_dt.strftime("%y%m%d%H"))
        end_dt_dir = int(end_dt.strftime("%y%m%d%H"))

        # Get the event files for the range of time
        event_files = list()
        for hr in sorted(self.hr_to_event_map.keys()):
            if start_dt_dir <= hr <= end_dt_dir:
                event_files.extend(self.get_event_files(hr, start_time_seconds, end_time_seconds))
        return event_files


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
                self.hr_to_event_map[hr] = list()
            self.hr_to_event_map[hr].append(event_file)

    def parse_event_files(self, event_files):
        traceParser = SMTFProfilerEvents()
        for event_file in event_files:
            traceParser.update_events_from_file(event_file)
        return traceParser


class S3MetricsReader(MetricsReader):
    def __init__(self, bucket_name):
        super().__init__()
        self.bucket_name = bucket_name

    def parse_event_files(self, event_files):
        traceParser = SMTFProfilerEvents()
        for event_file in event_files:
            file_read_request = ReadObjectRequest(path=event_file)
            event_data = S3Handler.get_object(file_read_request)
            event_string = event_data.decode("utf-8")
            json_data = json.loads(event_string)
            traceParser.read_events_from_json_data(json_data)
        return traceParser

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
                self.hr_to_event_map[hr] = list()
            self.hr_to_event_map[hr].append(f"s3://{self.bucket_name}/{event_file}")
