# Standard Library

import json
from datetime import datetime

# First Party
from smdebug.core.access_layer.s3handler import ListRequest, ReadObjectRequest, S3Handler
from smdebug.profiler.tf_profiler_parser import SMTFProfilerEvents


class S3TraceTrial:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.directories = list()
        self.prefix = "framework/pevents"
        self.hr_to_event_map = dict()

    def get_start_timestamp_from_file(self, filename):
        return filename.split("_")[0]

    def get_event_files(self, hr_directory, start_time_seconds, end_time_seconds):
        files = []
        for event_file in self.hr_to_event_map[hr_directory]:
            event_file_stamp = int(self.get_start_timestamp_from_file(event_file.split("/")[-1]))
            if start_time_seconds <= event_file_stamp <= end_time_seconds:
                files.append(event_file)
        return files

    def get_events(self, start_time_seconds, end_time_seconds):
        start_dt = datetime.utcfromtimestamp(start_time_seconds)
        end_dt = datetime.utcfromtimestamp(end_time_seconds)
        start_dt_dir = int(start_dt.strftime("%y%m%d%H"))
        end_dt_dir = int(end_dt.strftime("%y%m%d%H"))

        # Get the event files for the range of time
        event_files = list()
        for hr in sorted(self.hr_to_event_map.keys()):
            if start_dt_dir <= hr <= end_dt_dir:
                event_files.extend(self.get_event_files(hr, start_time_seconds, end_time_seconds))

        # Download files and parse the events
        traceParser = SMTFProfilerEvents()
        for event_file in event_files:
            file_read_request = ReadObjectRequest(path=event_file)
            event_data = S3Handler.get_object(file_read_request)
            event_string = event_data.decode("utf-8")
            json_data = json.loads(event_string)
            traceParser.read_events_from_json_data(json_data)

        range_events = traceParser.get_events_within_time_range(
            start_time_seconds, end_time_seconds
        )
        return range_events

    """
    Create a map of timestamp to filename
    """

    def load_event_file_list(self):
        list_dir = ListRequest(Bucket=self.bucket_name, Prefix=self.prefix, StartAfter=self.prefix)

        event_files = [x for x in S3Handler.list_prefix(list_dir) if "json" in x]
        print(event_files)
        for event_file in event_files:
            dirs = event_file.split("/")
            dirs.pop()
            hr = int(dirs.pop())
            if hr not in self.hr_to_event_map:
                self.hr_to_event_map[hr] = list()
            self.hr_to_event_map[hr].append(f"s3://{self.bucket_name}/{event_file}")


def test_trace_trial():
    bucket_name = "tornasole-dev"
    tt = S3TraceTrial(bucket_name)
    tt.load_event_file_list()
    tt.get_events(1589930980, 1589930995)


if __name__ == "__main__":
    test_trace_trial()
