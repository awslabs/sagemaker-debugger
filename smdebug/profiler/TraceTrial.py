# Standard Library

# from smdebug.core.collection_manager import CollectionManager
# from smdebug.core.index_reader import S3IndexReader
# from smdebug.core.s3_utils import list_s3_objects
# from smdebug.core.utils import get_path_to_collections
from datetime import datetime

# First Party
from smdebug.core.access_layer.s3handler import ListRequest, S3Handler


class TraceTrial:
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
        last_timestamp_read = start_time_seconds

        # Get the event files
        event_files = list()
        for hr in sorted(self.hr_to_event_map.keys()):
            if start_dt_dir <= hr <= end_dt_dir:
                event_files.append(self.get_event_files(hr, start_time_seconds, end_time_seconds))
        print(event_files)

    """
    Create a map of timestamp to filename
    """

    def list_files_in_directory(self, directory_name):
        pass

    def download_file(self, filename):
        pass

    def load_event_file_list(self):
        list_dir = ListRequest(Bucket=self.bucket_name, Prefix=self.prefix, StartAfter=self.prefix)

        event_files = [x for x in S3Handler.list_prefix(list_dir) if "json" in x]
        print(event_files)
        for event_file in event_files:
            dirs = event_file.split("/")
            file = dirs.pop()
            hr = int(dirs.pop())
            if hr not in self.hr_to_event_map:
                self.hr_to_event_map[hr] = list()
            self.hr_to_event_map[hr].append(f"s3://{self.bucket_name}{event_file}")

        print(self.hr_to_event_map)


def test_trace_trial():
    bucket_name = "tornasole-dev"
    tt = TraceTrial(bucket_name)
    tt.load_event_file_list()
    tt.get_events(1589837121, 1589843074)


if __name__ == "__main__":
    test_trace_trial()
