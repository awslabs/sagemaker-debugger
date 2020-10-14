# Standard Library
import time

# First Party
from smdebug.core.access_layer.utils import has_training_ended
from smdebug.core.logger import get_logger
from smdebug.exceptions import NoMoreProfilerData
from smdebug.profiler.algorithm_metrics_reader import (
    LocalAlgorithmMetricsReader,
    S3AlgorithmMetricsReader,
)
from smdebug.profiler.system_metrics_reader import LocalSystemMetricsReader, S3SystemMetricsReader


class ProfilerTrial:
    def __init__(self, name, dirname):
        self.name = name
        self.path = dirname
        self.logger = get_logger()
        self.first_timestamp = 0
        self.get_first_timestamp()

    def job_finished(self):
        if has_training_ended(self.path + "/system") or has_training_ended(
            self.path + "/framework"
        ):
            return True
        return False

    def get_first_timestamp(self):
        while self.first_timestamp == 0:
            if self.path.startswith("s3"):
                self.system_metrics_reader = S3SystemMetricsReader(self.path)
                self.framework_metrics_reader = S3AlgorithmMetricsReader(self.path)
            else:
                self.system_metrics_reader = LocalSystemMetricsReader(self.path)
                self.framework_metrics_reader = LocalAlgorithmMetricsReader(self.path)
            if self.system_metrics_reader.get_timestamp_of_first_available_file() != 0:
                self.first_timestamp = (
                    self.system_metrics_reader.get_timestamp_of_first_available_file()
                )
            if self.framework_metrics_reader.get_timestamp_of_first_available_file() != 0:
                if (
                    self.framework_metrics_reader.get_timestamp_of_first_available_file()
                    < self.first_timestamp
                ):
                    self.first_timestamp = (
                        self.framework_metrics_reader.get_timestamp_of_first_available_file()
                    )
            self.logger.info("Waiting for profiler data.")
            time.sleep(10)

    def get_latest_timestamp(self):
        latest_timestamp = 0
        self.system_metrics_reader.refresh_event_file_list()
        self.framework_metrics_reader.refresh_event_file_list()
        if self.system_metrics_reader.get_timestamp_of_latest_available_file() != 0:
            latest_timestamp = self.system_metrics_reader.get_timestamp_of_latest_available_file()
        if self.framework_metrics_reader.get_timestamp_of_latest_available_file() != 0:
            if (
                self.framework_metrics_reader.get_timestamp_of_latest_available_file()
                < latest_timestamp
            ):
                latest_timestamp = (
                    self.framework_metrics_reader.get_timestamp_of_latest_available_file()
                )
        return latest_timestamp

    def wait_for_data(self, end_time, start_time):

        if end_time < self.get_latest_timestamp():
            return

        while end_time > self.get_latest_timestamp() and self.job_finished() is not True:
            self.logger.info(
                f"Current timestamp {end_time} latest timestamp {self.get_latest_timestamp()}: waiting for new profiler data."
            )

            time.sleep(10)

        if self.job_finished():
            if start_time >= self.get_latest_timestamp():
                raise NoMoreProfilerData(end_time)

        return

    def get_system_metrics(self, start_time, end_time):
        events = self.system_metrics_reader.get_events(start_time, end_time)
        return events

    def get_framework_metrics(self, start_time, end_time):
        events = self.framework_metrics_reader.get_events(start_time, end_time)
        return events
