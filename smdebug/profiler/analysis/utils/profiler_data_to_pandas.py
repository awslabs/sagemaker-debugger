# Third Party
import pandas as pd

# First Party
from smdebug.profiler.algorithm_metrics_reader import (
    LocalAlgorithmMetricsReader,
    S3AlgorithmMetricsReader,
)
from smdebug.profiler.system_metrics_reader import LocalSystemMetricsReader, S3SystemMetricsReader
from smdebug.profiler.utils import us_since_epoch_to_human_readable_time


class PandasFrame:
    def __init__(self, path):

        self.path = path
        self.start_time = 0

        # Reader for system and framework metrics
        if path.startswith("s3"):
            self.system_metrics_reader = S3SystemMetricsReader(self.path)
            self.framework_metrics_reader = S3AlgorithmMetricsReader(self.path)
        else:
            self.system_metrics_reader = LocalSystemMetricsReader(self.path)
            self.framework_metrics_reader = LocalAlgorithmMetricsReader(self.path)

        self.last_timestamp_system_metrics = 0
        self.last_timestamp_framework_metrics = 0
        self.system_metrics = []
        self.framework_metrics = []

    def get_latest_data(self):

        # get all system metrics from last to current timestamp
        current_timestamp = self.system_metrics_reader.get_timestamp_of_latest_available_file()
        events = self.system_metrics_reader.get_events(
            self.last_timestamp_system_metrics, current_timestamp
        )
        self.last_timestamp_system_metrics = current_timestamp

        # append new events to existing list
        for event in events:
            self.system_metrics.append(
                [
                    # GPU and CPU metrics are recorded at slightly different timesteps, so we round the numbers
                    us_since_epoch_to_human_readable_time(int(event.timestamp * 1000) * 1000),
                    int(event.timestamp * 1000) * 1000,
                    event.value,
                    event.name,
                ]
            )

        # create data frame for system metrics
        system_metrics_df = pd.DataFrame(
            self.system_metrics, columns=["timestamp", "timestamp_us", "value", "system_metric"]
        )
        if system_metrics_df.shape[0] != 0:
            self.start_time = min(system_metrics_df["timestamp_us"])
        system_metrics_df["timestamp_us"] = system_metrics_df["timestamp_us"] - self.start_time

        # get all framework metrics from last to current timestamp
        self.framework_metrics_reader.refresh_event_file_list()
        current_timestamp = self.framework_metrics_reader.get_timestamp_of_latest_available_file()
        events = self.framework_metrics_reader.get_events(
            self.last_timestamp_framework_metrics, current_timestamp
        )
        self.last_timestamp_framework_metrics = current_timestamp

        # append new events to existing list
        for event in events:
            if event.event_args is not None and "step_num" in event.event_args:
                step = int(event.event_args["step_num"])
            else:
                step = -1
            if event.event_args is not None and "layer_name" in event.event_args:
                name = event.event_args["layer_name"]
            elif event.event_args is not None and "name" in event.event_args:
                name = event.event_args["name"]
            else:
                name = event.event_name

            self.framework_metrics.append(
                [
                    us_since_epoch_to_human_readable_time(event.start_time / 1000),
                    us_since_epoch_to_human_readable_time(event.end_time / 1000),
                    event.start_time / 1000.0,
                    event.end_time / 1000.0,
                    name,
                    step,
                ]
            )

        # create data frame for framework metrics
        framework_metrics_df = pd.DataFrame(
            self.framework_metrics,
            columns=[
                "start_time",
                "end_time",
                "start_time_us",
                "end_time_us",
                "framework_metric",
                "step",
            ],
        )
        framework_metrics_df["start_time_us"] = (
            framework_metrics_df["start_time_us"] - self.start_time
        )
        framework_metrics_df["end_time_us"] = framework_metrics_df["end_time_us"] - self.start_time

        return system_metrics_df, framework_metrics_df
