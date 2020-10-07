# Third Party
import pandas as pd

# First Party
from smdebug.core.utils import match_inc
from smdebug.profiler.algorithm_metrics_reader import (
    LocalAlgorithmMetricsReader,
    S3AlgorithmMetricsReader,
)
from smdebug.profiler.profiler_constants import CONVERT_TO_MICROSECS
from smdebug.profiler.system_metrics_reader import LocalSystemMetricsReader, S3SystemMetricsReader
from smdebug.profiler.utils import (
    convert_utc_datetime_to_microseconds,
    us_since_epoch_to_human_readable_time,
)


class PandasFrame:
    def __init__(self, path, use_in_memory_cache=False, scan_interval=5000000000):

        self.path = path
        self.step_time_mapping = dict()

        # Reader for system and framework metrics
        if path.startswith("s3"):
            self.system_metrics_reader = S3SystemMetricsReader(self.path)
            self.framework_metrics_reader = S3AlgorithmMetricsReader(
                self.path, use_in_memory_cache=use_in_memory_cache
            )
        else:
            self.system_metrics_reader = LocalSystemMetricsReader(self.path)
            self.framework_metrics_reader = LocalAlgorithmMetricsReader(
                self.path, use_in_memory_cache=use_in_memory_cache
            )

        # list to store metrics
        self.system_metrics = []
        self.framework_metrics = []

        # we read data in chunks
        self.interval = scan_interval
        self.start_time = self.system_metrics_reader.get_timestamp_of_first_available_file()

    def get_all_system_metrics(self, selected_system_metrics=[]):
        """
        Get system metrics
        :param systemk_metrics_list: list of system metrics.If not empty, function will only return framework events that are part of this list.
        :return: System metrics DataFrame
        """
        # get all system metrics from last to current timestamp

        start_timestamp = self.system_metrics_reader.get_timestamp_of_first_available_file()
        end_timestamp = (
            self.system_metrics_reader.get_timestamp_of_latest_available_file() + self.interval
        )
        sys_events_df, _ = self.get_profiler_data_by_time(
            start_timestamp,
            end_timestamp,
            cache_metrics=False,
            selected_system_metrics=selected_system_metrics,
            get_framework_metrics=False,
        )

        return sys_events_df

    def get_all_framework_metrics(self, selected_framework_metrics=[]):
        """
        Get framework metrics
        :param selected_framework_metrics: list of framework metrics.If not empty, function will only return framework events that are part of this list.
        :return: Framework metrics DataFrame
        """
        # get all framework metrics from last to current timestamp
        self.framework_metrics_reader.refresh_event_file_list()

        start_timestamp = (
            self.system_metrics_reader.get_timestamp_of_first_available_file()
        )  # bug: get_events does not return the very first event
        end_timestamp = self.framework_metrics_reader.get_timestamp_of_latest_available_file()

        _, fw_events_df = self.get_profiler_data_by_time(
            start_timestamp,
            end_timestamp,
            cache_metrics=False,
            selected_framework_metrics=selected_framework_metrics,
            get_system_metrics=False,
        )

        return fw_events_df

    def convert_datetime_to_timestamp(self, timestamp):
        """
        A helper function to convert datetime into timestamp
        :param timestep: timestamp in datetime
        :return: timestamp in microseconds
        """
        timestamp = pd.to_datetime(timestamp, format="%Y-%m-%dT%H:%M:%S:%f", utc=True)
        return convert_utc_datetime_to_microseconds(timestamp)

    def get_framework_metrics_by_timesteps(self, timestep_list=[], selected_framework_metrics=[]):
        """
        Get framework metrics for a list of timeranges. This function is useful when we want to correlate framework metrics with system metrics. Framework metrics have a begin and end timestamp. System metrics have only a single timestamp.
        :param timestep_list: list of timestamps
        :param selected_framework_metrics: list of framework metrics which will be stored in the dataframe
        :return: Framework metrics DataFrame
        """
        # get min and max search range
        timestep_list = sorted(timestep_list)
        start_time_us = self.convert_datetime_to_timestamp(timestep_list[0])
        end_time_us = self.convert_datetime_to_timestamp(timestep_list[-1])

        # to avoid out of memory issues, we read data in chunks
        current_time_us = start_time_us
        if end_time_us - start_time_us > self.interval:
            current_time_us = start_time_us + self.interval
        else:
            current_time_us = end_time_us
        results = {}
        results_detailed = {}
        counter = 0

        while start_time_us < end_time_us:
            # get all framework metrics from last to current timestamp
            self.framework_metrics_reader.refresh_event_file_list()
            events = self.framework_metrics_reader.get_events(start_time_us, current_time_us)

            # iterate over system metrics timestamps and find overlap
            for index, timestamp in enumerate(timestep_list[counter:]):
                timestamp = self.convert_datetime_to_timestamp(timestamp)
                if timestamp >= current_time_us:
                    counter = index
                    break
                for event in events:
                    if len(selected_framework_metrics) > 0 and (
                        event.event_name not in selected_framework_metrics
                        and event.event_phase not in selected_framework_metrics
                    ):
                        continue
                    if event.start_time < timestamp and event.end_time > timestamp:
                        if event.event_phase not in results:
                            results[event.event_phase] = 0
                        results[event.event_phase] += event.end_time - event.start_time
                        if "Step" not in event.event_name:
                            if event.event_name not in results_detailed:
                                results_detailed[event.event_name] = 0
                            results_detailed[event.event_name] += event.end_time - event.start_time
            # read the next chunk of framework metrics
            start_time_us = current_time_us
            if current_time_us + self.interval < end_time_us:
                current_time_us = current_time_us + self.interval
            else:
                current_time_us = end_time_us

        framework_metrics = {}
        training_phase = {}

        for key in results:
            if "Step" in key:
                training_phase[key] = results[key]
            else:
                framework_metrics[key] = results[key]

        if len(framework_metrics.values()) > 0:
            max_value = float(max(list(framework_metrics.values())))
            for key in framework_metrics:
                framework_metrics[key] = framework_metrics[key] / max_value

        return framework_metrics, results_detailed, training_phase

    def get_framework_metrics_by_begin_and_end_timesteps(
        self, begin_timestep_list, end_timestep_list, selected_framework_metrics=[]
    ):
        """
        Get framework metrics for a set of given timeranges. This function is useful when we want to correlate framework metrics such as steps with other framework metrics such as dataloading, preprocessing etc.
        :param begin_timestep_list: list of start of intervals in datetime
        :param end_timestep_list: list of end intervals in datetime
        :param selected_framework_metrics: list of framework metrics which will be stored in the dataframe
        :return: Framework metrics DataFrame
        """
        # Get min and max timestamps from the list of timeranges
        start_time_us = self.convert_datetime_to_timestamp(min(begin_timestep_list))
        end_time_us = self.convert_datetime_to_timestamp(max(end_timestep_list))

        # in order to avoid out of memory issues we will read only chunks of data
        current_time_us = start_time_us
        if end_time_us - start_time_us > self.interval:
            current_time_us = start_time_us + self.interval
        else:
            current_time_us = end_time_us
        framework_metrics = {}
        framework_metrics_detailed = {}
        counter = 0
        while start_time_us < end_time_us:
            # get all framework metrics from last to current timestamp
            self.framework_metrics_reader.refresh_event_file_list()
            events = self.framework_metrics_reader.get_events(start_time_us, current_time_us)

            # iterate over start and end time intervals and find overlaps in the current timerange
            for index, (begin_timestamp, end_timestamp) in enumerate(
                zip(begin_timestep_list[counter:], end_timestep_list[counter:])
            ):
                begin_timestamp = self.convert_datetime_to_timestamp(begin_timestamp)
                end_timestamp = self.convert_datetime_to_timestamp(end_timestamp)

                # if we are out of range, stop searching for overlaps
                if begin_timestamp >= current_time_us:
                    counter = index
                    break
                # iterate over events in the current timerange
                for event in events:
                    if len(selected_framework_metrics) > 0 and (
                        event.event_name not in selected_framework_metrics
                        and event.event_phase not in selected_framework_metrics
                    ):
                        continue
                    if event.end_time >= begin_timestamp and event.start_time <= end_timestamp:
                        if "Step" not in event.event_name:
                            if event.event_phase not in framework_metrics:
                                framework_metrics[event.event_phase] = 0
                            framework_metrics[event.event_phase] += (
                                event.end_time - event.start_time
                            )
                            if event.event_name not in framework_metrics_detailed:
                                framework_metrics_detailed[event.event_name] = 0
                            framework_metrics_detailed[event.event_name] += (
                                event.end_time - event.start_time
                            )
            # read the next chunk of data
            start_time_us = current_time_us
            if current_time_us + self.interval < end_time_us:
                current_time_us = current_time_us + self.interval
            else:
                current_time_us = end_time_us

        # normalize cumulative time to 0-1
        if len(list(framework_metrics.values())) > 0:
            max_value = float(max(list(framework_metrics.values())))
            for key in framework_metrics:
                framework_metrics[key] = framework_metrics[key] / max_value
            max_value = float(max(list(framework_metrics_detailed.values())))
            for key in framework_metrics_detailed:
                framework_metrics_detailed[key] = framework_metrics_detailed[key] / max_value

        return framework_metrics, framework_metrics_detailed

    def get_profiler_data_by_time(
        self,
        start_time_us,
        end_time_us,
        cache_metrics=False,
        selected_framework_metrics=[],
        selected_system_metrics=[],
        get_framework_metrics=True,
        get_system_metrics=True,
    ):
        """
        Get metrics data within a time interval.
        :param start_time_us: Start of the interval in microseconds
        :param end_time_us: End of the interval in microseconds
        :param cache_metrics: If True, collect and return all metrics requested so far, else,
        :param framework_metrics_list: list of framework metrics. If not empty, function will only return framework events that are part of this list.
        :param selected_system_metrics: list of system metrics. If not empty, function will only return system events that are part of this list.
        :param selected_framework_metrics: if True, get framework metrics
        :param get_system_metrics: if True: get system metrics
       return current request
        :return: System metrics DataFrame, Framework metrics DataFrame
        """
        # read system metrics
        system_metrics = []
        if get_system_metrics:
            events = self.system_metrics_reader.get_events(start_time_us, end_time_us)

            # append new events to existing list
            for event in events:
                if len(selected_system_metrics) > 0 and event.name not in selected_system_metrics:
                    continue
                system_metrics.append(
                    [
                        # GPU and CPU metrics are recorded at slightly different timesteps, so we round the numbers
                        us_since_epoch_to_human_readable_time(
                            int(event.timestamp * CONVERT_TO_MICROSECS)
                        ),
                        int(event.timestamp * CONVERT_TO_MICROSECS),
                        event.value,
                        event.name,
                        event.dimension,
                        event.node_id,
                        event.type,
                    ]
                )

            if cache_metrics is True:
                self.system_metrics.extend(system_metrics)
                system_metrics = self.system_metrics

        # create data frame for system metrics
        system_metrics_df = pd.DataFrame(
            system_metrics,
            columns=[
                "timestamp",
                "timestamp_us",
                "value",
                "system_metric",
                "dimension",
                "nodeID",
                "type",
            ],
        )

        system_metrics_df["timestamp_us"] = system_metrics_df["timestamp_us"] - self.start_time

        # get framework metrics
        framework_metrics = []
        if get_framework_metrics:
            # only fetch a subset of data to avoid out of memory issues
            if end_time_us - start_time_us > self.interval:
                current_time_us = start_time_us + self.interval
            else:
                current_time_us = end_time_us

            while start_time_us < end_time_us:
                # get all framework metrics from last to current timestamp
                self.framework_metrics_reader.refresh_event_file_list()
                events = self.framework_metrics_reader.get_events(start_time_us, current_time_us)

                # append new events to existing list
                for event in events:
                    if len(selected_framework_metrics) > 0 and (
                        event.event_name not in selected_framework_metrics
                        and event.event_phase not in selected_framework_metrics
                    ):
                        continue
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
                    if event.event_args is not None and "bytes_fetched" in event.event_args:
                        bytes_fetched = event.event_args["bytes_fetched"]
                    else:
                        bytes_fetched = -1

                    framework_metrics.append(
                        [
                            us_since_epoch_to_human_readable_time(event.start_time),
                            us_since_epoch_to_human_readable_time(event.end_time),
                            event.start_time,
                            event.end_time,
                            event.tid,
                            event.pid,
                            name,
                            step,
                            bytes_fetched,
                            event.event_phase,
                            event.node_id,
                        ]
                    )
                # read the next chunk of data
                start_time_us = current_time_us
                if current_time_us + self.interval < end_time_us:
                    current_time_us = current_time_us + self.interval
                else:
                    current_time_us = end_time_us

                if cache_metrics is True:
                    self.framework_metrics.extend(framework_metrics)
                    framework_metrics = self.framework_metrics

        # create data frame for framework metrics
        framework_metrics_df = pd.DataFrame(
            framework_metrics,
            columns=[
                "start_time",
                "end_time",
                "start_time_us",
                "end_time_us",
                "tid",
                "pid",
                "framework_metric",
                "step",
                "bytes",
                "process",
                "nodeID",
            ],
        )
        framework_metrics_df["start_time_us"] = (
            framework_metrics_df["start_time_us"] - self.start_time
        )
        framework_metrics_df["end_time_us"] = framework_metrics_df["end_time_us"] - self.start_time

        return (
            system_metrics_df[system_metrics_df.duplicated() == False],
            framework_metrics_df[framework_metrics_df.duplicated() == False],
        )

    def get_profiler_data_by_step(self, start_step, end_step, cache_metrics=False):
        """
        Get metrics data within a step interval. We find the mapping between step number and time interval for
        the step as some events may not be associated with a step number yet.
        :param start_step: Start of step interval
        :param end_step: End of step interval
        :param cache_metrics: If True, collect and return all metrics requested so far, else,
        return current request
        :return: System metrics DataFrame, Framework metrics DataFrame
        """
        sys_metrics_df, fw_metrics_df = (
            self.get_all_system_metrics(),
            self.get_all_framework_metrics(),
        )

        fw_metrics_df = fw_metrics_df[
            (fw_metrics_df["step"].between(start_step, end_step, inclusive=True))
        ]
        start_time, end_time = (
            fw_metrics_df["start_time_us"].min(),
            fw_metrics_df["end_time_us"].max(),
        )

        sys_metrics_df = sys_metrics_df[
            (sys_metrics_df["timestamp_us"].between(start_time, end_time, inclusive=True))
        ]

        return sys_metrics_df, fw_metrics_df

    def get_all_dataloader_metrics(self, selected_framework_metrics=[]):
        """
        Get framework metrics
        :param selected_framework_metrics: list of framework metrics.If not empty, function will only return framework events that are part of this list.
        :return: Framework metrics DataFrame
        """
        # get all framework metrics from last to current timestamp
        self.framework_metrics_reader.refresh_event_file_list()

        start_timestamp = (
            self.system_metrics_reader.get_timestamp_of_first_available_file()
        )  # bug: get_events does not return the very first event
        end_timestamp = self.framework_metrics_reader.get_timestamp_of_latest_available_file()

        fw_events_df = self._get_dataloader_profiler_data_by_time(
            start_timestamp,
            end_timestamp,
            cache_metrics=False,
            selected_framework_metrics=selected_framework_metrics,
        )

        return fw_events_df

    def _get_dataloader_profiler_data_by_time(
        self, start_time_us, end_time_us, cache_metrics=False, selected_framework_metrics=[]
    ):
        """
        Get metrics data within a time interval.
        :param start_time_us: Start of the interval in microseconds
        :param end_time_us: End of the interval in microseconds
        :param cache_metrics: If True, collect and return all metrics requested so far, else,
        :param framework_metrics_list: list of framework metrics. If not empty, function will only return framework events that are part of this list.
        :return: Framework metrics DataFrame
        """
        # get framework metrics
        framework_metrics = []
        # only fetch a subset of data to avoid out of memory issues
        if end_time_us - start_time_us > self.interval:
            current_time_us = start_time_us + self.interval
        else:
            current_time_us = end_time_us

        while start_time_us < end_time_us:
            # get all framework metrics from last to current timestamp
            self.framework_metrics_reader.refresh_event_file_list()
            events = self.framework_metrics_reader.get_events(start_time_us, current_time_us)

            # append new events to existing list
            for event in events:
                if len(selected_framework_metrics) > 0 and not (
                    match_inc(event.event_name, selected_framework_metrics)
                    or match_inc(event.event_phase, selected_framework_metrics)
                ):
                    continue
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
                if event.event_args is not None and "worker_id" in event.event_args:
                    worker_id = event.event_args["worker_id"]
                else:
                    worker_id = -1

                if event.event_args is not None and "num_workers" in event.event_args:
                    num_workers = event.event_args["num_workers"]
                else:
                    num_workers = -1

                if event.event_args is not None and "pin_memory" in event.event_args:
                    pin_memory = "True" if event.event_args["pin_memory"] is True else "False"
                else:
                    pin_memory = "NA"

                framework_metrics.append(
                    [
                        us_since_epoch_to_human_readable_time(event.start_time),
                        us_since_epoch_to_human_readable_time(event.end_time),
                        event.start_time,
                        event.end_time,
                        event.duration,
                        event.pid,
                        name,
                        step,
                        worker_id,
                        num_workers,
                        pin_memory,
                        event.event_phase,
                        event.node_id,
                    ]
                )
            # read the next chunk of data
            start_time_us = current_time_us
            if current_time_us + self.interval < end_time_us:
                current_time_us = current_time_us + self.interval
            else:
                current_time_us = end_time_us

            if cache_metrics is True:
                self.framework_metrics.extend(framework_metrics)
                framework_metrics = self.framework_metrics

        # create data frame for framework metrics
        framework_metrics_df = pd.DataFrame(
            framework_metrics,
            columns=[
                "start_time",
                "end_time",
                "start_time_us",
                "end_time_us",
                "duration_us",
                "pid",
                "framework_metric",
                "step",
                "worker_id",
                "num_workers",
                "pin_memory",
                "process",
                "node_id",
            ],
        )
        framework_metrics_df["start_time_us"] = (
            framework_metrics_df["start_time_us"] - self.start_time
        )
        framework_metrics_df["end_time_us"] = framework_metrics_df["end_time_us"] - self.start_time
        return framework_metrics_df[framework_metrics_df.duplicated() == False]
