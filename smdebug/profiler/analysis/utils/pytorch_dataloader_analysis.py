# Third Party
import pandas as pd


class PT_dataloader_analysis:
    def __init__(self, pandas_frame):
        self.pd_data_frame = pandas_frame
        self.dataIter_metric = self.pd_data_frame.get_all_dataloader_metrics(
            selected_framework_metrics=[".*DataLoaderIter::GetNext"]
        )
        self.initialize_metric = self.pd_data_frame.get_all_dataloader_metrics(
            selected_framework_metrics=["DataLoaderIterInitialize"]
        )
        self.dataWorker_metric = self.pd_data_frame.get_all_dataloader_metrics(
            selected_framework_metrics=["DataLoaderWorker"]
        )
        self.analyze_dataIter = self.dataIter_metric.size > 0
        self.analyze_initializer = self.initialize_metric.size > 0
        self.analyze_workers = self.dataWorker_metric.size > 0

    def _inspect_iters(self, itertype):
        if not self.analyze_initializer:
            print("Trace events corresponding to DataLoaderIter initialization are not present")
            return

        multiprocessingIter = self.initialize_metric.loc[
            self.initialize_metric["framework_metric"] == itertype
        ]
        if multiprocessingIter.size == 0:
            print(f"Training is not configured to use {itertype} iterator")
            return
        pin_memory = multiprocessingIter["pin_memory"].unique()[0]
        num_workers = multiprocessingIter["num_workers"].unique()[0]
        print(
            f"Training is configured to use {itertype} with pin_memory enabled = {pin_memory} and number of workers = {num_workers}"
        )

        print(
            f"The {itertype} is initialized for {len(multiprocessingIter.index)} times during training"
        )
        median_duration = multiprocessingIter["duration_us"].median()
        max_duration = multiprocessingIter["duration_us"].max()

        print(
            f"Median Duration {median_duration} and Maximum duration for initialization of iterators {max_duration}"
        )
        md = multiprocessingIter.loc[
            multiprocessingIter["duration_us"] >= (2 * median_duration),
            ["start_time", "end_time", "duration_us"],
        ]

        if md.size > 0:
            print(
                f"Start time and End time for iterator initialization that are outliers (duration > 2 * median)"
            )

    def analyze_dataloaderIter_initialization(self):
        self._inspect_iters("_MultiProcessingDataLoaderIter.__init__")
        self._inspect_iters("_SingleProcessDataLoaderIter.__init__")

    def analyze_dataloaderWorkers(self):
        if not self.analyze_workers:
            print("Trace events corresponding to DataLoaderWorker processes are not present")
            return
        total_num_workers = len(self.dataWorker_metric.index)
        print(f"Total number of workers spun off during the training {total_num_workers}")
        median_duration = self.dataWorker_metric["duration_us"].median()
        max_duration = self.dataWorker_metric["duration_us"].max()
        print(
            f"Median Duration {median_duration} and Maximum duration for worker processes {max_duration}"
        )
        md = self.dataWorker_metric.loc[
            self.dataWorker_metric["duration_us"] >= (2 * median_duration),
            ["start_time", "end_time", "duration_us", "worker_id"],
        ].sort_values(by=["duration_us"], ascending=False)
        md = md.reset_index(drop=True)
        if md.size > 0:
            print(
                f"Start time and End time for iterator initialization that are outliers (duration > 2 * median)"
            )
            return md
        else:
            print(f"No outliers found in dataloader workers")
            return None

    def analyze_dataloader_getnext(self):
        if not self.analyze_dataIter:
            print("Trace events corresponding to DataLoaderIter::GetNext calls are not present")
            return

        total_calls = len(self.dataIter_metric.index)
        print(f"Total number of GetNext calls made during the training {total_calls}")
        median_duration = self.dataIter_metric["duration_us"].median()
        max_duration = self.dataIter_metric["duration_us"].max()
        print(
            f"Median Duration {median_duration} and Maximum duration for worker processes {max_duration}"
        )
        md = self.dataIter_metric.loc[
            self.dataIter_metric["duration_us"] >= (2 * median_duration),
            ["start_time", "end_time", "duration_us", "worker_id"],
        ].sort_values(by=["duration_us"], ascending=False)
        md = md.reset_index(drop=True)
        if md.size > 0:
            print(
                f"Start time and End time for GetNext durations that are outliers (duration > 2 * median)"
            )
            return md
        else:
            print(f"No outliers found in dataloader getnext invocations.")
            return None

    def analyze_batchtime(self):
        if not self.analyze_dataIter:
            print("Trace events corresponding to DataLoaderIter::GetNext calls are not present")
            return
        # Convert start time and end time in pandas datetime object and sort them by end time.
        self.dataIter_metric["start_time"] = pd.to_datetime(
            self.dataIter_metric["start_time"], format="%Y-%m-%dT%H:%M:%S:%f"
        )
        self.dataIter_metric["end_time"] = pd.to_datetime(
            self.dataIter_metric["end_time"], format="%Y-%m-%dT%H:%M:%S:%f"
        )
        self.dataIter_metric = self.dataIter_metric.sort_values(by=["node_id", "pid", "end_time"])
        # Compute the 'BatchTime_in_seconds' for every GetNext call.
        self.dataIter_metric["BatchTime_in_seconds"] = (
            self.dataIter_metric.groupby(["node_id", "pid"])["start_time"].diff().dt.total_seconds()
        )
        self.dataIter_metric["previous_batch_start"] = (
            self.dataIter_metric.groupby(["node_id", "pid"])["start_time"].shift().fillna(-1)
        )
        median_duration = self.dataIter_metric["BatchTime_in_seconds"].median()
        print(f"Median time spent on each batch of data = {median_duration} seconds")
        # Get the outlier duration. The cell prints the dataframe that contains the outlier.
        md = self.dataIter_metric.loc[
            self.dataIter_metric["BatchTime_in_seconds"] >= (2 * median_duration),
            ["previous_batch_start", "start_time", "BatchTime_in_seconds", "worker_id"],
        ].sort_values(by=["BatchTime_in_seconds"], ascending=False)
        md = md.reset_index(drop=True)
        if md.size > 0:
            print(
                f"Start time and End time for processing the databatches that are outliers (duration > 2 * median)"
            )
            return md
        else:
            print(f"No outliers found in time taken to process the databatches.")
            return None

    def plot_the_window(
        self, start_timestamp, end_timestamp, select_events=[".*"], select_dimensions=[".*"]
    ):
        from smdebug.profiler.analysis.notebook_utils.timeline_charts import TimelineCharts

        framework_metrics_reader = self.pd_data_frame.framework_metrics_reader
        system_metrics_reader = self.pd_data_frame.system_metrics_reader
        framework_metrics_reader.refresh_event_file_list()
        system_metrics_reader.refresh_event_file_list()

        view_timeline_charts = TimelineCharts(
            system_metrics_reader,
            framework_metrics_reader,
            starttime=start_timestamp,
            endtime=end_timestamp,
            select_events=select_events,
            select_dimensions=select_dimensions,
        )
        return view_timeline_charts
