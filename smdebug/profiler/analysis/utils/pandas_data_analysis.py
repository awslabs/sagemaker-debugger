# Standard Library
from collections import defaultdict
from enum import Enum

# Third Party
import pandas as pd

# First Party
from smdebug.core.logger import get_logger


class StatsBy(Enum):
    """
    Enum to get stats by different categories.
    """

    # training phase such as TRAIN/EVAL/GLOBAL.
    TRAINING_PHASE = "training_phase"

    # framework metrics such as function names/ operator names
    FRAMEWORK_METRICS = "framework_metric"

    # event phase name as retrieved from events
    PROCESS = "process"


class Resource(Enum):
    """
    Enum to specify the device/resource specified in system metrics
    TODO: Add other resources as system metrics are updated
    """

    CPU = "cpu"

    GPU = "gpu"


# Container class for job stats
class JobStats(dict):
    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(pd.DataFrame.from_dict(self.__dict__).T)


class PandasFrameAnalysis:
    """
    This class contains some of the common utils that can be used with
    the system metrics and framework metrics DataFrames.
    The functions here only query the DataFrame and return results. The results
    will then have to be plotted/visualized by the user or other utils.
    """

    def __init__(self, system_df, framework_df):
        self.sys_metrics_df = system_df
        self.framework_metrics_df = framework_df

        self.framework_metrics_df["duration_us"] = (
            self.framework_metrics_df["end_time_us"] - self.framework_metrics_df["start_time_us"]
        )

    def get_job_statistics(self):
        """
        Returns a Dictionary with information about runtime of training job, initilization, training loop and finalization.
        """
        job_statistics = JobStats()
        job_statistics["start_time"] = min(self.sys_metrics_df["timestamp"])
        job_statistics["end_time"] = max(self.sys_metrics_df["timestamp"])
        job_statistics["job_duration"] = (
            max(self.sys_metrics_df["timestamp_us"]) - min(self.sys_metrics_df["timestamp_us"])
        ) / 1000
        step_0 = self.framework_metrics_df[
            (self.framework_metrics_df["step"] == 0)
            & (
                self.framework_metrics_df["framework_metric"].isin(
                    ["Step:ModeKeys.TRAIN", "Step:ModeKeys.GLOBAL"]
                )
            )
        ].reset_index(drop=True)
        job_statistics["training_loop_start"] = step_0["start_time"][0]
        job_statistics["training_loop_end"] = max(self.framework_metrics_df["end_time"])
        job_statistics["training_loop_duration"] = (
            max(self.framework_metrics_df["end_time_us"]) - step_0["start_time_us"]
        ) / 1000
        job_statistics["initialization"] = step_0["start_time_us"][0] / 1000
        job_statistics["finalization"] = (
            max(self.sys_metrics_df["timestamp_us"]) - max(self.framework_metrics_df["end_time_us"])
        ) / 1000
        job_statistics["initialization_%"] = (
            job_statistics["initialization"] / job_statistics["job_duration"]
        ) * 100
        job_statistics["training_loop_%"] = (
            job_statistics["training_loop_duration"] / job_statistics["job_duration"]
        ) * 100
        job_statistics["finalization_%"] = (
            job_statistics["finalization"] / job_statistics["job_duration"]
        ) * 100

        return job_statistics

    def get_step_statistics(self, by=StatsBy.TRAINING_PHASE):
        """
        Get average, minimum, maximum, p50, p95, p99 stats on step duration
        :param by: by default, stats are grouped by framework_metric. The other options are
        to get stats by training phase - train/eval/global or grouped by process. This parameter
        should be of type StatsBy
        """
        if not isinstance(by, StatsBy):
            get_logger("smdebug-profiler").info(f"{by} should be of type StatsBy")
            return None

        by = by.value
        step_stats = None
        if by in [StatsBy.FRAMEWORK_METRICS.value, StatsBy.PROCESS.value]:
            # TODO: Consider that some events may be occurring in parallel
            step_stats = (
                self.framework_metrics_df.groupby([by])["duration_us"]
                .describe(percentiles=[0.5, 0.95, 0.99])
                .unstack()
                .reset_index()
            )
            step_stats = step_stats.pivot(index=by, columns="level_0", values=0).reset_index()
            step_stats.columns.name = ""
            step_stats = step_stats.drop(["count", "std"], axis="columns")
            step_stats = step_stats[[by, "mean", "min", "max", "50%", "95%", "99%"]]
        elif by == StatsBy.TRAINING_PHASE.value:
            phase_metrics_df = self.framework_metrics_df[
                self.framework_metrics_df["framework_metric"].str.contains("Step:ModeKeys")
            ]
            step_stats = (
                phase_metrics_df.groupby(["framework_metric"])["duration_us"]
                .describe(percentiles=[0.5, 0.95, 0.99])
                .unstack()
                .reset_index()
            )
            step_stats = step_stats.pivot(
                index="framework_metric", columns="level_0", values=0
            ).reset_index()
            step_stats.columns.name = ""
            step_stats = step_stats.drop(["count", "std"], axis="columns")
            step_stats = step_stats[["framework_metric", "mean", "min", "max", "50%", "95%", "99%"]]
        if step_stats is not None:
            step_stats.columns = [
                by,
                "duration_mean_us",
                "duration_min_us",
                "duration_max_us",
                "duration_p50_us",
                "duration_p95_us",
                "duration_p99_us",
            ]
        return step_stats

    def _get_utilization_phase_by_time_interval(self, interval_df):
        """
        For a given set of framework metric intervals, what are the corresponding
        system metrics duration each period
        :param interval_df: DataFrame containing start time, end time, and name of the phase
        thats active during the interval.
        """

        def helper(start, end, phase):
            self.sys_metrics_df.loc[
                (self.sys_metrics_df["timestamp_us"].between(start, end, inclusive=True)), "phase"
            ] = phase

        interval_df.apply(
            lambda x: helper(x["start_time_us"], x["end_time_us"], x["phase"]), axis=1
        )

    def get_utilization_stats(self, resource=None, by=None, phase=None):
        """
        Get CPU/GPU utilization stats
        :param resource: system resource for which utilization stats have to be computed. Type: Resource
        :param by: By default, get overall utilization stats. When by="training_phase",
        utilization stats are provided per training phase interval. Type: StatsBy
        :param phase: List of training phase to find intervals for. If nothing is mentioned, intervals
        are determined for all training phases available.
        :return: Dataframe containing utilization stats
        """
        if (by is not None) and (not isinstance(by, StatsBy)):
            get_logger("smdebug-profiler").info(f"{by} should be of type StatsBy")
            return None
        if (resource is not None) and (not isinstance(resource, (list, Resource))):
            get_logger("smdebug-profiler").info(f"{resource} should be of type list or Resource")
            return None

        if resource is None:
            resources = [Resource.CPU.value, Resource.GPU.value]
        else:
            if isinstance(resource, Resource):
                resource = [resource]
            resources = [x.value for x in resource]

        if by == StatsBy.TRAINING_PHASE:
            interval_df = self.get_training_phase_intervals(phase)
            self._get_utilization_phase_by_time_interval(interval_df)

        df_for_concat = []
        for resrc in resources:
            sys_resrc_df = self.sys_metrics_df[
                self.sys_metrics_df["system_metric"].str.contains(resrc)
            ].reset_index()
            if by == StatsBy.TRAINING_PHASE:
                sys_resrc_df = (
                    sys_resrc_df.groupby("phase")["value"]
                    .describe(percentiles=[0.5, 0.95, 0.99])
                    .unstack()
                    .reset_index()
                )
                sys_resrc_df = sys_resrc_df.pivot(
                    index="phase", columns="level_0", values=0
                ).reset_index()
                sys_resrc_df.columns.name = ""
                sys_resrc_df = sys_resrc_df.drop(["count", "std"], axis="columns")
                sys_resrc_df = sys_resrc_df[["phase", "mean", "min", "max", "50%", "95%", "99%"]]
                sys_resrc_df.insert(0, "Resource", resrc)
                df_for_concat.append(sys_resrc_df)
                columns = [
                    "Resource",
                    "Training_phase",
                    "utilization_mean_%",
                    "utilization_min_%",
                    "utilization_max_%",
                    "utilization_p50_%",
                    "utilization_p95_%",
                    "utilization_p99_%",
                ]
            else:
                sys_resrc_df = (
                    sys_resrc_df.groupby(lambda _: resrc)["value"]
                    .describe(percentiles=[0.5, 0.95, 0.99])
                    .unstack()
                    .reset_index()
                )
                sys_resrc_df = sys_resrc_df.pivot(
                    index="level_1", columns="level_0", values=0
                ).reset_index()
                sys_resrc_df.columns.name = ""
                sys_resrc_df = sys_resrc_df.drop(["count", "std"], axis="columns")
                sys_resrc_df = sys_resrc_df[["level_1", "mean", "min", "max", "50%", "95%", "99%"]]
                df_for_concat.append(sys_resrc_df)
                columns = [
                    "Resource",
                    "utilization_mean_%",
                    "utilization_min_%",
                    "utilization_max_%",
                    "utilization_p50_%",
                    "utilization_p95_%",
                    "utilization_p99_%",
                ]

        util_stats = pd.concat(df_for_concat).reset_index(drop=True)
        util_stats.columns = columns
        return util_stats

    def get_device_usage_stats(self, device=Resource.CPU, utilization_ranges=None):
        """
        Find the usage spread based on utilization ranges. If ranges are not provided,
        >90, 10-90, <10 are considered
        :param device: Resource.cpu or Resource.gpu. Type: Resource
        :param utilization_ranges: list of tuples
        """
        if utilization_ranges is None:
            utilization_ranges = [(90, 100), (10, 90), (0, 10)]
        if not isinstance(utilization_ranges, list):
            get_logger("smdebug-profiler").info(
                f"{utilization_ranges} should be a list of tuples containing the ranges"
            )
            return {}
        if len(utilization_ranges) == 0:
            get_logger("smdebug-profiler").info(f"{utilization_ranges} cannot be empty")
            return {}
        if not isinstance(device, Resource):
            get_logger("smdebug-profiler").info(f"{device} should be of type Resource")
            return {}
        device = device.value
        device_sys_df = self.sys_metrics_df[
            self.sys_metrics_df["system_metric"].str.contains(device)
        ].reset_index()

        usage_dict = defaultdict(int)
        for utilization_range in utilization_ranges:
            if len(utilization_range) != 2:
                get_logger("smdebug-profiler").info(
                    f"Invalid range {utilization_range} for usage stats"
                )
                return {}
            else:
                start, end = utilization_range
                between_range = len(
                    device_sys_df[(device_sys_df["value"].between(start, end, inclusive=True))]
                )

            usage_dict[utilization_range] = between_range
        return usage_dict

    def get_training_phase_intervals(self, phase=None):
        """
        This function splits framework data into before train, train, between train and eval, eval, and after eval.
        :param phase: List of training phase to find intervals for. If nothing is mentioned, intervals
        are determined for all training phases available. Type: string or List of strings
        :return: DataFrame containing the intervals
        """
        process_list = self.framework_metrics_df["process"].unique()
        if phase is None:
            phase = [x for x in process_list if "Step:ModeKeys" in x]

        if isinstance(phase, str):
            phase = [phase]

        if not isinstance(phase, list):
            get_logger("smdebug-profiler").info(f"{phase} should be a list of strings")
            return None

        # Filter out phases that are not available in process list
        phase = [x for x in phase if x in process_list]

        if len(phase) == 0:
            get_logger("smdebug-profiler").info(
                f"None of the phase strings matched the phases available in the framework metrics DataFrame"
            )
            return None

        mode_df = self.framework_metrics_df[
            self.framework_metrics_df["framework_metric"].isin(phase)
        ]
        training_phases = mode_df["framework_metric"].unique()
        if len(phase) > 1:
            mode_df = mode_df.groupby(
                mode_df["framework_metric"].ne(mode_df["framework_metric"].shift()).cumsum()
            )
            mode_df = mode_df.apply(
                lambda x: pd.DataFrame(
                    {
                        "start_time_us": [x["start_time_us"].min()],
                        "end_time_us": [x["end_time_us"].max()],
                        "phase": [x["framework_metric"].iloc[0]],
                    }
                )
            ).reset_index(drop=True)
        else:
            mode_df = mode_df[["start_time_us", "end_time_us", "framework_metric"]].reset_index(
                drop=True
            )
            mode_df.rename({"framework_metric": "phase"}, axis="columns", inplace=True)

        for i in range(len(mode_df.index) - 1):
            ind = mode_df.index[i]
            next_index = ind + 0.5
            this_phase = mode_df["phase"][ind]
            next_phase = mode_df["phase"][mode_df.index[i + 1]]
            if this_phase in training_phases and next_phase in training_phases:
                row = {
                    "start_time_us": mode_df["end_time_us"][ind] + 1,
                    "end_time_us": mode_df["start_time_us"][mode_df.index[i + 1]] - 1,
                    "phase": "Between " + " and ".join(sorted([this_phase, next_phase])),
                }
                mode_df.loc[next_index] = row

        row = {
            "start_time_us": self.sys_metrics_df["timestamp_us"].min(),
            "end_time_us": mode_df["start_time_us"][0] - 1,
            "phase": "Before " + mode_df["phase"][0],
        }
        mode_df.loc[-1] = row
        mode_df = mode_df.sort_index().reset_index(drop=True)
        row = {
            "start_time_us": mode_df["end_time_us"][mode_df.index[-1]] + 1,
            "end_time_us": self.sys_metrics_df["timestamp_us"].max(),
            "phase": "After " + mode_df["phase"][mode_df.index[-1]],
        }
        mode_df.loc[mode_df.index[-1] + 1] = row
        return mode_df
