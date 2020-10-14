# First Party
# Standard Library
import shutil

# Third Party
import matplotlib.pyplot as plt
import numpy as np

from smdebug.exceptions import RuleEvaluationConditionMet
from smdebug.rules.rule import Rule


class StepOutlier(Rule):
    def __init__(
        self, base_trial, stddev=3, mode=None, n_outliers=10, scan_interval_us=60 * 1000 * 1000
    ):
        """
        This rule helps to detect outlier in step durations. Rule returns True if duration is larger than stddev * standard deviation.
        :param base_trial: the trial whose execution will invoke the rule
        :param stddev: factor by which to multiply the standard deviation. Default is 3
        :param mode: select mode under which steps have been saved and on which Rule should run on. Per default rule will run on steps from EVAL and TRAIN phase.
        :param n_outliers: How many outliers to ignore before rule returns True. Default 10.
        :param scan_interval_us: interval with which timeline files are scanned. Default is 60000000 us.
        """
        super().__init__(base_trial)
        self.stddev = stddev
        self.mode = mode
        self.n_outliers = n_outliers
        self.scan_interval_us = scan_interval_us
        self.step_durations = {}
        self.step_numbers = {}
        self.step_intervals = {}
        self.last_timestamp = self.base_trial.first_timestamp
        self.framework_metrics = {}
        self.framework_metrics_detailed = {}
        self.report[
            "RuleParameters"
        ] = f"threshold:{self.stddev}\nmode:{self.mode}\nn_outliers:{self.n_outliers}"
        self.report["Details"] = {}

    def invoke_at_step(self, step):
        pass

    def invoke(self, step):
        # iterate over timeline events
        current_timestamp = self.last_timestamp + self.scan_interval_us
        self.base_trial.wait_for_data(current_timestamp, self.last_timestamp)
        rule_condition = self.invoke_for_timerange(self.last_timestamp, current_timestamp)
        self.last_timestamp = current_timestamp
        if rule_condition:
            raise RuleEvaluationConditionMet(self.rule_name, step)

    def invoke_for_timerange(
        self, timestamp_start, timestamp_end, sys_events=None, framework_events=None
    ):

        # get system metric events
        if framework_events is None:
            events = self.base_trial.get_framework_metrics(timestamp_start, timestamp_end)
        else:
            events = framework_events

        # reset step duration dict
        self.step_intervals.update((key, {}) for key in self.step_intervals)

        # iterate over events and compute step duration
        for event in events:
            if "Step:ModeKeys" in event.event_name or (
                self.mode != None and event.event_name == self.mode
            ):
                # record information for profiler report
                self.report["Datapoints"] += 1

                # initialize dicts
                if event.node_id not in self.step_durations:
                    self.step_durations[event.node_id] = {}
                    self.step_numbers[event.node_id] = {}
                    self.step_intervals[event.node_id] = {}
                if event.event_name not in self.step_durations[event.node_id]:
                    self.step_durations[event.node_id][event.event_name] = []
                    self.step_numbers[event.node_id][event.event_name] = []

                # compute step duration
                diff = (event.end_time - event.start_time) / 1000000.0
                self.step_durations[event.node_id][event.event_name].append(diff)
                self.step_numbers[event.node_id][event.event_name].append(
                    event.event_args["step_num"]
                )

                # record begin and end time for step
                if event.event_args["step_num"] not in self.step_intervals[event.node_id]:
                    self.step_intervals[event.node_id][event.event_args["step_num"]] = (
                        event.start_time,
                        event.end_time,
                    )

        # iterate over training phases and get the number of step durations that exceeded the threshold
        for node_id in self.step_durations:
            for key in self.step_durations[node_id]:
                if len(self.step_durations[node_id][key]) > 100:

                    values = np.array(self.step_durations[node_id][key])

                    threshold = np.mean(values) + np.std(values) * self.stddev

                    # get indices of outliers
                    step_outliers = np.where(values > threshold)

                    # record step statistics
                    if node_id not in self.report["Details"]:
                        self.report["Details"][node_id] = {}
                    self.report["Details"][node_id] = {
                        "step_stats": {
                            "mean": np.mean(values),
                            "max": np.max(values),
                            "p99": np.quantile(values, 0.99),
                            "p95": np.quantile(values, 0.95),
                            "p50": np.quantile(values, 0.50),
                            "min": np.min(values),
                        }
                    }
                    # number of outliers
                    n = len(step_outliers[0])
                    # record information for profiler report
                    self.report["Details"][node_id]["number_of_outliers"] = n
                    self.report["Details"][node_id]["phase"] = key
                    self.report["Details"][node_id]["stddev"] = round(np.std(values), 2)
                    if n > self.n_outliers:

                        # create step histogram for normal steps
                        threshold = (
                            np.mean(values)
                            + np.std(self.step_durations[node_id][key]) * self.stddev
                        )
                        steps_normal = np.where(self.step_durations[node_id][key] < threshold)

                        values_normal = values[steps_normal]
                        probs, binedges = np.histogram(values_normal, bins=100)

                        self.logger.info(
                            f"Found {n} step durations on node {node_id} which exceeded {self.stddev} times the standard deviation of {round(np.std(values),2)} ms"
                        )

                        # record information for profiler report
                        self.report["Details"][node_id]["probs"] = probs.tolist()
                        self.report["Details"][node_id]["binedges"] = binedges.tolist()
                        self.report["Details"][node_id]["outliers"] = (
                            values[step_outliers].tolist(),
                        )
                        self.report["Details"][node_id]["step_numbers"] = np.array(
                            self.step_numbers[node_id][key]
                        )[step_outliers].tolist()
                        self.report["Violations"] += len(values[step_outliers].tolist())
                        self.report["RuleTriggered"] += 1

                        # find framework metrics that may be causing the step outliers
                        for outlier_step in self.report["Details"][node_id]["step_numbers"]:
                            if outlier_step in self.step_intervals[node_id]:
                                start_time_us, end_time_us = self.step_intervals[node_id][
                                    outlier_step
                                ]
                                for event in events:
                                    if (
                                        event.end_time >= start_time_us
                                        and event.start_time <= end_time_us
                                    ):
                                        if "Step" not in event.event_name:
                                            if event.event_phase not in self.framework_metrics:
                                                self.framework_metrics[event.event_phase] = 0
                                            self.framework_metrics[event.event_phase] += (
                                                event.end_time - event.start_time
                                            )
                                            if (
                                                event.event_name
                                                not in self.framework_metrics_detailed
                                            ):
                                                self.framework_metrics_detailed[
                                                    event.event_name
                                                ] = 0
                                            self.framework_metrics_detailed[event.event_name] += (
                                                event.end_time - event.start_time
                                            )

                        # create charts for profiler report
                        for node_id in self.report["Details"]:
                            if "binedges" in self.report["Details"][node_id]:

                                labels = list(self.framework_metrics.keys())
                                values = list(self.framework_metrics.values())
                                sizes = np.array(values) / float(np.sum(values)) * 100

                                # create pie chart for framework metrics (aggregated by event phase)
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                                title1 = ax1.set_title(
                                    f"Framework metrics during step outliers on node {node_id}"
                                )
                                ax1.pie(self.framework_metrics.values(), autopct="%1.1f%%")
                                ax1.legend(
                                    loc="lower center",
                                    labels=["%s, %1.1f %%" % (l, s) for l, s in zip(labels, sizes)],
                                    bbox_to_anchor=(0.5, -len(labels) * 0.1),
                                    borderaxespad=len(labels) * 2,
                                )

                                # create histogram for normal step durations
                                width = (
                                    self.report["Details"][node_id]["binedges"][1]
                                    - self.report["Details"][node_id]["binedges"][0]
                                )
                                ax2.bar(
                                    self.report["Details"][node_id]["binedges"][:-1],
                                    self.report["Details"][node_id]["probs"],
                                    width,
                                )

                                axes = plt.gca()
                                ax2.set_ylim([0, np.max(self.report["Details"][node_id]["probs"])])
                                ax2.legend([self.report["Details"][node_id]["phase"]])
                                ax2.set_xlabel("Step duration in s")
                                ax2.set_ylabel("Counts")
                                title2 = ax2.set_title(
                                    "Step duration histogram (without outliers) on node: " + node_id
                                )

                                # output filename
                                filename = node_id + "_" + "step_duration_histogram.png"

                                # save file
                                try:
                                    plt.savefig(
                                        "/opt/ml/processing/outputs/.sagemaker-ignore/" + filename,
                                        box_extra_artists=[title2, title1],
                                        # bbox_inches="tight",
                                    )
                                    shutil.move(
                                        "/opt/ml/processing/outputs/.sagemaker-ignore/" + filename,
                                        "/opt/ml/processing/outputs/profiler-reports/" + filename,
                                    )
                                except:
                                    self.logger.info("Error while saving file")

                                plt.close()

                            # bar chart for detailed framework metrics
                            plt.rcParams["axes.spines.left"] = False
                            plt.rcParams["axes.spines.right"] = False
                            plt.rcParams["axes.spines.top"] = False
                            plt.rcParams["axes.spines.bottom"] = False

                            values = list(self.framework_metrics_detailed.values())
                            labels = list(self.framework_metrics_detailed.keys())
                            if len(values) > 0:
                                fig, ax = plt.subplots(figsize=(15, int(len(values) / 4.0)))
                                ax.barh(y=labels, width=(values))
                                for i, v in enumerate(values):
                                    ax.text(v, i, str(v), color="black", ha="left", va="center")

                                if max(values) / min(values) > 1000:
                                    ax.set_xscale("log")
                                ax.set_xlabel("Time in us")

                                # output filename
                                filename = "histogram_step_outlier_framework.png"

                                # save file
                                try:
                                    plt.savefig(
                                        "/opt/ml/processing/outputs/.sagemaker-ignore/" + filename,
                                        bbox_inches="tight",
                                    )
                                    shutil.move(
                                        "/opt/ml/processing/outputs/.sagemaker-ignore/" + filename,
                                        "/opt/ml/processing/outputs/profiler-reports/" + filename,
                                    )
                                except:
                                    self.logger.info("Error while saving file")

                                plt.close()
                        return True

                    self.logger.info(
                        f"Average duration of {key} steps on node {node_id} is: {round(np.mean(self.step_durations[node_id][key]),2)} ms Standard deviation is: {round(np.std(self.step_durations[node_id][key]),2)} ms"
                    )

        return False
