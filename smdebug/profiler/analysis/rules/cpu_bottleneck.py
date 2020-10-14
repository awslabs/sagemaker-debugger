# First Party
# Standard Library
import shutil

# Third Party
import matplotlib.pyplot as plt
import numpy as np

from smdebug.exceptions import RuleEvaluationConditionMet
from smdebug.rules.rule import Rule


class CPUBottleneck(Rule):
    def __init__(
        self,
        base_trial,
        threshold=50,
        gpu_threshold=10,
        cpu_threshold=90,
        patience=1000,
        scan_interval_us=60 * 1000 * 1000,
    ):
        """
        This rule helps to detect if GPU is underutilized due to CPU bottlenecks. Rule returns True if number of CPU bottlenecks exceeds a predefined threshold.
        :param base_trial: the trial whose execution will invoke the rule
        :param threshold: defines the threshold behyond which Rule should return True. Default is 50 percent. So if there is a bottleneck more than 50% of the time during the training Rule will return True.
        :param gpu_threshold: threshold that defines when GPU is considered being under-utilized. Default is 10%
        :param cpu_threshold: threshold that defines high CPU utilization. Default is above 90%
        :param patience: How many values to record before checking for CPU bottlenecks. During training initilization, GPU is likely at 0 percent, so Rule should not check for underutilization immediatly. Default 1000.
        :param scan_interval_us: interval with which timeline files are scanned. Default is 60000000 us.
        """
        super().__init__(base_trial)
        self.threshold = threshold
        self.cpu_threshold = cpu_threshold
        self.gpu_threshold = gpu_threshold
        self.last_timestamp = self.base_trial.first_timestamp
        self.patience = patience
        self.scan_interval_us = scan_interval_us

        # placeholders
        self.cpu_bottlenecks = {}
        self.total_time_per_phase = {}
        self.total_time_per_event = {}
        self.datapoints = 0
        self.low_gpu = 0
        self.timestamp = 0
        # snapshot_window_seconds == time difference within which timestamps are considered to belong to the same system snapshot
        # e.g. metric gpu1 is taken at t1, t1+100, t1+200 and metric core1 is taken at t1+0.001, t1+100.001, t1+200.001
        # profileragent is running multiple threads which may not wake up at exactly the same timestamp
        self.snapshot_window_seconds = 0.02
        self.max_datapoints = 1000000
        self.report[
            "RuleParameters"
        ] = f"threshold:{self.threshold}\ncpu_threshold:{self.cpu_threshold}\ngpu_threshold:{ self.gpu_threshold}\npatience:{self.patience}"

    def invoke_at_step(self, step):
        pass

    def reset(self):
        self.cpu_bottlenecks = {}

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

        # reset dictionary
        if len(self.cpu_bottlenecks) > self.max_datapoints:
            self.reset()

        # get system metric events
        if sys_events is None:
            events = self.base_trial.get_system_metrics(timestamp_start, timestamp_end)
        else:
            events = sys_events

        gpu_values = {}
        cpu_values = {}

        for event in events:
            # system metrics reader reads events in increasing order, but events
            # from same time-snapshot are recorded at slightly different timestamps
            # events that are within 20ms interval are considered to belong to the same time-snapshot
            if np.abs(self.timestamp - event.timestamp) > self.snapshot_window_seconds:
                self.timestamp = event.timestamp
                self.datapoints += 1

            # get events where GPU utilization is low
            if event.dimension == "GPUUtilization" and event.value < self.gpu_threshold:
                if self.timestamp not in gpu_values:
                    self.low_gpu += 1
                    gpu_values[self.timestamp] = []
                gpu_values[self.timestamp].append(event.value)

            # get events when GPU utilization is hight
            elif event.dimension == "CPUUtilization" and event.value > self.cpu_threshold:
                if self.timestamp not in cpu_values:
                    cpu_values[self.timestamp] = []
                cpu_values[self.timestamp].append(event.value)

        # find timestamps when GPU utilization was low and CPU utilization high
        times = []
        for time in cpu_values:
            if time in gpu_values:
                self.cpu_bottlenecks[time] = [len(gpu_values[time]), len(cpu_values[time])]
                times.append(time)

        # compare number of CPU bottlenecks versus threshold
        if self.datapoints > self.patience:
            if (len(self.cpu_bottlenecks) / self.datapoints) * 100 > self.threshold:
                self.logger.info(
                    f"Found {len(self.cpu_bottlenecks)} CPU bottlenecks that caused low GPU utilizations. This is above the threshold of {self.threshold}%"
                )

                # record information for profiler report
                self.report["RuleTriggered"] += 1
                self.report["Violations"] = len(self.cpu_bottlenecks)
                self.report["Datapoints"] = self.datapoints

                # record CPU bottlenecks
                for timestamp in self.cpu_bottlenecks:
                    self.report["Details"][timestamp] = {}
                    self.report["Details"][timestamp]["GPUs"] = self.cpu_bottlenecks[timestamp][0]
                    self.report["Details"][timestamp]["CPUs"] = self.cpu_bottlenecks[timestamp][1]
                    self.report["Details"]["low_gpu_utilization"] = self.low_gpu

                # get framework metric events
                if framework_events is None:
                    fw_events = self.base_trial.get_framework_metrics(
                        timestamp_start, timestamp_end
                    )
                else:
                    fw_events = framework_events

                # iterate over CPU bottlenecks that have been seen in the current time segment
                for timestamp in times:
                    timestamp_us = timestamp * 1000 * 1000

                    # find framework metrics that may be causing the CPU bottleneck
                    for event in fw_events:
                        if event.start_time < timestamp_us and event.end_time > timestamp_us:

                            # aggregate framework metrics by event phase
                            if event.event_phase not in self.total_time_per_phase:
                                self.total_time_per_phase[event.event_phase] = 0
                            self.total_time_per_phase[event.event_phase] += (
                                event.end_time - event.start_time
                            )

                            # aggregate framework metrics by event name
                            if (
                                "Step" not in event.event_name
                                and event.event_name not in self.total_time_per_event
                            ):
                                self.total_time_per_event[event.event_name] = 0
                            if "Step" not in event.event_name:
                                self.total_time_per_event[event.event_name] += (
                                    event.end_time - event.start_time
                                )

                framework_metrics = {}
                training_phase = {}

                for key in self.total_time_per_phase:
                    if "Step" in key:
                        training_phase[key] = self.total_time_per_phase[key]
                    else:
                        framework_metrics[key] = self.total_time_per_phase[key]

                # create pie chart1: if GPU utilization was low how often was this caused by an CPU bottleneck
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
                title1 = ax1.set_title("Low GPU usage caused by CPU bottlenecks")
                ax1.pie(
                    [
                        self.report["Datapoints"] - self.report["Details"]["low_gpu_utilization"],
                        self.report["Details"]["low_gpu_utilization"] - len(self.report["Details"]),
                        len(self.report["Details"]),
                    ],
                    autopct="%1.1f%%",
                )
                ax1.legend(
                    [
                        "GPU usage above threshold",
                        "GPU usage below threshold",
                        "Low GPU usage due to CPU bottlenecks",
                    ],
                    loc="lower center",
                )

                # create pie chart2: framework metrics (aggregated by event phase)
                labels = list(framework_metrics.keys())
                values = list(framework_metrics.values())
                sizes = np.array(values) / float(np.sum(values)) * 100
                ax2.pie(framework_metrics.values(), autopct="%1.1f%%", labeldistance=1.4)
                title2 = ax2.set_title(
                    "Time spent in framework metrics (aggregated by event phase)"
                )
                ax2.legend(
                    loc="lower center",
                    labels=["%s, %1.1f %%" % (l, s) for l, s in zip(labels, sizes)],
                    bbox_to_anchor=(0.5, -len(labels) * 0.1),
                    borderaxespad=len(labels) * 2,
                )

                # create pie chart3: training phase
                labels = list(training_phase.keys())
                values = list(training_phase.values())
                sizes = np.array(values) / float(np.sum(values)) * 100
                ax3.pie(values, autopct="%1.1f%%", labeldistance=1.4)
                title3 = ax3.set_title("Time spent in training and validation phase")
                ax3.legend(
                    loc="lower center",
                    labels=["%s, %1.1f %%" % (l, s) for l, s in zip(labels, sizes)],
                    bbox_to_anchor=(0.5, -len(labels) * 0.1),
                    borderaxespad=len(labels) * 2,
                )

                # output filename
                filename = "pie_charts_cpu_bottleneck.png"

                # save file
                try:
                    plt.savefig(
                        "/opt/ml/processing/outputs/.sagemaker-ignore" + filename,
                        bbox_extra_artists=[title1, title2, title3],
                        # bbox_inches="tight",
                    )
                    shutil.move(
                        "/opt/ml/processing/outputs/.sagemaker-ignore" + filename,
                        "/opt/ml/processing/outputs/profiler-reports/" + filename,
                    )
                except:
                    self.logger.info("Error while saving file")

                plt.close()

                # create bar chart for detailed framework metrics
                plt.rcParams["axes.spines.left"] = False
                plt.rcParams["axes.spines.right"] = False
                plt.rcParams["axes.spines.top"] = False
                plt.rcParams["axes.spines.bottom"] = False

                values = list(self.total_time_per_event.values())
                if len(values):
                    fig, ax = plt.subplots(figsize=(15, int(len(values) / 4.0)))
                    ax.barh(y=list(self.total_time_per_event.keys()), width=(values))

                    for i, v in enumerate(self.total_time_per_event.values()):
                        ax.text(v, i, str(v), color="black", ha="left", va="center")

                    if max(values) / min(values) > 1000:
                        ax.set_xscale("log")
                    ax.set_xlabel("Time in us")

                    # output filename
                    filename = "histogram_cpu_bottleneck_framework.png"

                    # save file
                    try:
                        plt.savefig(
                            "/opt/ml/processing/outputs/.sagemaker-ignore" + filename,
                            bbox_inches="tight",
                        )
                        shutil.move(
                            "/opt/ml/processing/outputs/.sagemaker-ignore" + filename,
                            "/opt/ml/processing/outputs/profiler-reports/" + filename,
                        )
                    except:
                        self.logger.info("Error while saving file")

                    plt.close()
                return True

        self.logger.info(f"Found {len(self.cpu_bottlenecks)} CPU bottlenecks")
        self.report["Violations"] = len(self.cpu_bottlenecks)
        self.report["Datapoints"] = self.datapoints
        self.report["Details"]["low_gpu_utilization"] = self.low_gpu
        for timestamp in self.cpu_bottlenecks:
            self.report["Details"][timestamp] = {}
            self.report["Details"][timestamp]["GPUs"] = self.cpu_bottlenecks[timestamp][0]
            self.report["Details"][timestamp]["CPUs"] = self.cpu_bottlenecks[timestamp][1]
        return False
