# First Party
# Third Party
import numpy as np

from smdebug.exceptions import RuleEvaluationConditionMet
from smdebug.profiler.analysis.rules import utils
from smdebug.rules.rule import Rule


class IOBottleneck(Rule):
    def __init__(
        self,
        base_trial,
        threshold=50,
        gpu_threshold=10,
        io_threshold=50,
        patience=1000,
        scan_interval_us=60 * 1000 * 1000,
    ):
        """
        This rule helps to detect if GPU is underutilized due to IO bottlenecks. Rule returns True if number of IO bottlenecks exceeds a predefined threshold.
        :param base_trial: the trial whose execution will invoke the rule
        :param threshold: defines the threshold when Rule should return True. Default is 50 percent. So if there is a bottleneck more than 50% of the time during the training Rule will return True.
        :param gpu_threshold: threshold that defines when GPU is considered being under-utilized. Default is 70%
        :param io_threshold: threshold that defines high IO wait time. Default is above 50%
        :param patience: How many values to record before checking for IO bottlenecks. During training initilization, GPU is likely at 0 percent, so Rule should not check for underutilization immediatly. Default 1000.
        :param scan_interval_us: interval with which timeline files are scanned. Default is 60000000 us.
        """
        super().__init__(base_trial)
        self.threshold = threshold
        self.io_threshold = io_threshold
        self.gpu_threshold = gpu_threshold
        self.last_timestamp = self.base_trial.first_timestamp
        self.patience = patience
        self.scan_interval_us = scan_interval_us
        self.timestamp = 0
        self.datapoints = 0
        self.io_bottlenecks = {}

        # dicts for correlating framework metrics with CPU bottlenecks
        self.buffer = {
            "cpu_events": {},
            "gpu_events": {},
            "step_phases": {},
            "forward_events": {},
            "backward_events": {},
            "phase_durations": {},
            "horovod": {},
        }

        self.low_gpu = 0
        # snapshot_window_seconds == time difference within which timestamps are considered to belong to the same system snapshot
        # e.g. metric gpu1 is taken at t1, t1+100, t1+200 and metric core1 is taken at t1+0.001, t1+100.001, t1+200.001
        # profileragent is running multiple threads which may not wake up at exactly the same timestamp
        self.snapshot_window_seconds = 0.02
        self.max_datapoints = 1000000
        self.report[
            "RuleParameters"
        ] = f"threshold:{self.threshold}\nio_threshold:{self.io_threshold}\ngpu_threshold:{ self.gpu_threshold}\npatience:{self.patience}"

    def reset(self):
        self.io_bottlenecks = {}

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

        # reset dictionary
        if len(self.io_bottlenecks) > self.max_datapoints:
            self.reset()

        # get system metrics
        if sys_events is None:
            events = self.base_trial.get_system_metrics(timestamp_start, timestamp_end)
        else:
            events = sys_events

        gpu_values = {}
        io_values = {}

        for event in events:

            # round timestamp
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
            elif event.dimension == "I/OWaitPercentage" and event.value > self.io_threshold:
                if self.timestamp not in io_values:
                    io_values[self.timestamp] = []
                io_values[self.timestamp].append(event.value)

        # find timestamps when GPU utilization was low and IO wait time high
        times = []
        for time in io_values:
            if time in gpu_values:
                self.io_bottlenecks[time] = [len(gpu_values[time]), len(io_values[time])]
                times.append(time)

        # compare number of IO bottlenecks versus threshold
        if self.datapoints > self.patience:
            if (len(self.io_bottlenecks) / self.datapoints) * 100 > self.threshold:
                self.logger.info(
                    f"Found {len(self.io_bottlenecks)} IO bottlenecks that caused low GPU utilizations. This is above the threshold of {self.threshold}%"
                )

                # record information for profiler report
                self.report["RuleTriggered"] += 1
                self.report["Violations"] = len(self.io_bottlenecks)
                self.report["Datapoints"] = self.datapoints

                # record IO bottlenecks

                self.report["Details"]["bottlenecks"] = {}
                for timestamp in self.io_bottlenecks:
                    self.report["Details"]["bottlenecks"][timestamp] = {}
                    self.report["Details"]["bottlenecks"][timestamp]["GPUs"] = self.io_bottlenecks[
                        timestamp
                    ][0]
                    self.report["Details"]["bottlenecks"][timestamp]["CPUs"] = self.io_bottlenecks[
                        timestamp
                    ][1]
                    self.report["Details"]["low_gpu_utilization"] = self.low_gpu

                # get framework metrics
                if framework_events is None:
                    fw_events = self.base_trial.get_framework_metrics(
                        timestamp_start, timestamp_end
                    )
                else:
                    fw_events = framework_events

                # iterate over IO bottlenecks that have been seen in the current time segment
                for timestamp in times:
                    timestamp_us = timestamp * 1000 * 1000

                    # find framework metrics that may be causing the CPU bottleneck
                    utils.aggregate_framework_metrics(
                        fw_events, self.report, self.buffer, timestamp_us
                    )
                return True

        self.logger.info(f"Found {len(self.io_bottlenecks)} IO bottlenecks")

        # record information for profiler report
        self.report["Violations"] = len(self.io_bottlenecks)
        self.report["Datapoints"] = self.datapoints
        self.report["Details"]["low_gpu_utilization"] = self.low_gpu

        self.report["Details"]["bottlenecks"] = {}
        for timestamp in self.io_bottlenecks:
            self.report["Details"]["bottlenecks"][timestamp] = {}
            self.report["Details"]["bottlenecks"][timestamp]["GPUs"] = self.io_bottlenecks[
                timestamp
            ][0]
            self.report["Details"]["bottlenecks"][timestamp]["CPUs"] = self.io_bottlenecks[
                timestamp
            ][1]
        return False
