# First Party
# Third Party
import numpy as np

from smdebug.exceptions import RuleEvaluationConditionMet
from smdebug.profiler.analysis.rules import utils
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
                self.report["Details"]["bottlenecks"] = {}
                for timestamp in self.cpu_bottlenecks:
                    self.report["Details"]["bottlenecks"][timestamp] = {}
                    self.report["Details"]["bottlenecks"][timestamp]["GPUs"] = self.cpu_bottlenecks[
                        timestamp
                    ][0]
                    self.report["Details"]["bottlenecks"][timestamp]["CPUs"] = self.cpu_bottlenecks[
                        timestamp
                    ][1]
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
                    utils.aggregate_framework_metrics(
                        fw_events, self.report, self.buffer, timestamp_us
                    )
                return True

        self.logger.info(f"Found {len(self.cpu_bottlenecks)} CPU bottlenecks")
        self.report["Violations"] = len(self.cpu_bottlenecks)
        self.report["Datapoints"] = self.datapoints
        self.report["Details"]["low_gpu_utilization"] = self.low_gpu

        self.report["Details"]["bottlenecks"] = {}
        for timestamp in self.cpu_bottlenecks:
            self.report["Details"]["bottlenecks"][timestamp] = {}
            self.report["Details"]["bottlenecks"][timestamp]["GPUs"] = self.cpu_bottlenecks[
                timestamp
            ][0]
            self.report["Details"]["bottlenecks"][timestamp]["CPUs"] = self.cpu_bottlenecks[
                timestamp
            ][1]
        return False
