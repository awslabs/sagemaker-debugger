# First Party
# Third Party
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
        self.buffer = {
            "cpu_events": {},
            "gpu_events": {},
            "step_phases": {},
            "forward_events": {},
            "backward_events": {},
            "phase_durations": {},
            "horovod": {},
        }

        self.report[
            "RuleParameters"
        ] = f"threshold:{self.stddev}\nmode:{self.mode}\nn_outliers:{self.n_outliers}\nstddev:{self.stddev}"
        self.report["Details"] = {"step_details": {}}

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
                    if node_id not in self.report["Details"]["step_details"]:
                        self.report["Details"]["step_details"][node_id] = {}
                    self.report["Details"]["step_details"][node_id] = {
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

                    self.report["Details"]["step_details"][node_id]["number_of_outliers"] = n
                    self.report["Details"]["step_details"][node_id]["phase"] = key
                    self.report["Details"]["step_details"][node_id]["stddev"] = round(
                        np.std(values), 2
                    )
                    # create step histogram for normal steps
                    threshold = (
                        np.mean(values) + np.std(self.step_durations[node_id][key]) * self.stddev
                    )
                    steps_normal = np.where(self.step_durations[node_id][key] < threshold)

                    values_normal = values[steps_normal]
                    probs, binedges = np.histogram(values_normal, bins=100)

                    # record information for profiler report
                    self.report["Details"]["step_details"][node_id]["probs"] = probs.tolist()
                    self.report["Details"]["step_details"][node_id]["binedges"] = binedges.tolist()

                    self.report["Details"]["step_details"][node_id]["step_numbers"] = np.array(
                        self.step_numbers[node_id][key]
                    )[step_outliers].tolist()

                    if n > self.n_outliers:

                        self.logger.info(
                            f"Found {n} step durations on node {node_id} which exceeded {self.stddev} times the standard deviation of {round(np.std(values),2)} ms"
                        )
                        self.report["Violations"] += len(values[step_outliers].tolist())
                        self.report["RuleTriggered"] += 1

                        # find framework metrics that may be causing the step outliers
                        for outlier_step in self.report["Details"]["step_details"][node_id][
                            "step_numbers"
                        ]:
                            if outlier_step in self.step_intervals[node_id]:
                                start_time_us, end_time_us = self.step_intervals[node_id][
                                    outlier_step
                                ]

                                # find framework metrics that may be causing the CPU bottleneck
                                aggregate_framework_metrics(
                                    events, self.report, self.buffer, start_time_us, end_time_us
                                )

                    self.logger.info(
                        f"Average duration of {key} steps on node {node_id} is: {round(np.mean(self.step_durations[node_id][key]),2)} ms Standard deviation is: {round(np.std(self.step_durations[node_id][key]),2)} ms"
                    )
        if self.report["RuleTriggered"] > 0:
            return True
        return False
