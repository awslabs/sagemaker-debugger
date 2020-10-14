# First Party
# Third Party
import numpy as np

from smdebug.exceptions import RuleEvaluationConditionMet
from smdebug.rules.rule import Rule


class MaxInitializationTime(Rule):
    def __init__(self, base_trial, threshold=20, scan_interval_us=60 * 1000 * 1000):
        """
        This rule helps to detect if the training intialization is taking too much time. The rule waits until first
        step is available.

        :param base_trial: the trial whose execution will invoke the rule
        :param threshold: defines the threshold in minutes to wait for first step to become available. Default is 20 minutes.
        :param scan_interval_us: interval with which timeline files are scanned. Default is 60000000 us.
        """
        super().__init__(base_trial)
        self.threshold = threshold
        self.scan_interval_us = scan_interval_us
        self.first_timestamp = self.base_trial.first_timestamp
        self.last_timestamp = self.base_trial.first_timestamp
        self.system_start_time = np.inf
        self.system_end_time = 0
        self.report["RuleParameters"] = f"threshold:{self.threshold}"
        self.report["Details"]["step_num"] = {}

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

        # get first and last timestamp of system metrics: used in profiler report to compute total training time
        if sys_events is None:
            events = self.base_trial.get_system_metrics(timestamp_start, timestamp_end)
        else:
            events = sys_events
        for event in events:
            if event.timestamp < self.system_start_time:
                self.system_start_time = event.timestamp
            if event.timestamp > self.system_end_time:
                self.system_end_time = event.timestamp
            self.report["Details"]["job_start"] = self.system_start_time
            self.report["Details"]["job_end"] = self.system_end_time

        # iterate over framework metrics to get step information
        if framework_events is None:
            events = self.base_trial.get_framework_metrics(timestamp_start, timestamp_end)
        else:
            events = framework_events
        for event in events:
            if "Step:ModeKeys" in event.event_name:
                self.report["Datapoints"] += 1

                # store first and last step: used in profiler report to compute initialization, finalization and time spent in training loop
                if "first" not in self.report["Details"]["step_num"]:
                    self.report["Details"]["step_num"]["first"] = event.start_time
                else:
                    self.report["Details"]["step_num"]["last"] = event.start_time

                if int(event.event_args["step_num"]) % 500 == 0:
                    self.logger.info(
                        f"Step {event.event_args['step_num']} at {event.start_time} us"
                    )

        # rule triggers, if first step has not been seen yet and time since training job start is above the threshold
        if (
            self.last_timestamp - self.first_timestamp > self.threshold * 60000000
            and len(self.report["Details"]["step_num"]) == 0
        ):
            self.logger.info(
                f"Waited more than {self.threshold} minutes and training loop has still not started yet."
            )

            # record information for profiler report
            self.report["Violations"] += 1
            self.report["RuleTriggered"] += 1
            return True

        return False
