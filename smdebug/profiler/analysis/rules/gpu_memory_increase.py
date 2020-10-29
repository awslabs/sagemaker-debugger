# Third Party
import numpy as np

# First Party
from smdebug.exceptions import RuleEvaluationConditionMet
from smdebug.rules.rule import Rule


class GPUMemoryIncrease(Rule):
    def __init__(
        self, base_trial, increase=5, patience=1000, window=10, scan_interval_us=60 * 1000 * 1000
    ):
        """
        This rule helps to detect large increase in memory usage on GPUs. The rule computes the moving average of continous datapoints and compares it against the moving average of previous iteration.
        :param base_trial: the trial whose execution will invoke the rule
        :param increase: defines the threshold for absolute memory increase.Default is 5%. So if moving average increase from 5% to 6%, the rule will fire.
        :param patience: defines how many continous datapoints to capture before Rule runs the first evluation. Default is 1000
        :param window: window size for computing moving average of continous datapoints
        :param scan_interval_us: interval with which timeline files are scanned. Default is 60000000 us.
        """
        super().__init__(base_trial)
        self.increase = increase
        self.patience = patience
        self.window = window
        self.scan_interval_us = scan_interval_us

        self.last_timestamp = self.base_trial.first_timestamp
        self.values = {}
        self.max_datapoints = 1000000
        self.report[
            "RuleParameters"
        ] = f"increase:{self.increase}\npatience:{self.patience}\nwindow:{self.window}"

    def invoke_at_step(self, step):
        pass

    def reset(self):
        self.values = {}

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
        if sys_events is None:
            events = self.base_trial.get_system_metrics(timestamp_start, timestamp_end)
        else:
            events = sys_events

        # iterate over events and get GPU memory usage values
        for event in events:

            if event.dimension == "GPUMemoryUtilization":

                # record node id
                if event.node_id not in self.values:
                    self.values[event.node_id] = {}

                # record gpu id
                if event.name not in self.values[event.node_id]:
                    self.values[event.node_id][event.name] = []
                self.values[event.node_id][event.name].append(event.value)

                # record number of datapoints for profiler report
                nvalues = len(self.values[event.node_id][event.name])
                self.report["Datapoints"] = nvalues

                if nvalues > self.max_datapoints:
                    self.reset()

                values = self.values[event.node_id][event.name]

                # compute moving average
                if len(values) > self.window + 1:

                    # check for memory increase
                    diff = np.mean(values[-self.window :]) - np.mean(values[-self.window - 1 : -2])

                    # rule triggers if moving average increased by more than pre-defined threshold
                    if diff > self.increase:
                        if len(values) > self.patience:
                            self.logger.info(
                                f"Current memory usage on GPU {event.name} on node {event.node_id} is: {values[-1]}%. Average memory increased by more than {diff}%"
                            )

                            # record information for profiler report
                            self.report["Violations"] += 1
                            self.report["RuleTriggered"] += 1

                            if event.node_id not in self.report["Details"]:
                                self.report["Details"][event.node_id] = {}

                            # record data for box plot
                            self.report["Details"][event.node_id][event.name] = {
                                "increase": diff,
                                "gpu_max": np.max(values),
                                "p05": np.quantile(values, 0.05),
                                "p25": np.quantile(values, 0.25),
                                "p50": np.quantile(values, 0.50),
                                "p75": np.quantile(values, 0.75),
                                "p95": np.quantile(values, 0.95),
                            }
                            iqr = (
                                self.report["Details"][event.node_id][event.name]["p75"]
                                - self.report["Details"][event.node_id][event.name]["p25"]
                            )
                            upper = (
                                self.report["Details"][event.node_id][event.name]["p75"] + 1.5 * iqr
                            )
                            lower = (
                                self.report["Details"][event.node_id][event.name]["p25"] - 1.5 * iqr
                            )

                            self.report["Details"][event.node_id][event.name]["upper"] = min(
                                upper, np.quantile(values, 1)
                            )
                            self.report["Details"][event.node_id][event.name]["lower"] = max(
                                lower, np.quantile(values, 0.0)
                            )

                            self.report["Details"]["last_timestamp"] = self.last_timestamp

        # log current values
        for node_id in self.values:
            for gpu_id in self.values[node_id]:
                values = self.values[node_id][gpu_id]
                self.logger.info(
                    f"Current memory usage on GPU {gpu_id} on node {node_id} is: {values[-1]}%"
                )

        if self.report["RuleTriggered"] > 0:
            return True

        return False
