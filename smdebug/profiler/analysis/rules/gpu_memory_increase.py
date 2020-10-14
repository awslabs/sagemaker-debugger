# Third Party
# Standard Library
import shutil

import matplotlib.pyplot as plt
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
        self.last_movinge_average = 0
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

                # compute moving average
                if len(self.values[event.node_id][event.name]) > self.window:
                    current_movinge_average = np.mean(
                        self.values[event.node_id][event.name][-self.window :]
                    )

                    # check for memory increase
                    if self.last_movinge_average != 0:
                        diff = current_movinge_average - self.last_movinge_average

                        # rule triggers if moving average increased by more than pre-defined threshold
                        if diff > self.increase:
                            if len(self.values[event.node_id][event.name]) > self.patience:
                                self.logger.info(
                                    f"Current memory usage on GPU {event.name} on node {event.node_id} is: {self.values[event.node_id][event.name][-1]}%. Average memory increased by more than {diff}%"
                                )

                                # record information for profiler report
                                self.report["Violations"] += 1
                                self.report["RuleTriggered"] += 1
                                self.report["Details"][event.timestamp] = {
                                    "gpu_id": event.name,
                                    "node_id": event.node_id,
                                    "increase": diff,
                                    "memory": self.values[event.node_id][event.name][-1],
                                }

                                # create boxplot for profiler report
                                for node_id in self.values:
                                    fig, ax = plt.subplots()
                                    positions = np.arange(len(self.values[node_id]))
                                    plt.title(f"Boxplot for GPU memory on node {node_id}")
                                    plt.boxplot(
                                        list(self.values[node_id].values()), positions=positions
                                    )
                                    ax.set_xlabel("GPU")
                                    ax.set_ylabel("Memory")

                                    # output filename
                                    filename = node_id + "box_plot_gpu_memory.png"

                                    # save file
                                    try:
                                        plt.savefig(
                                            "/opt/ml/processing/outputs/.sagemaker-ignore/"
                                            + filename,
                                            bbox_inches="tight",
                                        )
                                        shutil.move(
                                            "/opt/ml/processing/outputs/.sagemaker-ignore/"
                                            + filename,
                                            "/opt/ml/processing/outputs/profiler-reports/"
                                            + filename,
                                        )
                                    except:
                                        self.logger.info("Error while saving file")

                                    plt.close()
                                return True
                    self.last_movinge_average = current_movinge_average

        # log current values
        for node_id in self.values:
            for key in self.values[node_id]:
                self.logger.info(
                    f"Current memory usage on GPU {key} on node {node_id} is: {self.values[node_id][key][-1]}%"
                )
        return False
