# Third Party
# Standard Library
import shutil

import matplotlib.pyplot as plt
import numpy as np

# First Party
from smdebug.exceptions import RuleEvaluationConditionMet
from smdebug.rules.rule import Rule


class LowGPUUtilization(Rule):
    def __init__(
        self,
        base_trial,
        threshold_p95=70,
        threshold_p5=10,
        window=500,
        patience=1000,
        scan_interval_us=60 * 1000 * 1000,
    ):
        """
        This rule helps to detect if GPU utilization is low or suffers from fluctuations. This is checked for each single GPU on each worker node.
        Rule returns True if 95th quantile is below threshold_p95 which indicates under-utilization.
        Rule returns true if 95th quantile is above threshold_p95 and 5th quantile is below threshold_p5 which indicates fluctuations.
        :param base_trial: the trial whose execution will invoke the rule
        :param threshold_p95: threshold for 95th quantile below which GPU is considered to be underutilized. Default is 70 percent.
        :param threshold_p5: threshold for 5th quantile. Default is 10 percent.
        :param window: number of past datapoints which are used to compute the quantiles.
        :param patience: How many values to record before checking for underutilization/fluctuations. During training initilization, GPU is likely at 0 percent, so Rule should not check for underutilization immediately. Default 1000.
        :param scan_interval_us: interval with which timeline files are scanned. Default is 60000000 us.
        """
        super().__init__(base_trial)
        self.threshold_p95 = threshold_p95
        self.threshold_p5 = threshold_p5
        self.window = window
        self.patience = patience
        self.scan_interval_us = scan_interval_us
        self.values = {}
        self.max_datapoints = 1000000
        self.last_timestamp = self.base_trial.first_timestamp
        self.report[
            "RuleParameters"
        ] = f"threshold_p95:{self.threshold_p95}\nthreshold_p5:{self.threshold_p5}\nwindow:{window}\npatience:{self.patience}"

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

    def record_violations(self, values, node_id, gpu_id):

        # record information for profiler report
        self.report["RuleTriggered"] += 1
        self.report["Violations"] += 1
        self.report["Datapoints"] = len(values)
        if node_id not in self.report["Details"]:
            self.report["Details"][node_id] = {}
        if gpu_id not in self.report["Details"][node_id]:
            self.report["Details"][node_id][gpu_id] = {}

        self.report["Details"][node_id][gpu_id] = {
            "gpu_95": np.quantile(values[-self.window :], 0.95),
            "gpu_5": np.quantile(values[-self.window :], 0.05),
        }

        # create box plot for gpu utilization
        fig, ax = plt.subplots()
        positions = np.arange(len(self.values[node_id]))
        plt.title(f"Boxplot for GPU utilization on node {node_id}")
        plt.boxplot(list(self.values[node_id].values()), positions=positions)
        ax.set_xlabel("GPU")
        ax.set_ylabel("Utilization")

        # output filename
        filename = node_id + "_" + "box_plot_gpu_utilization.png"

        # save file
        try:
            plt.savefig(
                "/opt/ml/processing/outputs/.sagemaker-ignore/" + filename, bbox_inches="tight"
            )
            shutil.move(
                "/opt/ml/processing/outputs/.sagemaker-ignore/" + filename,
                "/opt/ml/processing/outputs/profiler-reports/" + filename,
            )
        except:
            self.logger.info("Error while saving file")

        plt.close()

    def invoke_for_timerange(
        self, timestamp_start, timestamp_end, sys_events=None, framework_events=None
    ):

        # get system metrics for the current time interval
        if sys_events is None:
            events = self.base_trial.get_system_metrics(timestamp_start, timestamp_end)
        else:
            events = sys_events

        # get GPU utilization values per gpu id and per worker node
        for event in events:

            if event.dimension == "GPUUtilization":
                if event.node_id not in self.values:
                    self.values[event.node_id] = {}
                if event.name not in self.values[event.node_id]:
                    self.values[event.node_id][event.name] = []
                self.values[event.node_id][event.name].append(event.value)

        # iterate over values and find GPU underutilization and fluctuations
        for node_id in self.values:
            for gpu_id in self.values[node_id]:
                values = self.values[node_id][gpu_id]

                # record information for profiler report
                self.report["Datapoints"] = len(values)

                if len(values) > self.max_datapoints:
                    self.reset()

                if len(values) > self.patience:

                    # GPU not used
                    if np.max(values) == 0:
                        self.logger.info(f"{gpu_id} of node-id {node_id} is not used.")

                    # low GPU utilization
                    if np.quantile(values[-self.window :], 0.95) < self.threshold_p95:
                        self.logger.info(
                            f"{gpu_id} utilization of node-id {node_id}: 95th quantile is {np.quantile(values[-self.window:], 0.95)}% and below {self.threshold_p95}%"
                        )
                        self.record_violations(values, node_id, gpu_id)
                        return True

                    # GPU fluctuations
                    elif (
                        np.quantile(values[-self.window :], 0.95) > self.threshold_p95
                        and np.quantile(values[-self.window :], 0.05) < self.threshold_p5
                    ):
                        self.logger.info(
                            f"{gpu_id} utilization of node-id {node_id}: 95th quantile is {np.quantile(values[-self.window:], 0.95)}% and above threshold_p95 {self.threshold_p95} - 5th is {np.quantile(values, 0.05)}% and below threshold_p5 {self.threshold_p5}"
                        )
                        self.record_violations(values, node_id, gpu_id)
                        return True
                    else:
                        self.logger.info(
                            f"{gpu_id} utilization of node-id {node_id}: 95th quantile of GPU utilization is {np.quantile(values[-self.window:], 0.95)}% - 5th is is {np.quantile(values[-self.window:], 0.05)}%"
                        )
        return False
