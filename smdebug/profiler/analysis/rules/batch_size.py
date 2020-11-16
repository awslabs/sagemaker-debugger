# First Party
# Third Party
import numpy as np

from smdebug.exceptions import RuleEvaluationConditionMet
from smdebug.rules.rule import Rule


class BatchSize(Rule):
    def __init__(
        self,
        base_trial,
        cpu_threshold_p95=70,
        gpu_threshold_p95=70,
        gpu_memory_threshold_p95=70,
        patience=1000,
        window=500,
        scan_interval_us=60 * 1000 * 1000,
    ):
        """
        This rule helps to detect if GPU is underulitized because of the batch size being too small.
        To detect this the rule analyzes the average GPU memory footprint, CPU and GPU utilization.
        If utilization on CPU, GPU and memory footprint is on average low , it may indicate that user
        can either run on a smaller instance type or that batch size could be increased. This analysis does not
        work for frameworks that heavily over-allocate memory. Increasing batch size could potentially lead to
        a processing/dataloading bottleneck, because more data needs to be pre-processed in each iteration.

        :param base_trial: the trial whose execution will invoke the rule
        :param cpu_threshold_p95: defines the threshold for 95th quantile of CPU utilization.Default is 70%.
        :param gpu_threshold_p95: defines the threshold for 95th quantile of GPU utilization.Default is 70%.
        :param gpu_memory_threshold_p95: defines the threshold for 95th quantile of GPU memory utilization.Default is 70%.
        :param patience: defines how many datapoints to capture before Rule runs the first evluation. Default 100
        :param window: window size for computing quantiles.
        :param scan_interval_us: interval with which timeline files are scanned. Default is 60000000 us.
        """
        super().__init__(base_trial)
        self.cpu_threshold_p95 = cpu_threshold_p95
        self.gpu_threshold_p95 = gpu_threshold_p95
        self.gpu_memory_threshold_p95 = gpu_memory_threshold_p95
        self.patience = patience
        self.window = window
        self.scan_interval_us = scan_interval_us
        self.gpu_memory = {}
        self.gpu_utilization = {}
        self.cpu_utilization = {}
        self.core_ids = {}
        self.max_datapoints = 1000000
        self.last_timestamp = self.base_trial.first_timestamp
        self.report[
            "RuleParameters"
        ] = f"cpu_threshold_p95:{self.cpu_threshold_p95}\ngpu_threshold_p95:{self.gpu_threshold_p95}\ngpu_memory_threshold_p95:{self.gpu_memory_threshold_p95}\npatience:{self.patience}\nwindow:{self.window}"

    def invoke_at_step(self, step):
        pass

    def reset(self):
        self.gpu_memory = {}
        self.gpu_utilization = {}
        self.cpu_utilization = {}

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

        total_cpu = {}

        # iterate over events
        for event in events:

            # get gpu memory utilization values per node and gpu id
            if event.dimension == "GPUMemoryUtilization":
                if event.node_id not in self.gpu_memory:
                    self.gpu_memory[event.node_id] = {}
                if event.name not in self.gpu_memory[event.node_id]:
                    self.gpu_memory[event.node_id][event.name] = []
                self.gpu_memory[event.node_id][event.name].append(event.value)
                # reset dictionaries
                if len(self.gpu_memory) > self.max_datapoints:
                    self.reset()

            # get gpu utilization values per node and gpu id
            if event.dimension == "GPUUtilization":
                if event.node_id not in self.gpu_utilization:
                    self.gpu_utilization[event.node_id] = {}
                if event.name not in self.gpu_utilization[event.node_id]:
                    self.gpu_utilization[event.node_id][event.name] = []
                self.gpu_utilization[event.node_id][event.name].append(event.value)

            # get cpu utilization values per node
            if event.dimension == "CPUUtilization":
                if event.name not in self.core_ids:
                    self.core_ids[event.name] = 0
                if event.node_id not in total_cpu:
                    total_cpu[event.node_id] = {}
                if event.timestamp not in total_cpu[event.node_id]:
                    total_cpu[event.node_id][event.timestamp] = 0
                total_cpu[event.node_id][event.timestamp] += event.value

        # compute cpu total
        for node_id in total_cpu:

            if node_id not in self.cpu_utilization:
                self.cpu_utilization[node_id] = []

            for timestamp in total_cpu[node_id]:
                self.cpu_utilization[node_id].append(
                    total_cpu[node_id][timestamp] / len(self.core_ids)
                )

        # iterate over values and compare thresholds
        for node_id in self.cpu_utilization:

            # record number of datapoints for profiler report
            nvalues = len(self.cpu_utilization[node_id])
            self.report["Datapoints"] = nvalues

            # run rule evaluation if enough datapoints have been loaded
            if nvalues > self.window and nvalues > self.patience:

                # check for CPU underutilization
                cpu_p95 = np.quantile(self.cpu_utilization[node_id][-self.window :], 0.95)
                if cpu_p95 < self.cpu_threshold_p95:

                    # iterate over utilization per GPU
                    if self.gpu_utilization:
                        for gpu_id in self.gpu_utilization[node_id]:
                            if len(self.gpu_utilization[node_id][gpu_id]) > self.window:
                                # check for GPU underutilization and memory underutilization
                                gpu_p95 = np.quantile(
                                    self.gpu_utilization[node_id][gpu_id][-self.window :], 0.95
                                )

                                gpu_memory_p95 = np.quantile(
                                    self.gpu_memory[node_id][gpu_id][-self.window :], 0.95
                                )

                                if (
                                    gpu_p95 < self.gpu_threshold_p95
                                    and gpu_memory_p95 < self.gpu_memory_threshold_p95
                                ):
                                    self.logger.info(
                                        f"Node {node_id} GPU {gpu_id} utilization p95 is {gpu_p95}% which is below the threshold of {self.gpu_threshold_p95}% and memory p95 is {gpu_memory_p95}% which is below the threshold of {self.gpu_memory_threshold_p95}%. Overall CPU utilization p95 is {cpu_p95}% which is below the threshold of {self.cpu_threshold_p95}%."
                                    )
                                    # record information for profiler report
                                    self.report["RuleTriggered"] += 1
                                    self.report["Violations"] += 1
                                    if node_id not in self.report["Details"]:
                                        self.report["Details"][node_id] = {}

                                    self.report["Details"][node_id]["cpu"] = {
                                        "p25": np.quantile(self.cpu_utilization[node_id], 0.25),
                                        "p50": np.quantile(self.cpu_utilization[node_id], 0.50),
                                        "p75": np.quantile(self.cpu_utilization[node_id], 0.75),
                                        "p95": np.quantile(self.cpu_utilization[node_id], 0.95),
                                    }
                                    iqr = (
                                        self.report["Details"][node_id]["cpu"]["p75"]
                                        - self.report["Details"][node_id]["cpu"]["p25"]
                                    )
                                    upper = (
                                        self.report["Details"][node_id]["cpu"]["p75"] + 1.5 * iqr
                                    )
                                    lower = (
                                        self.report["Details"][node_id]["cpu"]["p25"] - 1.5 * iqr
                                    )
                                    self.report["Details"][node_id]["cpu"]["upper"] = min(
                                        upper, np.quantile(self.cpu_utilization[node_id], 1)
                                    )
                                    self.report["Details"][node_id]["cpu"]["lower"] = max(
                                        lower, np.quantile(self.cpu_utilization[node_id], 0.0)
                                    )

                                    self.report["Details"][node_id][gpu_id] = {
                                        "p25": np.quantile(
                                            self.gpu_utilization[node_id][gpu_id], 0.25
                                        ),
                                        "p50": np.quantile(
                                            self.gpu_utilization[node_id][gpu_id], 0.50
                                        ),
                                        "p75": np.quantile(
                                            self.gpu_utilization[node_id][gpu_id], 0.75
                                        ),
                                        "p95": np.quantile(
                                            self.gpu_utilization[node_id][gpu_id], 0.95
                                        ),
                                    }
                                    iqr = (
                                        self.report["Details"][node_id][gpu_id]["p75"]
                                        - self.report["Details"][node_id][gpu_id]["p25"]
                                    )
                                    upper = (
                                        self.report["Details"][node_id][gpu_id]["p75"] + 1.5 * iqr
                                    )
                                    lower = (
                                        self.report["Details"][node_id][gpu_id]["p25"] - 1.5 * iqr
                                    )
                                    self.report["Details"][node_id][gpu_id]["upper"] = min(
                                        upper, np.quantile(self.gpu_utilization[node_id][gpu_id], 1)
                                    )
                                    self.report["Details"][node_id][gpu_id]["lower"] = max(
                                        lower,
                                        np.quantile(self.gpu_utilization[node_id][gpu_id], 0.0),
                                    )
                                    key = f"{gpu_id}_memory"
                                    self.report["Details"][node_id][key] = {
                                        "p25": np.quantile(self.gpu_memory[node_id][gpu_id], 0.25),
                                        "p50": np.quantile(self.gpu_memory[node_id][gpu_id], 0.50),
                                        "p75": np.quantile(self.gpu_memory[node_id][gpu_id], 0.75),
                                        "p95": np.quantile(self.gpu_memory[node_id][gpu_id], 0.95),
                                    }
                                    iqr = (
                                        self.report["Details"][node_id][key]["p75"]
                                        - self.report["Details"][node_id][gpu_id]["p25"]
                                    )
                                    upper = self.report["Details"][node_id][key]["p75"] + 1.5 * iqr
                                    lower = self.report["Details"][node_id][key]["p25"] - 1.5 * iqr
                                    self.report["Details"][node_id][key]["upper"] = min(
                                        upper, np.quantile(self.gpu_memory[node_id][gpu_id], 1)
                                    )
                                    self.report["Details"][node_id][key]["lower"] = max(
                                        lower, np.quantile(self.gpu_memory[node_id][gpu_id], 0.0)
                                    )
                                    self.report["Details"]["last_timestamp"] = self.last_timestamp

                else:
                    self.logger.info(f"Node {node_id} Overall CPU utilization p95 is {cpu_p95}% ")

        if self.report["RuleTriggered"] > 0:
            return True

        return False
