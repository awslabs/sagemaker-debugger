# Third Party
import numpy as np

# First Party
from smdebug.exceptions import RuleEvaluationConditionMet
from smdebug.rules.rule import Rule


class OverallSystemUsage(Rule):
    def __init__(self, base_trial, scan_interval_us=60 * 1000 * 1000):
        """
        This rule measures overall system usage per worker node. The rule currently only aggregates values per node
        and computes their percentiles. The rule does currently not take any threshold parameters into account
        nor can it trigger. The reason behind that is that other rules already cover cases such as underutilization and
        they do it at a more fine-grained level e.g. per GPU. We may change this in the future.
        :param base_trial: the trial whose execution will invoke the rule
        :param scan_interval_us: interval with which timeline files are scanned. Default is 60000000 us.
        """
        super().__init__(base_trial)
        self.scan_interval_us = scan_interval_us
        self.last_timestamp = self.base_trial.first_timestamp
        self.previous_timestamp = 0

        self.gpu_memory = {}
        self.cpu_memory = {}
        self.gpu = {}
        self.cpu = {}
        self.network = {}
        self.io = {}

        self.gpu_ids = {}
        self.core_ids = {}

        self.datapoints = 0
        self.max_datapoints = 1000000
        self.report["RuleParameters"] = f""

    def invoke_at_step(self, step):
        pass

    def reset(self):
        self.gpu_memory = {}
        self.cpu_memory = {}
        self.gpu = {}
        self.cpu = {}
        self.network = {}
        self.io = {}

    def invoke(self, step):
        # iterate over timeline events
        current_timestamp = self.last_timestamp + self.scan_interval_us
        self.base_trial.wait_for_data(current_timestamp, self.last_timestamp)
        rule_condition = self.invoke_for_timerange(self.last_timestamp, current_timestamp)
        self.last_timestamp = current_timestamp
        if rule_condition:
            raise RuleEvaluationConditionMet(self.rule_name, step)

    def record_metrics(self, values, name, n):

        if name not in self.report["Details"]:
            self.report["Details"][name] = {}

        for node_id in values:
            if node_id not in self.report["Details"][name]:
                self.report["Details"][name][node_id] = {}

            vals = list(values[node_id].values())
            max_value = np.max(vals) / n
            max_value = max_value if max_value > 0 else 0
            p99 = np.quantile(vals, 0.99) / n
            p99 = p99 if p99 > 0 else 0
            p95 = np.quantile(vals, 0.95) / n
            p95 = p95 if p95 > 0 else 0
            p50 = np.quantile(vals, 0.50) / n
            p50 = p50 if p50 > 0 else 0
            min_value = np.min(vals) / n
            min_value = min_value if min_value > 0 else 0

            self.report["Details"][name][node_id] = {
                "max": round(max_value, 2),
                "p99": round(p99, 2),
                "p95": round(p95, 2),
                "p50": round(p50, 2),
                "min": round(min_value, 2),
            }

    def invoke_for_timerange(
        self, timestamp_start, timestamp_end, sys_events=None, framework_events=None
    ):
        if sys_events is None:
            events = self.base_trial.get_system_metrics(timestamp_start, timestamp_end)
        else:
            events = sys_events

        # iterate over events and compute step duration
        for event in events:

            # total GPU memory per worker node
            if event.dimension == "GPUMemoryUtilization":
                if event.node_id not in self.gpu_memory:
                    self.gpu_memory[event.node_id] = {}
                if event.timestamp not in self.gpu_memory[event.node_id]:
                    self.gpu_memory[event.node_id][event.timestamp] = 0
                self.gpu_memory[event.node_id][event.timestamp] += event.value

            # total GPU utilization per worker node
            if event.dimension == "GPUUtilization":
                if event.name not in self.gpu_ids:
                    self.gpu_ids[event.name] = 0
                if event.node_id not in self.gpu:
                    self.gpu[event.node_id] = {}
                if event.timestamp not in self.gpu[event.node_id]:
                    self.gpu[event.node_id][event.timestamp] = 0
                self.gpu[event.node_id][event.timestamp] += event.value

            # total CPU memory per worker node
            if event.name == "MemoryUsedPercent":
                if event.node_id not in self.cpu_memory:
                    self.cpu_memory[event.node_id] = {}
                if event.timestamp not in self.cpu_memory[event.node_id]:
                    self.cpu_memory[event.node_id][event.timestamp] = 0
                self.cpu_memory[event.node_id][event.timestamp] += event.value

            # total CPU utilization per worker node
            if event.dimension == "CPUUtilization":
                if event.name not in self.core_ids:
                    self.core_ids[event.name] = 0
                if event.node_id not in self.cpu:
                    self.cpu[event.node_id] = {}
                if event.timestamp not in self.cpu[event.node_id]:
                    self.cpu[event.node_id][event.timestamp] = 0
                self.cpu[event.node_id][event.timestamp] += event.value

            # total network utilization per worker node
            if event.dimension == "Algorithm":
                if event.node_id not in self.network:
                    self.network[event.node_id] = {}
                if event.timestamp not in self.network[event.node_id]:
                    self.network[event.node_id][event.timestamp] = 0
                self.network[event.node_id][event.timestamp] += event.value

            # total IO wait time per worker node
            if event.dimension == "I/OWaitPercentage":
                if event.node_id not in self.io:
                    self.io[event.node_id] = {}
                if event.timestamp not in self.io[event.node_id]:
                    self.io[event.node_id][event.timestamp] = 0
                self.io[event.node_id][event.timestamp] += event.value

        # record statistics for profiler report
        if len(self.cpu.keys()) > 0:
            nvalues = len(self.cpu[list(self.cpu.keys())[0]])
            self.report["Datapoints"] = nvalues
            # reset dictionaries
            if nvalues > self.max_datapoints:
                self.reset()

        self.record_metrics(self.network, "Network", 1)
        self.record_metrics(self.gpu, "GPU", len(self.gpu_ids))
        self.record_metrics(self.cpu, "CPU", len(self.core_ids))
        self.record_metrics(self.cpu_memory, "CPU memory", 1)
        self.record_metrics(self.gpu_memory, "GPU memory", len(self.gpu_ids))
        self.record_metrics(self.io, "I/O", len(self.core_ids))

        return False
