# First Party
# Third Party
import numpy as np
from scipy import stats

from smdebug.exceptions import RuleEvaluationConditionMet
from smdebug.rules.rule import Rule


class LoadBalancing(Rule):
    def __init__(self, base_trial, threshold=0.2, patience=1000, scan_interval_us=60 * 1000 * 1000):
        """
        This rule helps to detect issues in workload balancing between multiple GPUs.
        It computes a histogram of utilization per GPU and measures the distance between those histograms.
        If the histogram exceeds a pre-defined threshold then rule triggers.
        :param threshold: difference between 2 histograms 0.2
        :param patience: how many values to record before checking for loadbalancing issues
        :param base_trial: the trial whose execution will invoke the rule
        :param scan_interval_us: interval with which timeline files are scanned. Default is 60000000 us.
        """
        super().__init__(base_trial)
        self.threshold = threshold
        self.scan_interval_us = scan_interval_us
        self.last_timestamp = self.base_trial.first_timestamp
        self.patience = patience
        self.gpus = {}
        self.histogram = {}
        self.max_datapoints = 1000000
        self.report["RuleParameters"] = f"threshold:{self.threshold}\npatience:{self.patience}"

    def invoke_at_step(self, step):
        pass

    def reset(self):
        self.gpus = {}
        self.histogram = {}

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

        # get usage per node_id and per gpu_id
        if sys_events is None:
            events = self.base_trial.get_system_metrics(timestamp_start, timestamp_end)
        else:
            events = sys_events
        for event in events:

            if event.dimension == "GPUUtilization":
                if event.node_id not in self.gpus:
                    self.gpus[event.node_id] = {}
                    self.histogram[event.node_id] = {}
                if event.name not in self.gpus[event.node_id]:
                    self.gpus[event.node_id][event.name] = []
                    self.histogram[event.node_id][event.name] = []
                self.gpus[event.node_id][event.name].append(event.value)

                # record number of processed event for profiler report
                nvalues = len(self.gpus[event.node_id][event.name])
                self.report["Datapoints"] = nvalues

                # reset dictionaries
                if nvalues > self.max_datapoints:
                    self.reset()

        # compute histogram of usage per node_id and per gpu_id
        for node_id in self.gpus:
            for gpu_id in self.gpus[node_id]:
                if len(self.gpus[node_id][gpu_id]) > self.patience:
                    values = self.gpus[node_id][gpu_id]
                    bins = np.arange(0, 100, 2)
                    probs, binedges = np.histogram(values, bins=bins)
                    self.histogram[node_id][gpu_id] = probs

        # list of node ids
        node_ids = list(self.histogram.keys())
        len_node_ids = len(node_ids)

        # iterate over all possible combinations and compute distance between histograms
        for node_id1 in range(len_node_ids):
            for node_id2 in range(node_id1, len_node_ids):

                # get keys
                node1 = node_ids[node_id1]
                node2 = node_ids[node_id2]

                # list of gpu ids
                gpu_ids = list(self.histogram[node1].keys())
                len_gpu_ids = len(gpu_ids)

                if len(self.histogram[node1][gpu_ids[0]]) > 0:

                    for gpu_id1 in range(len_gpu_ids):
                        for gpu_id2 in range(gpu_id1, len_gpu_ids):

                            # get keys
                            gpu1 = gpu_ids[gpu_id1]
                            gpu2 = gpu_ids[gpu_id2]
                            if gpu1 != gpu2:
                                # compute distance between histograms
                                m = (self.histogram[node1][gpu1] + self.histogram[node2][gpu2]) / 2
                                divergence = (
                                    stats.entropy(self.histogram[node1][gpu1], m)
                                    + stats.entropy(self.histogram[node2][gpu2], m)
                                ) / 2
                                distance = np.sqrt(divergence)

                                # compare distance with threshold
                                if distance > self.threshold:
                                    self.logger.info(
                                        f"Workload on node {node_id} between GPUs {gpu1} and {gpu2} differs by {distance} which is above the threshold {self.threshold}"
                                    )

                                    # record information for profiler report
                                    self.report["Violations"] += 1
                                    self.report["RuleTriggered"] += 1

                                    if node1 not in self.report["Details"]:
                                        self.report["Details"][node1] = {
                                            "workloads": {},
                                            "distances": {},
                                        }
                                    self.report["Details"][node1]["workloads"][
                                        gpu1
                                    ] = self.histogram[node1][gpu1].tolist()
                                    self.report["Details"][node1]["workloads"][
                                        gpu2
                                    ] = self.histogram[node1][gpu2].tolist()
                                    if gpu1 not in self.report["Details"][node1]["distances"]:
                                        self.report["Details"][node1]["distances"][gpu1] = {}
                                    self.report["Details"][node1]["distances"][gpu1][
                                        gpu2
                                    ] = distance

                                else:
                                    self.logger.info(
                                        f"Workload on node {node1} between GPUs {gpu1} and {gpu2} differs by {distance}"
                                    )
        if self.report["RuleTriggered"] > 0:
            return True
        return False
