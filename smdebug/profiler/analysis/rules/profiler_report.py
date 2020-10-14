# First Party
from smdebug.exceptions import RuleEvaluationConditionMet
from smdebug.rules.rule import Rule


class ProfilerReport(Rule):
    def __init__(self, base_trial, rules=[], scan_interval_us=60 * 1000 * 1000):
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
        :param scan_interval: interval with which timeline files are scanned. Default is 60000000.
        """
        super().__init__(base_trial)
        if len(rules) == 0:
            raise Exception("You must specify at least one rule to run for profiler report.")

        self.rules = rules
        self.last_timestamp = self.base_trial.first_timestamp
        self.scan_interval_us = scan_interval_us

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

    def invoke_for_timerange(self, timestamp_start, timestamp_end):
        # TODO make sure below caches the data
        sys_events = self.base_trial.get_system_metrics(timestamp_start, timestamp_end)
        framework_events = self.base_trial.get_framework_metrics(timestamp_start, timestamp_end)

        # get_system_metrics should be able to cache to local dir (.sagemaker-ignore in rule_output directory)
        # read from local dir, delete in memory at end of function
        for rule in self.rules:
            # TODO log invoking rule
            self.logger.info(
                f"Invoking rule:{rule.rule_name} for timestamp_start:{timestamp_start} to timestamp_end:{timestamp_end}"
            )
            rule.invoke_for_timerange(timestamp_start, timestamp_end, sys_events, framework_events)
            # TODO finished rule invocation
        return False
