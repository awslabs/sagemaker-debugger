# First Party
# Standard Library
import json
import os
import pathlib
import shutil

from smdebug.exceptions import NoMoreProfilerData, RuleEvaluationConditionMet
from smdebug.profiler.analysis.rules.batch_size import BatchSize
from smdebug.profiler.analysis.rules.cpu_bottleneck import CPUBottleneck
from smdebug.profiler.analysis.rules.gpu_memory_increase import GPUMemoryIncrease
from smdebug.profiler.analysis.rules.io_bottleneck import IOBottleneck
from smdebug.profiler.analysis.rules.load_balancing import LoadBalancing
from smdebug.profiler.analysis.rules.low_gpu_utilization import LowGPUUtilization
from smdebug.profiler.analysis.rules.max_initialization_time import MaxInitializationTime
from smdebug.profiler.analysis.rules.overall_system_usage import OverallSystemUsage
from smdebug.profiler.analysis.rules.plot_visualizations.plot_visualizations import (
    PlotVisualizations,
)
from smdebug.profiler.analysis.rules.step_outlier import StepOutlier
from smdebug.rules.rule import Rule


class ProfilerReport(Rule):
    def __init__(
        self,
        base_trial,
        scan_interval_us=60 * 1000 * 1000,
        nb_path="/opt/ml/code/profiler_report.ipynb",
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
        :param scan_interval: interval with which timeline files are scanned. Default is 60000000.
        """
        super().__init__(base_trial)

        # Note that for ProfilerReport, all the following
        self.rules = [
            CPUBottleneck(base_trial),
            IOBottleneck(base_trial),
            LowGPUUtilization(base_trial),
            StepOutlier(base_trial),
            GPUMemoryIncrease(base_trial),
            BatchSize(base_trial),
            MaxInitializationTime(base_trial),
            LoadBalancing(base_trial),
            OverallSystemUsage(base_trial),
        ]
        self.last_timestamp = self.base_trial.first_timestamp
        self.scan_interval_us = scan_interval_us
        self.nb_path = nb_path
        # report_dir is a local directory path where we could save reports into and allow service to publish.
        report_dir = os.path.join(self.base_trial.output_dir, "profiler-reports")
        if report_dir and not os.path.exists(report_dir):
            pathlib.Path(report_dir).mkdir(parents=True, exist_ok=True)
        self.report_dir = report_dir

        # boolean flag indicating whether any rule condition has been met so far.
        self.rule_condition_met = False
        self.logger.info(
            "Output files of ProfilerReport Rule will be saved to {}".format(self.report_dir)
        )

    def invoke_at_step(self, step):
        pass

    def invoke(self, step):
        # iterate over timeline events
        current_timestamp = self.last_timestamp + self.scan_interval_us

        # Different from normal rule, when we reach the end of profiling, instead of throw NoMoreProfilerData
        # we will check whether any rule condition has been met and throw RuleEvaluationConditionMet
        try:
            self.base_trial.wait_for_data(current_timestamp, self.last_timestamp)
        except NoMoreProfilerData as e:
            if self.rule_condition_met:
                raise RuleEvaluationConditionMet(self.rule_name, step, end_of_rule=True)
            else:
                # End the training job with NoMoreProfilerData
                raise e

        rule_condition = self.invoke_for_timerange(self.last_timestamp, current_timestamp)
        self.rule_condition_met = self.rule_condition_met or rule_condition
        self.last_timestamp = current_timestamp

    def _generate_report(self, rule):
        report_name = rule.rule_name + ".json"
        temp_path = os.path.join(self.base_trial.temp_dir, report_name)
        target_path = os.path.join(self.report_dir, report_name)
        with open(temp_path, "w") as f:
            json.dump(rule.report, f)

        # Move the temp file to target path
        shutil.move(temp_path, target_path)

    def invoke_for_timerange(self, timestamp_start, timestamp_end):
        # TODO make sure below caches the data
        sys_events = self.base_trial.get_system_metrics(timestamp_start, timestamp_end)
        framework_events = self.base_trial.get_framework_metrics(timestamp_start, timestamp_end)

        is_condition_met: bool = False  # The boolean flag indicating whether any sub rule met evaluation condition.
        # get_system_metrics should be able to cache to local dir (.sagemaker-ignore in rule_output directory)
        # read from local dir, delete in memory at end of function
        for rule in self.rules:
            # TODO log invoking rule
            self.logger.info(
                f"Invoking rule:{rule.rule_name} for timestamp_start:{timestamp_start} to timestamp_end:{timestamp_end}"
            )
            try:
                rule_condition = rule.invoke_for_timerange(
                    timestamp_start, timestamp_end, sys_events, framework_events
                )
            except:
                self.logger.error(f"Error running rule {rule.name}")

            is_condition_met = is_condition_met or rule_condition
            if self.report_dir:
                # Only dump the report if the report directory is specified.
                self._generate_report(rule)
            # TODO finished rule invocation

        # As all sub rules processed, generate the report
        # This indicates the end of rule, before ending, generate a HTML report with sub rules.
        if os.path.exists(self.nb_path):
            try:
                rule = PlotVisualizations(
                    self.base_trial,
                    create_html=True,
                    nb_full_path=self.nb_path,
                    output_full_path=os.path.join(self.report_dir, "profiler-report.ipynb"),
                )
                rule._plot_visualization(last_found_step=0)
            except Exception as e:
                self.logger.error("Exception during HTML Report generation: {}".format(e))
        else:
            self.logger.error(
                "Missing profiler report notebook at {}. Skip HTML report generation".format(
                    self.nb_path
                )
            )

        return is_condition_met
