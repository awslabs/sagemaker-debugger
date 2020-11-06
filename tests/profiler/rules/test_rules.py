# Third Party
import pytest

# First Party
from smdebug.exceptions import (
    NoMoreProfilerData,
    RuleEvaluationConditionMet,
    StepUnavailable,
    TensorUnavailable,
    TensorUnavailableForStep,
)
from smdebug.profiler.analysis.rules.batch_size import BatchSize
from smdebug.profiler.analysis.rules.cpu_bottleneck import CPUBottleneck
from smdebug.profiler.analysis.rules.gpu_memory_increase import GPUMemoryIncrease
from smdebug.profiler.analysis.rules.io_bottleneck import IOBottleneck
from smdebug.profiler.analysis.rules.load_balancing import LoadBalancing
from smdebug.profiler.analysis.rules.low_gpu_utilization import LowGPUUtilization
from smdebug.profiler.analysis.rules.max_initialization_time import MaxInitializationTime
from smdebug.profiler.analysis.rules.overall_system_usage import OverallSystemUsage
from smdebug.profiler.analysis.rules.profiler_report import ProfilerReport
from smdebug.profiler.analysis.rules.step_outlier import StepOutlier
from smdebug.trials import create_trial


def invoke_rule(rule_obj, start_step=0, end_step=None):
    """
    This will serve as a rule invoker for following tests. Different from normal invoker, this invoker
    will return several value indicating the rule status:
    - ending_timestamp: timestamp where NoMoreProfilerData is thrown
    - ending_step: step num where NoMoreProfilerData is thrown
    - condition_met_count: the number of times where RuleEvaluationConditionMet is thrown
    - unavailability_count: the number of times where TensorUnavailableForStep, StepUnavailable
      or TensorUnavailable is thrown.
    """
    ending_step = ending_timestamp = None
    condition_met_count = unavailability_count = 0
    step = start_step if start_step is not None else 0
    while (end_step is None) or (step < end_step):
        try:
            rule_obj.invoke(step)
        except (TensorUnavailableForStep, StepUnavailable, TensorUnavailable):
            unavailability_count += 1
        except RuleEvaluationConditionMet:
            condition_met_count += 1
        except NoMoreProfilerData as e:
            ending_step = step
            ending_timestamp = e.timestamp
            break
        step += 1
    return ending_step, ending_timestamp, condition_met_count, unavailability_count


@pytest.mark.slow
def test_gpu_usage_rule():
    bucket_name = "s3://smdebug-testing/resources/tf2_detailed_profile/profiler-output"
    trial = create_trial(bucket_name, profiler=True, output_dir="/tmp")
    rule = LowGPUUtilization(trial)

    ending_step, ending_timestamp, condition_met_count, unavailability_count = invoke_rule(rule)
    assert ending_step == 3
    assert ending_timestamp == 1596668460000000
    assert condition_met_count == 0
    assert unavailability_count == 0
    report = rule.report
    assert report["Datapoints"] == 320
    assert report["Violations"] == 0
    assert report["RuleTriggered"] == 0

    rule = LowGPUUtilization(trial, threshold_p95=0, threshold_p5=0, patience=0)
    ending_step, ending_timestamp, condition_met_count, unavailability_count = invoke_rule(rule)
    assert ending_step == 3
    assert ending_timestamp == 1596668460000000
    assert condition_met_count == 0
    assert unavailability_count == 0
    report = rule.report
    assert report["Datapoints"] == 320
    assert report["Violations"] == 0
    assert report["RuleTriggered"] == 0


@pytest.mark.slow
def test_cpu_bottleneck_rule():
    bucket_name = "s3://smdebug-testing/resources/tf2_detailed_profile/profiler-output"
    trial = create_trial(bucket_name, profiler=True, output_dir="/tmp")
    rule = CPUBottleneck(trial)
    ending_step, ending_timestamp, condition_met_count, unavailability_count = invoke_rule(rule)
    assert ending_step == 3
    assert ending_timestamp == 1596668460000000
    assert condition_met_count == 0
    assert unavailability_count == 0
    report = rule.report
    assert report["Datapoints"] == 320
    assert report["Violations"] == 105
    assert report["RuleTriggered"] == 0
    assert len(report["Details"]["bottlenecks"]) == 105

    rule = CPUBottleneck(trial, gpu_threshold=100, cpu_threshold=0, patience=0)
    ending_step, ending_timestamp, condition_met_count, unavailability_count = invoke_rule(rule)
    assert ending_step == 3
    assert ending_timestamp == 1596668460000000
    assert condition_met_count == 3
    assert unavailability_count == 0
    report = rule.report
    assert report["Datapoints"] == 320
    assert report["Violations"] == 319
    assert report["RuleTriggered"] == 3
    assert len(report["Details"]["bottlenecks"]) == 319


@pytest.mark.slow
def test_step_outlier_rule():
    bucket_name = "s3://smdebug-testing/resources/tf2_detailed_profile/profiler-output"
    trial = create_trial(bucket_name, profiler=True, output_dir="/tmp")
    rule = StepOutlier(trial, n_outliers=1)
    ending_step, ending_timestamp, condition_met_count, unavailability_count = invoke_rule(rule)
    assert ending_step == 3
    assert ending_timestamp == 1596668460000000
    assert condition_met_count == 0
    assert unavailability_count == 0
    report = rule.report
    assert report["Datapoints"] == 0
    assert report["Violations"] == 0
    assert report["RuleTriggered"] == 0

    rule = StepOutlier(trial, stddev=100, n_outliers=100)
    ending_step, ending_timestamp, condition_met_count, unavailability_count = invoke_rule(rule)
    assert ending_step == 3
    assert ending_timestamp == 1596668460000000
    assert condition_met_count == 0
    assert unavailability_count == 0
    report = rule.report
    assert report["Datapoints"] == 0
    assert report["Violations"] == 0
    assert report["RuleTriggered"] == 0


@pytest.mark.slow
def test_gpu_memory_increase_rule():
    bucket_name = "s3://smdebug-testing/resources/tf2_detailed_profile_0930/profiler-output"
    trial = create_trial(bucket_name, profiler=True, output_dir="/tmp")
    rule = GPUMemoryIncrease(trial)
    ending_step, ending_timestamp, condition_met_count, unavailability_count = invoke_rule(rule)
    assert ending_step == 6
    assert ending_timestamp == 1600942860000000
    assert condition_met_count == 0
    assert unavailability_count == 0
    report = rule.report
    assert report["Datapoints"] == 662
    assert report["Violations"] == 0
    assert report["RuleTriggered"] == 0

    rule = GPUMemoryIncrease(trial, increase=0, patience=0)
    ending_step, ending_timestamp, condition_met_count, unavailability_count = invoke_rule(rule)
    assert ending_step == 6
    assert ending_timestamp == 1600942860000000
    assert condition_met_count == 1
    assert unavailability_count == 0
    report = rule.report
    assert report["Datapoints"] == 662
    assert report["Violations"] == 148
    assert report["RuleTriggered"] == 148
    assert report["Details"]["algo-1"]["gpu0"]["gpu_max"] == 5.0
    assert report["Details"]["algo-1"]["gpu6"]["increase"] == 0.022222222222222143


@pytest.mark.slow
def test_batch_size_rule():
    bucket_name = "s3://smdebug-testing/resources/tf2_detailed_profile/profiler-output"
    trial = create_trial(bucket_name, profiler=True, output_dir="/tmp")
    rule = BatchSize(trial)
    ending_step, ending_timestamp, condition_met_count, unavailability_count = invoke_rule(rule)
    assert ending_step == 3
    assert ending_timestamp == 1596668460000000
    assert condition_met_count == 0
    assert unavailability_count == 0
    report = rule.report
    assert report["Datapoints"] == 319
    assert report["Violations"] == 0
    assert report["RuleTriggered"] == 0

    rule = BatchSize(
        trial,
        cpu_threshold_p95=100,
        gpu_threshold_p95=100,
        gpu_memory_threshold_p95=100,
        patience=0,
        window=10,
    )
    ending_step, ending_timestamp, condition_met_count, unavailability_count = invoke_rule(rule)
    assert ending_step == 3
    assert ending_timestamp == 1596668460000000
    assert condition_met_count == 3
    assert unavailability_count == 0
    report = rule.report
    assert report["Datapoints"] == 319
    assert report["Violations"] == 48
    assert report["RuleTriggered"] == 48
    assert report["Details"]["algo-1"]["cpu"]["lower"] == 0.42125
    assert report["Details"]["algo-1"]["gpu1"]["p95"] == 0


@pytest.mark.slow
def test_max_initialization_time_rule():
    bucket_name = "s3://smdebug-testing/resources/tf2_detailed_profile/profiler-output"
    trial = create_trial(bucket_name, profiler=True, output_dir="/tmp")
    rule = MaxInitializationTime(trial)
    ending_step, ending_timestamp, condition_met_count, unavailability_count = invoke_rule(rule)
    assert ending_step == 3
    assert ending_timestamp == 1596668460000000
    assert condition_met_count == 0
    assert unavailability_count == 0
    report = rule.report
    assert report["Datapoints"] == 0
    assert report["Violations"] == 0
    assert report["RuleTriggered"] == 0
    assert report["Details"]["job_start"] == 1596668240.3562295

    rule = MaxInitializationTime(trial, threshold=0)
    ending_step, ending_timestamp, condition_met_count, unavailability_count = invoke_rule(rule)
    assert ending_step == 3
    assert ending_timestamp == 1596668460000000
    assert condition_met_count == 2
    assert unavailability_count == 0
    report = rule.report
    assert report["Datapoints"] == 0
    assert report["Violations"] == 2
    assert report["RuleTriggered"] == 2
    assert report["Details"]["job_end"] == 1596668399.9164186


@pytest.mark.slow
def test_loadbalancing_rule():
    bucket_name = "s3://smdebug-testing/resources/tf2_detailed_profile/profiler-output"
    trial = create_trial(bucket_name, profiler=True, output_dir="/tmp")
    rule = LoadBalancing(trial)
    ending_step, ending_timestamp, condition_met_count, unavailability_count = invoke_rule(rule)
    assert ending_step == 3
    assert ending_timestamp == 1596668460000000
    assert condition_met_count == 0
    assert unavailability_count == 0
    report = rule.report
    assert report["Datapoints"] == 320
    assert report["Violations"] == 0
    assert report["RuleTriggered"] == 0

    rule = LoadBalancing(trial, threshold=0)
    ending_step, ending_timestamp, condition_met_count, unavailability_count = invoke_rule(rule)
    assert ending_step == 3
    assert ending_timestamp == 1596668460000000
    assert condition_met_count == 0
    assert unavailability_count == 0
    report = rule.report
    assert report["Datapoints"] == 320
    assert report["Violations"] == 0
    assert report["RuleTriggered"] == 0


@pytest.mark.slow
def test_io_bottleneck_rule():
    bucket_name = "s3://smdebug-testing/resources/tf2_detailed_profile_0930/profiler-output"
    trial = create_trial(bucket_name, profiler=True, output_dir="/tmp")
    rule = IOBottleneck(trial)
    ending_step, ending_timestamp, condition_met_count, unavailability_count = invoke_rule(rule)
    assert ending_step == 6
    assert ending_timestamp == 1600942860000000
    assert condition_met_count == 0
    assert unavailability_count == 0
    report = rule.report
    assert report["Datapoints"] == 668
    assert report["Violations"] == 177
    assert report["RuleTriggered"] == 0
    assert report["Details"]["low_gpu_utilization"] == 651
    assert len(report["Details"]["bottlenecks"]) == 177

    rule = IOBottleneck(trial, gpu_threshold=100, io_threshold=0, patience=0)
    ending_step, ending_timestamp, condition_met_count, unavailability_count = invoke_rule(rule)
    assert ending_step == 6
    assert ending_timestamp == 1600942860000000
    assert condition_met_count == 4
    assert unavailability_count == 0
    report = rule.report
    assert report["Datapoints"] == 668
    assert report["Violations"] == 368
    assert report["RuleTriggered"] == 4
    assert report["Details"]["low_gpu_utilization"] == 662


@pytest.mark.slow
def test_overall_system_usage_rule():
    bucket_name = "s3://smdebug-testing/resources/tf2_detailed_profile/profiler-output"
    trial = create_trial(bucket_name, profiler=True, output_dir="/tmp")
    rule = OverallSystemUsage(trial)

    ending_step, ending_timestamp, condition_met_count, unavailability_count = invoke_rule(rule)
    assert ending_step == 3
    assert ending_timestamp == 1596668460000000
    assert condition_met_count == 0
    assert unavailability_count == 0
    report = rule.report
    assert report["Datapoints"] == 319
    assert report["Violations"] == 0
    assert report["RuleTriggered"] == 0
    assert report["Details"]["CPU"]["algo-1"]["max"] == 6.462968750000001
    assert report["Details"]["CPU"]["algo-1"]["p99"] == 5.492731249999999
    assert report["Details"]["CPU"]["algo-1"]["p50"] == 2.5025
    assert report["Details"]["CPU"]["algo-1"]["min"] == 0.42125


@pytest.mark.slow
def test_profiler_report_rule_condition_met():
    from smdebug.rules import rule_invoker

    bucket_name = "s3://smdebug-testing/resources/tf2_detailed_profile_0930/profiler-output"
    trial = create_trial(bucket_name, profiler=True, output_dir="/tmp")
    rule = ProfilerReport(trial)
    # Overwrite a sub rule to ensure at least one rule condition is met.
    rule.rules = [IOBottleneck(trial, gpu_threshold=100, io_threshold=0, patience=0)]
    try:
        rule_invoker.invoke_rule(rule, raise_eval_cond=True)
    except RuleEvaluationConditionMet as e:
        # For this test job, the rule will stop at step 6.
        assert e.step == 6
        return
    # Should not reach the end. Always expect RuleEvaluationConditionMet if any rule condition met.
    assert False


@pytest.mark.slow
def test_profiler_report_rule_condition_not_met():
    from smdebug.rules import rule_invoker

    bucket_name = "s3://smdebug-testing/resources/tf2_detailed_profile_0930/profiler-output"
    trial = create_trial(bucket_name, profiler=True, output_dir="/tmp")
    rule = ProfilerReport(trial)
    # Overwrite a sub rule to ensure NO rule condition met.
    rule.rules = [IOBottleneck(trial)]
    try:
        rule_invoker.invoke_rule(rule, raise_eval_cond=True)
        # If the invoke rule ended without a ruleEvaluation thrown, no rule condition is met.
        assert not rule.rule_condition_met
    except RuleEvaluationConditionMet:
        # The test should not trigger any rule condition.
        assert False
