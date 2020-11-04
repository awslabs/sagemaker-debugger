# Third Party
import pytest

# First Party
from smdebug.exceptions import NoMoreProfilerData, RuleEvaluationConditionMet
from smdebug.profiler.analysis.rules.batch_size import BatchSize
from smdebug.profiler.analysis.rules.cpu_bottleneck import CPUBottleneck
from smdebug.profiler.analysis.rules.dataloaders import Dataloaders
from smdebug.profiler.analysis.rules.gpu_memory_increase import GPUMemoryIncrease
from smdebug.profiler.analysis.rules.io_bottleneck import IOBottleneck
from smdebug.profiler.analysis.rules.load_balancing import LoadBalancing
from smdebug.profiler.analysis.rules.low_gpu_utilization import LowGPUUtilization
from smdebug.profiler.analysis.rules.max_initialization_time import MaxInitializationTime
from smdebug.profiler.analysis.rules.overall_system_usage import OverallSystemUsage
from smdebug.profiler.analysis.rules.profiler_report import ProfilerReport
from smdebug.profiler.analysis.rules.step_outlier import StepOutlier
from smdebug.rules.rule_invoker import invoke_rule
from smdebug.trials import create_trial


@pytest.mark.slow
def test_gpu_usage_rule():
    bucket_name = "s3://smdebug-testing/resources/tf2_detailed_profile/profiler-output"
    trial = create_trial(bucket_name, profiler=True)
    rule = LowGPUUtilization(trial)
    try:
        invoke_rule(rule, raise_eval_cond=True)
        assert False
    except NoMoreProfilerData as e:
        print(
            "The training has ended and there is no more data to be analyzed. This is expected behavior."
        )
    except RuleEvaluationConditionMet as e:
        print(e)
        assert e.step == 1

    rule = LowGPUUtilization(trial, threshold_p95=0, threshold_p5=0, patience=0)
    try:
        invoke_rule(rule, raise_eval_cond=True)
        assert False
    except NoMoreProfilerData as e:
        print(
            "The training has ended and there is no more data to be analyzed. This is expected behavior."
        )
        print(e.timestamp)
        assert e.timestamp == 1596668460000000
    except RuleEvaluationConditionMet as e:
        print(e)


@pytest.mark.slow
def test_cpu_bottleneck_rule():
    bucket_name = "s3://smdebug-testing/resources/tf2_detailed_profile/profiler-output"
    trial = create_trial(bucket_name, profiler=True)
    rule = CPUBottleneck(trial)
    try:
        invoke_rule(rule, raise_eval_cond=True)
        assert False
    except NoMoreProfilerData as e:
        print(
            "The training has ended and there is no more data to be analyzed. This is expected behavior."
        )
        assert e.timestamp == 1596668460000000
    except RuleEvaluationConditionMet as e:
        print(e)

    rule = CPUBottleneck(trial, gpu_threshold=100, cpu_threshold=0, patience=0)
    try:
        invoke_rule(rule, raise_eval_cond=True)
        assert False
    except NoMoreProfilerData as e:
        print(
            "The training has ended and there is no more data to be analyzed. This is expected behavior."
        )
    except RuleEvaluationConditionMet as e:
        print(e)
        assert e.step == 0


@pytest.mark.slow
def test_step_outlier_rule():
    bucket_name = "s3://smdebug-testing/resources/tf2_detailed_profile/profiler-output"
    trial = create_trial(bucket_name, profiler=True)
    rule = StepOutlier(trial, n_outliers=1)
    try:
        invoke_rule(rule, raise_eval_cond=True)
        assert False
    except NoMoreProfilerData as e:
        print(
            "The training has ended and there is no more data to be analyzed. This is expected behavior."
        )
    except RuleEvaluationConditionMet as e:
        assert e == 3

    rule = StepOutlier(trial, stddev=100, n_outliers=100)
    try:
        invoke_rule(rule, raise_eval_cond=True)
        assert False
    except NoMoreProfilerData as e:
        print(
            "The training has ended and there is no more data to be analyzed. This is expected behavior."
        )
        assert e.timestamp == 1596668460000000
    except RuleEvaluationConditionMet as e:
        print(e)


@pytest.mark.slow
def test_gpu_memory_increase_rule():
    bucket_name = "s3://smdebug-testing/resources/tf2_detailed_profile_0930/profiler-output"
    trial = create_trial(bucket_name, profiler=True)
    rule = GPUMemoryIncrease(trial)
    try:
        invoke_rule(rule, raise_eval_cond=True)
        assert False
    except NoMoreProfilerData as e:
        print(
            "The training has ended and there is no more data to be analyzed. This is expected behavior."
        )
        print(e.timestamp)
        assert e.timestamp == 1600942860000000
    except RuleEvaluationConditionMet as e:
        print(e)

    rule = GPUMemoryIncrease(trial, increase=0, patience=0)
    try:
        invoke_rule(rule, raise_eval_cond=True)
        assert False
    except NoMoreProfilerData as e:
        print(
            "The training has ended and there is no more data to be analyzed. This is expected behavior."
        )
    except RuleEvaluationConditionMet as e:
        print(e)
        assert e.step == 5


@pytest.mark.slow
def test_batch_size_rule():
    bucket_name = "s3://smdebug-testing/resources/tf2_detailed_profile/profiler-output"
    trial = create_trial(bucket_name, profiler=True)
    rule = BatchSize(trial)
    try:
        invoke_rule(rule, raise_eval_cond=True)
        assert False
    except NoMoreProfilerData as e:
        print(
            "The training has ended and there is no more data to be analyzed. This is expected behavior."
        )
        print(e.timestamp)
        assert e.timestamp == 1596668460000000
    except RuleEvaluationConditionMet as e:
        print(e)

    rule = BatchSize(
        trial,
        cpu_threshold_p95=100,
        gpu_threshold_p95=100,
        gpu_memory_threshold_p95=100,
        patience=0,
        window=10,
    )
    try:
        invoke_rule(rule, raise_eval_cond=True)
        assert False
    except NoMoreProfilerData as e:
        print(
            "The training has ended and there is no more data to be analyzed. This is expected behavior."
        )
    except RuleEvaluationConditionMet as e:
        print(e)
        assert e.step == 0


@pytest.mark.slow
def test_max_initialization_time_rule():
    bucket_name = "s3://smdebug-testing/resources/tf2_detailed_profile/profiler-output"
    trial = create_trial(bucket_name, profiler=True)
    rule = MaxInitializationTime(trial)
    try:
        invoke_rule(rule, raise_eval_cond=True)
        assert False
    except NoMoreProfilerData as e:
        print(
            "The training has ended and there is no more data to be analyzed. This is expected behavior."
        )
        assert e.timestamp == 1596668460000000
    except RuleEvaluationConditionMet as e:
        print(e)

    rule = MaxInitializationTime(trial, threshold=0)
    try:
        invoke_rule(rule, raise_eval_cond=True)
        assert False
    except NoMoreProfilerData as e:
        print(
            "The training has ended and there is no more data to be analyzed. This is expected behavior."
        )
    except RuleEvaluationConditionMet as e:
        print(e)
        assert e.step == 1


@pytest.mark.slow
def test_dataloaders_rule():
    bucket_name = "s3://smdebug-testing/resources/tf2_detailed_profile/profiler-output"
    trial = create_trial(bucket_name, profiler=True)
    rule = Dataloaders(trial)
    try:
        invoke_rule(rule, raise_eval_cond=True)
        assert False
    except NoMoreProfilerData as e:
        print(
            "The training has ended and there is no more data to be analyzed. This is expected behavior."
        )
    except RuleEvaluationConditionMet as e:
        print(e)
        assert e.step == 3
    rule = Dataloaders(trial, min_threshold=0, max_threshold=100)
    try:
        invoke_rule(rule, raise_eval_cond=True)
        assert False
    except NoMoreProfilerData as e:
        print(
            "The training has ended and there is no more data to be analyzed. This is expected behavior."
        )
        assert e.timestamp == 1596668460000000
    except RuleEvaluationConditionMet as e:
        print(e)
        assert e.step == 3


@pytest.mark.slow
def test_loadbalancing_rule():
    bucket_name = "s3://smdebug-testing/resources/tf2_detailed_profile/profiler-output"
    trial = create_trial(bucket_name, profiler=True)
    rule = LoadBalancing(trial)
    try:
        invoke_rule(rule, raise_eval_cond=True)
        assert False
    except NoMoreProfilerData as e:
        print(
            "The training has ended and there is no more data to be analyzed. This is expected behavior."
        )
        assert e.timestamp == 1596668460000000
    except RuleEvaluationConditionMet as e:
        print(e)
    rule = LoadBalancing(trial, threshold=0)
    try:
        invoke_rule(rule, raise_eval_cond=True)
        assert False
    except NoMoreProfilerData as e:
        print(
            "The training has ended and there is no more data to be analyzed. This is expected behavior."
        )
    except RuleEvaluationConditionMet as e:
        print(e)
        assert e.step == 0


@pytest.mark.slow
def test_io_bottleneck_rule():
    bucket_name = "s3://smdebug-testing/resources/tf2_detailed_profile_0930/profiler-output"
    trial = create_trial(bucket_name, profiler=True)
    rule = IOBottleneck(trial)
    try:
        invoke_rule(rule, raise_eval_cond=True)
        assert False
    except NoMoreProfilerData as e:
        print(
            "The training has ended and there is no more data to be analyzed. This is expected behavior."
        )
        assert e.timestamp == 1600942860000000
    except RuleEvaluationConditionMet as e:
        print(e)

    rule = IOBottleneck(trial, gpu_threshold=100, io_threshold=0, patience=0)
    try:
        invoke_rule(rule, raise_eval_cond=True)
        assert False
    except NoMoreProfilerData as e:
        print(
            "The training has ended and there is no more data to be analyzed. This is expected behavior."
        )
    except RuleEvaluationConditionMet as e:
        print(e)
        assert e.step == 2


@pytest.mark.slow
def test_overall_system_usage_rule():
    bucket_name = "s3://smdebug-testing/resources/tf2_detailed_profile/profiler-output"
    trial = create_trial(bucket_name, profiler=True)
    rule = OverallSystemUsage(trial)
    try:
        invoke_rule(rule, raise_eval_cond=True)
        assert False
    except NoMoreProfilerData as e:
        print(
            "The training has ended and there is no more data to be analyzed. This is expected behavior."
        )
        assert e.timestamp == 1596668460000000
    except RuleEvaluationConditionMet as e:
        print(e)


@pytest.mark.slow
def test_profiler_report_rule_condition_met():
    bucket_name = "s3://smdebug-testing/resources/tf2_detailed_profile_0930/profiler-output"
    trial = create_trial(bucket_name, profiler=True, output_dir="/tmp")
    rule = ProfilerReport(trial)
    # Overwrite a sub rule to ensure at least one rule condition is met.
    rule.rules = [IOBottleneck(trial, gpu_threshold=100, io_threshold=0, patience=0)]
    try:
        invoke_rule(rule, raise_eval_cond=True)
    except RuleEvaluationConditionMet as e:
        # For this test job, the rule will stop at step 6.
        assert e.step == 6
        return
    # Should not reach the end. Always expect RuleEvaluationConditionMet if any rule condition met.
    assert False


@pytest.mark.slow
def test_profiler_report_rule_condition_not_met():
    bucket_name = "s3://smdebug-testing/resources/tf2_detailed_profile_0930/profiler-output"
    trial = create_trial(bucket_name, profiler=True, output_dir="/tmp")
    rule = ProfilerReport(trial)
    # Overwrite a sub rule to ensure NO rule condition met.
    rule.rules = [IOBottleneck(trial)]
    try:
        invoke_rule(rule, raise_eval_cond=True)
        # If the invoke rule ended without a ruleEvaluation thrown, no rule condition is met.
        assert not rule.rule_condition_met
    except RuleEvaluationConditionMet:
        # The test should not trigger any rule condition.
        assert False
