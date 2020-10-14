# Standard Library
import json
import os
from multiprocessing import Process

# First Party
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
from smdebug.profiler.analysis.rules.profiler_report import ProfilerReport
from smdebug.profiler.analysis.rules.step_outlier import StepOutlier
from smdebug.rules.rule_invoker import invoke_rule
from smdebug.trials import create_trial

os.mkdir("/opt/ml/processing/outputs/profiler-reports/")
os.mkdir("/opt/ml/processing/outputs/.sagemaker-ignore")


def dump_report(rule_object):
    with open(
        "/opt/ml/processing/outputs/profiler-reports/" + rule_object.rule_name + ".json", "w"
    ) as fp:
        json.dump(rule.report, fp)


# run the rule
def run_rule(rule_obj, should_dump_report=False):
    try:
        invoke_rule(rule_obj)  # , raise_eval_cond=True)
    except NoMoreProfilerData:
        print(
            "The training has ended and there is no more data to be analyzed. This is expected behavior."
        )
    except RuleEvaluationConditionMet as e:
        print(e)
    if should_dump_report is True:
        dump_report()

    return


# path to profiler data
profiler_path = os.environ["S3_PATH"]
trial = create_trial(profiler_path, profiler=True)

if "TRIGGER_ALL" in os.environ:
    # create list of rules
    rules = []
    rules.append(CPUBottleneck(trial, gpu_threshold=100, cpu_threshold=0))
    rules.append(IOBottleneck(trial, gpu_threshold=100, io_threshold=-1))
    rules.append(LowGPUUtilization(trial, threshold_p95=100, threshold_p5=100))
    rules.append(StepOutlier(trial, stddev=1, n_outliers=0))
    rules.append(GPUMemoryIncrease(trial, increase=0, window=10))
    rules.append(
        BatchSize(trial, cpu_threshold_p95=100, gpu_threshold_p95=100, gpu_memory_threshold_p95=100)
    )
    rules.append(MaxInitializationTime(trial))
    # rules.append(Dataloaders(trial))
    rules.append(LoadBalancing(trial, threshold=0))
    rules.append(OverallSystemUsage(trial))
else:
    # create list of rules
    rules = []
    rules.append(CPUBottleneck(trial))
    rules.append(IOBottleneck(trial))
    rules.append(LowGPUUtilization(trial))
    rules.append(StepOutlier(trial))
    rules.append(GPUMemoryIncrease(trial))
    rules.append(BatchSize(trial))
    rules.append(MaxInitializationTime(trial))
    # rules.append(Dataloaders(trial))
    rules.append(LoadBalancing(trial))
    rules.append(OverallSystemUsage(trial))

# create subprocess for each rule
should_run_parallel = False
if should_run_parallel:
    processes = []
    for rule in rules:
        p = Process(target=run_rule, args=(rule, True))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
else:
    run_rule(ProfilerReport(trial, rules))
    for rule in rules:
        dump_report(rule)

rule = PlotVisualizations(
    trial,
    create_html=True,
    nb_full_path="profiler_report.ipynb",
    output_full_path="/opt/ml/processing/outputs/profiler-report.ipynb",
)
rule._plot_visualization(last_found_step=0)
