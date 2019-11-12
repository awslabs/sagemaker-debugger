# Standard Library
import shutil
import subprocess
import sys
import uuid

# Third Party
import numpy as np
import pytest
from tests.analysis.utils import generate_data

# First Party
from smdebug.exceptions import *
from smdebug.rules.generic import ExplodingTensor
from smdebug.rules.rule_invoker import invoke_rule
from smdebug.trials import create_trial


def dump_data():
    run_id = str(uuid.uuid4())
    base_path = "ts_output/rule_invoker/"
    path = base_path + run_id

    num_tensors = 3

    generate_data(
        path=base_path,
        trial=run_id,
        num_tensors=num_tensors,
        step=0,
        tname_prefix="foo",
        worker="algo-1",
        shape=(1,),
        data=np.array([np.nan]),
    )
    generate_data(
        path=base_path,
        trial=run_id,
        num_tensors=num_tensors,
        step=1,
        tname_prefix="foo",
        worker="algo-1",
        shape=(1,),
    )
    generate_data(
        path=base_path,
        trial=run_id,
        num_tensors=num_tensors,
        step=2,
        tname_prefix="foo",
        worker="algo-1",
        shape=(1,),
        data=np.array([np.nan]),
    )
    return path


def test_invoker_exception():
    path = dump_data()
    tr = create_trial(path)
    r = ExplodingTensor(tr, collection_names="gradients,weights")

    c = 0
    for start_step in range(2):
        try:
            invoke_rule(r, start_step=start_step, end_step=3, raise_eval_cond=True)
        except RuleEvaluationConditionMet as e:
            c += 1
    assert c == 2
    shutil.rmtree(path)


def test_invoker_rule_default_args():
    path = dump_data()
    rcode = subprocess.check_call(
        [
            sys.executable,
            "-m",
            "smdebug.rules.rule_invoker",
            "--trial-dir",
            path,
            "--rule-name",
            "VanishingGradient",
            "--end-step",
            "3",
        ]
    )
    assert rcode == 0
    shutil.rmtree(path)


def test_invoker_rule_pass_kwargs():
    path = dump_data()
    rcode = subprocess.check_call(
        [
            sys.executable,
            "-m",
            "smdebug.rules.rule_invoker",
            "--rule-name",
            "VanishingGradient",
            "--trial-dir",
            path,
            "--threshold",
            "0.001",
            "--end-step",
            "3",
        ]
    )
    assert rcode == 0
    shutil.rmtree(path)


@pytest.mark.slow  # 0:04 to run
def test_invoker_rule_pass_other_trials():
    path1 = dump_data()
    path2 = dump_data()
    rcode = subprocess.check_call(
        [
            sys.executable,
            "-m",
            "smdebug.rules.rule_invoker",
            "--trial-dir",
            path1,
            "--other-trials",
            path2,
            "--rule-name",
            "SimilarAcrossRuns",
            "--end-step",
            "3",
        ]
    )
    assert rcode == 0
    shutil.rmtree(path1)
    shutil.rmtree(path2)
