from tests.analysis.utils import generate_data
from tornasole.rules.generic import LossNotDecreasing
from tornasole.trials import create_trial
from tornasole.exceptions import *
from tornasole.rules.rule_invoker import invoke_rule
import uuid
import numpy as np


def dump_data(values):
  run_id = str(uuid.uuid4())
  base_path = 'ts_output/rule_invoker/'
  num_tensors = 2
  shape = (1,)
  i = 0
  for v in values:
    generate_data(path=base_path, trial=run_id, num_tensors=num_tensors,
                  step=i, tname_prefix='loss', worker='algo-1', shape=shape,
                  data=np.ones(shape=shape) * v, export_colls=True if i==0 else False)
    i+=4
  return base_path + run_id


def test_default():
  path = dump_data([0.3, 0.33, 0.31, 0.30, 0.28])
  tr = create_trial(path)
  rules = [ LossNotDecreasing(tr, tensor_regex='loss, losses'),
            LossNotDecreasing(tr, collection_names=['default']),
            LossNotDecreasing(tr)]
  for r in rules:
    try:
      invoke_rule(r, start_step=0, end_step=17, raise_eval_cond=True)
      assert False
    except RuleEvaluationConditionMet as e:
      print(e)
      assert e.step == 12

    try:
      invoke_rule(r, start_step=13, end_step=17, raise_eval_cond=True)
    except RuleEvaluationConditionMet as e:
      print(e)
      assert False


def test_min_diff():
  path = dump_data([0.3, 0.33, 0.31, 0.30, 0.28, 0.2, 0.42])
  tr = create_trial(path)
  r = LossNotDecreasing(tr, diff_percent=20)
  try:
    invoke_rule(r, start_step=0, end_step=25, raise_eval_cond=True)
    assert False
  except RuleEvaluationConditionMet as e:
    print(e)
    assert e.step == 24
