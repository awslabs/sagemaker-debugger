from tests.analysis.utils import generate_data

from tornasole.rules.generic import UnchangedTensor
from tornasole.trials import create_trial
import uuid
import numpy as np
from tornasole.exceptions import *
from tornasole.rules.rule_invoker import invoke_rule

def test_unchanged():
  run_id = str(uuid.uuid4())
  base_path = 'ts_output/rule_invoker/'
  path = base_path + run_id

  num_tensors = 3

  shape = (10, 3, 2)
  generate_data(path=base_path, trial=run_id, num_tensors=num_tensors,
                step=0, tname_prefix='foo', worker='algo-1', shape=shape,
                data=np.ones(shape=shape))
  generate_data(path=base_path, trial=run_id, num_tensors=num_tensors,
                step=1, tname_prefix='foo', worker='algo-1', shape=shape,
                data=np.ones(shape=shape))
  generate_data(path=base_path, trial=run_id, num_tensors=num_tensors,
                step=2, tname_prefix='foo', worker='algo-1', shape=shape,
                data=np.ones(shape=shape))

  generate_data(path=base_path, trial=run_id, num_tensors=num_tensors,
                step=5, tname_prefix='boo', worker='algo-1', shape=shape,
                data=np.ones(shape=shape))

  tr = create_trial(path)
  r = UnchangedTensor(tr, tensor_regex='.*')

  invoke_rule(r, start_step=0, end_step=2, raise_eval_cond=True)

  try:
    invoke_rule(r, start_step=0, end_step=3, raise_eval_cond=True)
    assert False
  except RuleEvaluationConditionMet:
    pass

  try:
    invoke_rule(r, start_step=2, end_step=3, raise_eval_cond=True)
    assert False
  except RuleEvaluationConditionMet:
    pass

  invoke_rule(r, start_step=3, end_step=6, raise_eval_cond=True)
