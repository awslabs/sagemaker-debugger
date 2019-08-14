from tests.analysis.utils import generate_data

from tornasole.rules.generic import ExplodingTensor
from tornasole.trials import create_trial
import uuid
import numpy as np
from tornasole.exceptions import *
from tornasole.rules.rule_invoker import invoke_rule

def test_invoker_exception():
  run_id = str(uuid.uuid4())
  base_path = 'ts_output/rule_invoker/'
  path = base_path + run_id

  num_tensors = 3

  generate_data(path=base_path, trial=run_id, num_tensors=num_tensors,
                step=0, tname_prefix='foo', worker='algo-1', shape=(1,),
                data=np.array([np.nan]))
  generate_data(path=base_path, trial=run_id, num_tensors=num_tensors,
                step=1, tname_prefix='foo', worker='algo-1', shape=(1,))
  generate_data(path=base_path, trial=run_id, num_tensors=num_tensors,
                step=2, tname_prefix='foo', worker='algo-1', shape=(1,),
                data=np.array([np.nan]))

  tr = create_trial(path)
  r = ExplodingTensor(tr)

  c = 0
  for start_step in range(2):
    try:
      invoke_rule(r, start_step=start_step, end_step=3, raise_rule_eval=True)
    except RuleEvaluationConditionMet as e:
      c += 1
  assert c == 2