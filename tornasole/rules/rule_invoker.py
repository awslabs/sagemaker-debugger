from tornasole.exceptions import *
from tornasole.core.utils import get_logger

logger = get_logger()

def invoke_rule(rule_obj, start_step=0, end_step=None, raise_rule_eval=False):
  step = start_step if start_step is not None else 0
  logger.info('Started execution of rule {}'.format(type(rule_obj).__name__))
  while (end_step is None) or (step < end_step):
    try:
      rule_obj.invoke(step)
    except (TensorUnavailableForStep, StepUnavailable) as e:
      logger.debug(str(e))
    except RuleEvaluationConditionMet as e:
      if raise_rule_eval:
        raise e
    step += 1

if __name__ == '__main__':
  import argparse
  from tornasole.trials import create_trial

  parser = argparse.ArgumentParser()
  parser.add_argument('--trial-dir', type=str)
  parser.add_argument('--rule-name', type=str)
  parser.add_argument('--start-step', type=int)
  parser.add_argument('--end-step', type=int)
  parser.add_argument('--raise-rule-eval-cond-exception', type=bool, default=False)

  parser.add_argument('--collections', default=[], type=str, action='append',
                      help="""List of collection names. The rule will inspect tensors belonging to those collections. 
                      Required for allzero rule.""")
  parser.add_argument('--tensor-regex', default=[], type=str, action='append',
                      help="""List of regex patterns. The rule will inspect tensors that match these 
                      patterns. Required for allzero rule.""")
  args = parser.parse_args()

  if args.rule_name is None:
    raise RuntimeError('Needs rule name to invoke')

  tr = create_trial(args.trial_dir, range_steps=(args.start_step, args.end_step))
  if args.rule_name.lower() == 'vanishinggradient':
    from tornasole.rules.generic.vanishing_grad import VanishingGradient
    r = VanishingGradient(tr)
  elif args.rule_name.lower() == 'explodingtensor':
    from tornasole.rules.generic.exploding_tensor import ExplodingTensor
    r = ExplodingTensor(tr)
  elif args.rule_name.lower() == 'weightupdateratio':
    from tornasole.rules.generic.weight_update_ratio import WeightUpdateRatio
    r = WeightUpdateRatio(tr)
  elif args.rule_name.lower() == 'allzero':
    if len(args.collections) == 0 and len(args.tensor_regex) == 0:
      raise ValueError('Please provide either the list of collection names or list of regex patterns for invoking '
                       'this rule.')
    from tornasole.rules.generic.all_zero import AllZero
    r = AllZero(tr, args.collections, args.tensor_regex)
  else:
    raise ValueError('Please invoke any rules which take multiple trials, '
                     'or custom rules by passing the rule object to '
                     'invoke_rule() function. We do not currently '
                     'support running such rules from this python script.'
                     'Please refer to examples/scripts/ for examples'
                     'on how to call invoke_rule')

  invoke_rule(r, start_step=args.start_step, end_step=args.end_step,
              raise_rule_eval=args.raise_rule_eval_cond_exception)
