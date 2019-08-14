from tornasole.exceptions import *
from tornasole.core.utils import get_logger
from tornasole.trials import create_trial
from tornasole.rules.generic import *
import inspect

logger = get_logger()


def create_trials(comma_separated_trial_dirs, **kwargs):
  trials = []
  for tr_dir in comma_separated_trial_dirs.split(','):
    trials.append(create_trial(tr_dir, **kwargs))
  return trials


def get_rule_arguments(args_dict, classobj):
  args = []
  kwargs = {}
  class_argspec = inspect.getfullargspec(classobj.__init__)
  start = len(class_argspec.args) - len(class_argspec.defaults)
  # skip 0 and 1 as they are self and base_trial
  for i in range(2, start):
    arg_name = class_argspec.args[i]
    if arg_name not in args_dict:
      raise ValueError('Required argument {} for {} is missing'
                       .format(arg_name, classobj.__name__))
    if arg_name == 'other_trials':
      other_trials = create_trials(args_dict['other_trials'],
                                   range_steps=(args_dict['start_step'],
                                                args_dict['end_step']))
      args.append(other_trials)
    else:
      args.append(args_dict[arg_name])
  for d in class_argspec.defaults:
    arg_name = class_argspec.args[start]
    if arg_name not in args_dict:
      # set default
      kwargs[arg_name] = d
    else:
      kwargs[arg_name] = args_dict[arg_name]
    start += 1
  return args, kwargs


def get_rule(rule_name):
  rule_name_lower = rule_name.lower()
  if rule_name_lower == 'vanishinggradient':
    return VanishingGradient
  elif rule_name_lower == 'explodingtensor':
    return ExplodingTensor
  elif rule_name_lower == 'allzero':
    return AllZero
  elif rule_name_lower == 'weightupdateratio':
    return WeightUpdateRatio
  elif rule_name_lower == 'similaracrossruns':
    return SimilarAcrossRuns
  elif rule_name_lower == 'unchangedtensor':
    return UnchangedTensor
  else:
    raise ValueError('rule_invoker does not recognize the rule')


def create_rule(args, args_dict):
  tr = create_trial(args.trial_dir, range_steps=(args.start_step, args.end_step))
  rule_class = get_rule(args.rule_name)
  rule_args, rule_kwargs = get_rule_arguments(args_dict, rule_class)
  r = rule_class(tr, *rule_args, **rule_kwargs)
  return r


def invoke_rule(rule_obj, start_step=0, end_step=None, raise_eval_cond=False):
  step = start_step if start_step is not None else 0
  logger.info('Started execution of rule {} at step {}'.format(type(rule_obj).__name__, step))
  while (end_step is None) or (step < end_step):
    try:
      rule_obj.invoke(step)
    except (TensorUnavailableForStep, StepUnavailable) as e:
      logger.debug(str(e))
    except RuleEvaluationConditionMet as e:
      if raise_eval_cond:
        raise e
    step += 1
  # decrementing because we increment step in the above line
  logger.info('Ended execution of rule {} at end_step {}'
              .format(type(rule_obj).__name__, step - 1))

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Rule invoker takes the below arguments and'
                                               'any argument taken by the rules. The arguments not'
                                               'mentioned below are automatically passed when'
                                               'creating the rule objects.')
  parser.add_argument('--trial-dir', type=str, required=True)
  parser.add_argument('--rule-name', type=str, required=True)
  parser.add_argument('--other-trials', type=str,
                      help='comma separated paths for '
                           'other trials taken by the rule')
  parser.add_argument('--start-step', type=int)
  parser.add_argument('--end-step', type=int)
  parser.add_argument('--raise-rule-eval-cond-exception',
                      type=bool, default=False)
  parsed, unknown = parser.parse_known_args()
  for arg in unknown:
    if arg.startswith('--'):
      parser.add_argument(arg, type=str)
  args = parser.parse_args()
  args_dict = vars(args)
  r = create_rule(args, args_dict)
  invoke_rule(r, start_step=args.start_step, end_step=args.end_step,
              raise_eval_cond=args.raise_rule_eval_cond_exception)
  