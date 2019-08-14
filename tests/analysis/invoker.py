from tornasole.exceptions import *
from tornasole.core.utils import get_logger
logger = get_logger()

def invoke_rule(rule_obj, flag, start_step, end_step):
    step = start_step if start_step is not None else 0
    logger.info('Started execution of rule {}'.format(type(rule_obj).__name__))
    return_false = False
    while (end_step is None) or (step < end_step): # if end_step is not provided, do infinite checking
        try:
            rule_obj.invoke(step)
            if flag == 'False':
                return_false = True
            elif flag == 'True':
                # every step should return True in this case,
                # meaning exception condition should be met
                assert False
            step += 1
        except StepUnavailable as e:
            logger.info(e)
            step += 1
        except TensorUnavailableForStep as e:
            logger.info(e)
            step += 1
        except RuleEvaluationConditionMet as e:
            logger.info(e)
            step += 1

    # if flag is False, return_false should be True after the loop
    if flag == 'False':
        assert return_false
    logger.info('Ending execution of rule {} with step={} '.format(rule_obj.__class__.__name__, step))


if __name__ == '__main__':
  import argparse
  from tornasole.trials import create_trial

  parser = argparse.ArgumentParser()
  parser.add_argument('--tornasole_path', type=str)
  parser.add_argument('--rule_name', type=str)
  parser.add_argument('--start_step', type=int)
  parser.add_argument('--end_step', type=int)
  parser.add_argument('--flag', type=str, default=None)

  parser.add_argument('--weightupdateratio_large_threshold', type=float, default=10)
  parser.add_argument('--weightupdateratio_small_threshold', type=float, default=0.00000001)

  parser.add_argument('--vanishinggradient_threshold', type=float, default=0.0000001)
  parser.add_argument('--collections', default=[], type=str, action='append',
                      help="""List of collection names. The rule will inspect tensors belonging to those collections. Required for allzero 
                      rule.""")
  parser.add_argument('--tensor-regex', default=[], type=str, action='append',
                      help="""List of regex patterns. The rule will inspect tensors that match these 
                      patterns. Required for allzero 
                      rule.""")
  args = parser.parse_args()
  if args.rule_name is None:
    raise RuntimeError('Needs rule name to invoke')

  tr = create_trial(args.tornasole_path, range_steps=(args.start_step, args.end_step))
  if args.rule_name.lower() == 'vanishinggradient':
    from tornasole.rules.generic import VanishingGradient
    r = VanishingGradient(tr, threshold=args.vanishinggradient_threshold)
  elif args.rule_name.lower() == 'explodingtensor':
    from tornasole.rules.generic import ExplodingTensor
    r = ExplodingTensor(tr)
  elif args.rule_name.lower() == 'weightupdateratio':
    from tornasole.rules.generic import WeightUpdateRatio
    r = WeightUpdateRatio(tr,
                          large_threshold=args.weightupdateratio_large_threshold,
                          small_threshold=args.weightupdateratio_small_threshold)
  elif args.rule_name.lower() == 'allzero':
    if len(args.collections) == 0 and len(args.tensor_regex) == 0:
      raise ValueError('Please provide either the list of collection names or list of regex patterns for invoking '
                       'this rule.')
    from tornasole.rules.generic import AllZero
    r = AllZero(tr, args.collections, args.tensor_regex)
  else:
    raise ValueError('Please invoke any rules which take multiple trials, '
                     'or custom rules by passing the rule object to '
                     'invoke_rule() function. We do not currently '
                     'support running such rules from this python script.'
                     'Please refer to examples/scripts/ for examples'
                     'on how to call invoke_rule')
  invoke_rule(r, flag=args.flag, start_step=args.start_step, end_step=args.end_step)
