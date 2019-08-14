import argparse
from tornasole.rules.generic import SimilarAcrossRuns
from tornasole.trials import create_trial
from tornasole.rules.rule_invoker import invoke_rule

parser = argparse.ArgumentParser()
parser.add_argument('--trial-dir', default=[], type=str, action='append')
parser.add_argument('--include', default=[], type=str, action='append',
        help="""List of REs for tensors to include for this check""")
parser.add_argument('--start-step', default=0, type=int)
parser.add_argument('--end-step', type=int)
args = parser.parse_args()
if len(args.trial_dir) != 2:
  raise RuntimeError('This rule requires two trials')

trials = []
for t in args.trial_dir:
  trials.append(create_trial(t, range_steps=(args.start_step, args.end_step)))

sr = SimilarAcrossRuns(trials[0], trials[1], include_regex=args.include)
invoke_rule(sr, start_step=args.start_step, end_step=args.end_step)

