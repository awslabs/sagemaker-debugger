import argparse
from tornasole.trials import create_trial
from tornasole.rules.generic import WeightUpdateRatio
from tornasole.rules.rule_invoker import invoke_rule

parser = argparse.ArgumentParser()
parser.add_argument('--trial-dir', type=str)
parser.add_argument('--start-step', type=int, default=0)
parser.add_argument('--end-step', type=int)
parser.add_argument('--large-threshold', type=float, default=10)
parser.add_argument('--small-threshold', type=float, default=0.00000001)
args = parser.parse_args()

trial = create_trial(args.trial_dir, range_steps=(args.start_step, args.end_step))
wur = WeightUpdateRatio(trial,
                        large_threshold=args.large_threshold,
                        small_threshold=args.small_threshold)
invoke_rule(wur, start_step=args.start_step, end_step=args.end_step)