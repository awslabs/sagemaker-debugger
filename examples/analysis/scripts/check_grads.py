import argparse
from tornasole.trials import create_trial
from tornasole.rules.generic import VanishingGradient
from tornasole.rules import invoke_rule

#example trial root
# local: /home/ubuntu/tmp/pycharm_project_932/repositories/tornasole_tf/examples/training_scripts/tornasole_outputs/
# s3: s3://huilgolr-tf/tornasole/tornasole_outputs

parser = argparse.ArgumentParser()
parser.add_argument('--trial-dir', type=str)
parser.add_argument('--threshold', type=float, default=0.0000001)
args = parser.parse_args()

trial_obj = create_trial(args.trial_dir)
vr = VanishingGradient(base_trial=trial_obj, threshold=args.threshold)
invoke_rule(vr)

