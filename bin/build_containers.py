import subprocess
import argparse

parser = argparse.ArgumentParser(description='Build Tornasole binaries')
parser.add_argument('--tag', type=str, default='latest',
                    help='Pass the tag to upload the image to ECR with. '
                         'You might want to set the tag to latest for '
                         'final images.')
args = parser.parse_args()

FRAMEWORK_VERSIONS = {'mxnet': '1.4.1', 'tensorflow': '1.13.1', 'pytorch': '1.1.0'}
for f, v in FRAMEWORK_VERSIONS.items():
    subprocess.check_call(['bash',
                           'bin/sagemaker-containers/{}/{}/build.sh'.format(f, v),
                           args.tag])
