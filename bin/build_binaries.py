import subprocess
import argparse
import sys
import os

parser = argparse.ArgumentParser(description='Build Tornasole binaries')
parser.add_argument('--upload', default=False,
                    dest='upload', action='store_true',
                    help='Pass --upload if you want to upload the binaries'
                         'built to the s3 location')
args = parser.parse_args()

VERSION = '0.3'
BINARIES = ['mxnet', 'tensorflow', 'pytorch', 'rules']

for b in BINARIES:
    if b == 'rules':
        env_var = 'TORNASOLE_FOR_RULES'
    else:
        env_var = 'TORNASOLE_WITH_' + b.upper()
    env = dict(os.environ)
    env[env_var] = '1'
    subprocess.check_call([sys.executable, 'setup.py', 'bdist_wheel', '--universal'],
                          env=env)
    if args.upload:
        subprocess.check_call(['aws', 's3', 'cp', 'dist/tornasole-{}-py2.py3-none-any.whl'.format(VERSION),
                               's3://tornasole-binaries-use1/tornasole_{}/py3/'.format(b)])

    subprocess.check_call(['rm', '-rf', 'dist', 'build', '*.egg-info'])
