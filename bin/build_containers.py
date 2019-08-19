import subprocess
import argparse
from multiprocessing import Process
import os


# you can clean all caches used by docker with `docker system prune -a`
def build_container(framework, version, args):
    with open(os.path.join(args.logs_path, '{}.log'.format(framework)), "w") as logfile:
        subprocess.check_call(['bash',
                               'bin/sagemaker-containers/{}/{}/build.sh'.format(framework, version),
                               args.tag], stdout=logfile, stderr=logfile)


parser = argparse.ArgumentParser(description='Build Tornasole binaries')
parser.add_argument('--tag', type=str, default='latest',
                    help='Pass the tag to upload the image to ECR with. '
                         'You might want to set the tag to latest for '
                         'final images.')
parser.add_argument('--single-process', action='store_true',
                    dest='single_process', default=False)
parser.add_argument('--logs-path', type=str, default='bin/sagemaker-containers/logs/')
args = parser.parse_args()

FRAMEWORK_VERSIONS = {'mxnet': '1.4.1', 'tensorflow': '1.13.1', 'pytorch': '1.1.0'}

if not os.path.exists(args.logs_path):
    os.makedirs(args.logs_path)

processes = []
for f, v in FRAMEWORK_VERSIONS.items():
    p = Process(target=build_container, args=(f, v, args))
    p.start()
    print('Started building container for {}. You can find the log at {}.log'
          .format(f, os.path.join(args.logs_path, f)))
    if args.single_process:
        p.join()
    else:
        processes.append(p)

if not args.single_process:
    for p in processes:
        p.join()

