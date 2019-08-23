import subprocess
import argparse
from multiprocessing import Process
import os
from time import sleep


FRAMEWORK_VERSIONS = {
    'mxnet': '1.4.1',
    'pytorch': '1.1.0',
    'tensorflow': '1.13.1',
    }


def run_command(command_list, stdout, stderr):
  subprocess.check_call(command_list, stdout=stdout, stderr=stderr)


# you can clean all caches used by docker with `docker system prune -a`
def build_container(framework, version, args):
    command = ['bash', 'bin/sagemaker-containers/{}/{}/build.sh'.format(framework, version), args.tag]
    if not args.single_process:
        with open(os.path.join(args.logs_path, '{}.log'.format(framework)), "w") as logfile:
            run_command(command, stdout=logfile, stderr=logfile)
    else:
        run_command(command, None, None)


parser = argparse.ArgumentParser(description='Build Tornasole binaries')
parser.add_argument('--tag', type=str, default='temp',
                    help='Pass the tag to upload the image to ECR with. '
                         'You might want to set the tag to latest for '
                         'final images.')
parser.add_argument('--single-process', action='store_true',
                    dest='single_process', default=False)
parser.add_argument('--logs-path', type=str, default='bin/sagemaker-containers/logs/')
args = parser.parse_args()

if not os.path.exists(args.logs_path):
    os.makedirs(args.logs_path)

processes = []
for f, v in FRAMEWORK_VERSIONS.items():
    p = Process(name=f, target=build_container, args=(f, v, args))
    p.start()
    print('Started building container for {}. You can find the log at {}.log'
          .format(f, os.path.join(args.logs_path, f)))
    if args.single_process:
        p.join()
    else:
        processes.append(p)

if not args.single_process:
    ended_processes = set()
    while True:
        if len(processes) == len(ended_processes):
            break
        for p in processes:
            if p not in ended_processes and not p.is_alive():
                p.join()
                print(f'Finished process {p.name}')
                ended_processes.add(p)
        sleep(3)

