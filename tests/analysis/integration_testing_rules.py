import os
import sys
# set environment variable values for tornasole logger
os.environ['TORNASOLE_LOG_LEVEL'] = 'DEBUG'
os.environ['TORNASOLE_LOG_ALL_TO_STDOUT'] = 'FALSE'
os.environ['TORNASOLE_LOG_PATH'] = 'tornasole.log'
# if true, block stdout prints on console, if false, show stdout prints on console
stdout = os.environ.get('BLOCK_STDOUT', default='FALSE') == 'TRUE'
# if true, block stderr prints on console, if false, show stderr prints on console
stderr = os.environ.get('BLOCK_STDERR', default='FALSE') == 'TRUE'
# block stdout and stderr on top level scripts, including imported modules
if stdout:
    f = open(os.devnull, 'w')
    sys.stdout = f
if stderr:
    f = open(os.devnull, 'w')
    sys.stderr = f

import shutil
from multiprocessing import *
from tornasole.core.utils import get_logger
import yaml
import time
import asyncio
import aioboto3
from tornasole.core.access_layer.s3handler import S3Handler, ListRequest
from subprocess import Popen, PIPE
from time import sleep
import re
import boto3
import glob

TEST_NAME_INDEX = 0
FRAMEWORK_INDEX = 1
SHOULD_RUN_INDEX = 2
TEST_INFO_INDEX = 3

VALUES_INDEX = 0
SERIAL_MODE_INDEX = 1
PARALLEL_MODE_INDEX = 2
LOCAL_MODE_INDEX = 3
S3_MODE_INDEX = 4

TRAIN_SCRIPT_INDEX = 0
TRAIN_SCRIPT_ARGS_INDEX = 1
TEST_SCRIPT_INDEX = 2
TEST_SCRIPT_ARGS_INDEX = 3
BUCKET = 'integrationtestlog' 


logger = get_logger()

# store path to config file and test mode for testing rule scrip with training script
class TestRules():
    def __init__(self, framework, path_to_config=None, env_dict={}, test_case_list=[], test_case_regex=None):
        """
        :param mode: mode could be either 'tensorflow' or 'mxnet'
        :param path_to_config: the path of config file which contains path to training and test scripts and corresponding arg strings
        """
        self.framework = framework
        # if path_to_config is not defined, then use default dir './config 1.yaml'
        self.path_to_config = path_to_config if path_to_config is not None else './config.yaml'
        # load config file
        f = open(self.path_to_config)
        self.config_file = yaml.load(f)
        self.serial_and_parallel = {}
        self.serial_and_parallel['serial_mode'] = self.config_file[VALUES_INDEX][SERIAL_MODE_INDEX]['serial']
        self.serial_and_parallel['parallel_mode'] = self.config_file[VALUES_INDEX][PARALLEL_MODE_INDEX]['parallel']
        self.local_and_s3 = {}
        self.local_and_s3['local_mode'] = self.config_file[VALUES_INDEX][LOCAL_MODE_INDEX]['local']
        self.local_and_s3['s3_mode'] = self.config_file[VALUES_INDEX][S3_MODE_INDEX]['s3']
        self.stdout_mode = stdout
        self.stderr_mode = stderr
        self.test_cases = test_case_list
        self.test_case_regex = test_case_regex
        self.CI_or_local_mode = os.environ.get('CI_OR_LOCAL', default=env_dict['CI_OR_LOCAL']
            if 'CI_OR_LOCAL' in env_dict and env_dict['CI_OR_LOCAL'] is not None else None)
        if self.CI_or_local_mode != 'LOCAL' and self.CI_or_local_mode is not None:
            raise Exception('Wrong test mode!')
        # if run on local mode, overwrite environment variable values to be local path
        if self.CI_or_local_mode == 'LOCAL':
            # if user doesn't specify environment variable value, then use default path prefix '.'
            os.environ['tf_path'] = env_dict['tf_path'] if 'tf_path' in env_dict and \
                                                           env_dict['tf_path'] is not None else '.'
            os.environ['pytorch_path'] = env_dict['pytorch_path'] if 'pytorch_path' in env_dict and \
                                                                     env_dict['pytorch_path'] is not None else '.'
            os.environ['mxnet_path'] = env_dict['mxnet_path'] if 'mxnet_path' in env_dict and \
                                                                 env_dict['mxnet_path'] is not None else '.'
            os.environ['rules_path'] = env_dict['rules_path'] if 'rules_path' in env_dict and \
                                                                 env_dict['rules_path'] is not None else '.'
            os.environ['core_path'] = env_dict['core_path'] if 'core_path' in env_dict and \
                                                               env_dict['core_path'] is not None else '.'
            os.environ['CODEBUILD_SRC_DIR'] = env_dict['CODEBUILD_SRC_DIR'] if 'CODEBUILD_SRC_DIR' in env_dict and \
                                                               env_dict['CODEBUILD_SRC_DIR'] is not None else '.'
        # create a local folder to store log files
        if not os.path.exists('./integration_test_log'):
            os.mkdir('./integration_test_log/')

    # delete the s3 folders using aioboto3
    async def del_folder(self, bucket, keys):
        loop = asyncio.get_event_loop()
        client = aioboto3.client('s3', loop=loop)
        await asyncio.gather(*[client.delete_object(Bucket=bucket, Key=key) for key in keys])
        await client.close()

    # delete outputs generated by all training processes
    # local_trials: trial dirs on local, e.g., './output/trial'
    def delete_local_trials(self, local_trials):
        for trial in local_trials:
            trial_root = trial.split('/')[1]
            if os.path.exists(trial):
                shutil.rmtree(trial_root)

    # delete the s3 folders using aioboto3
    # s3_trials: trial dirs on s3, e.g., 's3://bucket_name/trial'
    def delete_s3_trials(self, s3_trials):
        s3_handler = S3Handler()
        list_req = []
        bucket_name = ''
        for trial in s3_trials:
            bucket_name = trial.split('/')[2]
            trial_name = trial.split('/')[3]
            list_req.append(ListRequest(Bucket=bucket_name, Prefix=trial_name))
        keys = s3_handler.list_prefixes(list_req)
        # flat nested list
        keys = [item for sublist in keys for item in sublist]
        loop = asyncio.get_event_loop()
        task = loop.create_task(self.del_folder(bucket_name, keys))
        loop.run_until_complete(task)

    def upload_log_to_s3(self, s3_prefix):
        files_to_upload = []
        ##  tornasole.log
        from shutil import copyfile
        s3 = boto3.client('s3')
        copyfile('tornasole.log', 'tornasole-s3.log')
        files_to_upload.append('tornasole-s3.log')
        ##  integration_test_log
        for r, d, f in os.walk('integration_test_log'):
            for file in f:
                files_to_upload.append(os.path.join(r, file))
        ## local_test 
        for r, d, f in os.walk('local_test'):
            for file in f:
                files_to_upload.append(os.path.join(r, file))
        ## upload 
        for log_file in files_to_upload:
            logger.info("Uploading file: {} to s3".format(log_file))
            with open(log_file) as lf:
                s3.put_object(Bucket=BUCKET, Key= s3_prefix + "/" + log_file, Body=lf)


    def delete_local_log(self):
        files = glob.glob('./integration_test_log/*')
        for f in files:
            os.remove(f)

    # run a 'job' in serial. a 'job' is a training/test scripts combination
    def run_job_in_serial(self, path_to_train_script, train_script_args, path_to_test_script,
                          test_script_args, trial_dir, job_name, mode, time_stamp):
        self.run_one_script(path_to_train_script, train_script_args, trial_dir, job_name, mode, time_stamp, 'TRAIN')
        self.run_one_script(path_to_test_script, test_script_args, trial_dir, job_name, mode, time_stamp, 'TEST')

    # run one script only
    def run_one_script(self, path_to_script, script_args, trial_dir, job_name, mode, time_stamp, train_test_str=""):
        # replace $path by environment variable value
        path_prefix = path_to_script.split('/')[0].strip('$')
        # get environment variable's value
        path_prefix = os.environ[path_prefix]
        path_postfix = path_to_script.split('/')[1:]
        path_postfix = '/'.join(path_postfix)
        path_to_script = path_prefix + '/' + path_postfix
        # check test running either on local or s3
        local_or_s3 = ''
        if trial_dir.startswith('s3'):
            local_or_s3 = 's3_mode'
        else:
            local_or_s3 = 'local_mode'
        commands = "python {} --tornasole_path {} {}".format(path_to_script, trial_dir, script_args)
        logger.info("IntegrationTest running command {}".format(commands))
        # use subprocess to execute cmd line prompt
        command_list = commands.split()
        logger.info("command_list : {}".format(command_list))

        # create a subprocess using Popen
        p = Popen(command_list,
                  stdout=PIPE if self.stdout_mode else None,
                  stderr=PIPE if self.stderr_mode else None,
                  env=dict(os.environ,
                           TORNASOLE_LOG_CONTEXT='{}_{}'.format(path_to_script, trial_dir),
                           TORNASOLE_LOG_PATH='./integration_test_log/{}_{}_{}_{}_{}.log'.
                           format(job_name.replace('/', '_'), local_or_s3, mode,
                                  time_stamp, train_test_str),
                           TORNASOLE_LOG_LEVEL='debug',
                           TORNASOLE_LOG_ALL_TO_STDOUT='FALSE'))

        out, error = p.communicate()
        if p.returncode == 0:
            logger.info("script {} of job {} in {} in {} ran {} successfully".format(path_to_script, job_name, local_or_s3, mode, train_test_str))
        
        else:
            logger.error("script {} of job {} in {} in {} {} failed with error {} , "
                         "output is:{}".format(path_to_script, job_name,
                                               local_or_s3, mode, train_test_str,
                                               error, out))
            ## returning exit code
            exit(p.returncode)

    def _is_test_allowed(self, job):
        name = job[TEST_NAME_INDEX]
        in_test_cases = True
        if len(self.test_cases) > 0:
            in_test_cases = name in self.test_cases
            logger.info("Test cases specified, in_test_cases is {} testname:{}".format(in_test_cases, name))
        mathes_regex = True
        if self.test_case_regex is not None:
            if re.match(self.test_case_regex, job[TEST_NAME_INDEX]):
                mathes_regex = True
            else:
                mathes_regex = False
            logger.info("Test regex specified, mathes_regex is {} testname:{}".format(mathes_regex, name))

        is_enabled = job[SHOULD_RUN_INDEX]
        logger.info("Test {} is enabled:{}".format(name, is_enabled))
        is_framework_allowed = (job[FRAMEWORK_INDEX] == self.framework)
        is_allowed = is_enabled and (in_test_cases or mathes_regex) and is_framework_allowed
        logger.info("Test {} is allowed:{}".format(name, is_allowed))
        return is_allowed
    # run 'job's provided by user. a 'job' is a training/test scripts combination
    # mode: testing mode, either 'auto' or 'manual'
    # jobs: a list of lists, the sublist is called a ‘job’
    # each job is run in serial and parallel on both local and s3
    def run_jobs(self):
        process_list = []
        local_trials = set()
        s3_trials = set()
        log_files = []
        trial_dir = ''
        #print(self.__dict__)
        print(os.environ)
        upload_time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

        # execute all the 'job's
        for job in self.config_file:
            # format of a 'job' is:
            # - test_case_name
            # - framework_name
            # - *Enable/*Disable
            # - [<path_train_script>,
            #    <train_script_args>,
            #    <path_test_script>,
            #    <test_script_args>
            #   ]
            framework = job[FRAMEWORK_INDEX]
            if job[TEST_NAME_INDEX] == 'values':
                continue
            ALLOWED_FRAMEWORK = ['tensorflow', 'pytorch', 'mxnet'] ## Note values is first dict in yaml file. It's a hack
            if framework not in ALLOWED_FRAMEWORK:
                raise Exception('Wrong test case category', job[TEST_NAME_INDEX])

            if not self._is_test_allowed(job):
                continue
            # if user has specified regex search for certain test cases, only of these, which are turned on would run
            else:
                job_info = job[TEST_INFO_INDEX]
                for mode_1 in self.serial_and_parallel:
                    if self.serial_and_parallel[mode_1]:
                        time_stamp = time.time()
                        for mode_2 in self.local_and_s3:
                            if self.local_and_s3[mode_2]:
                                if mode_2 == 'local_mode':
                                    trial_dir = './local_test/trial_{}_{}_{}'.format(
                                        job[TEST_NAME_INDEX].replace('/', '_'), mode_1, time_stamp)
                                    local_trials.add(trial_dir)
                                else:
                                    trial_dir = 's3://{}/{}/s3_trials/trial_{}_{}_{}'.format(BUCKET,
                                        upload_time_str, job[TEST_NAME_INDEX].replace('/', '_'), mode_1, time_stamp)
                                    s3_trials.add(trial_dir)
                                if mode_1 == 'serial_mode':
                                    process_list.append(Process(target=self.run_job_in_serial, args=(
                                        job_info[TRAIN_SCRIPT_INDEX], job_info[TRAIN_SCRIPT_ARGS_INDEX],
                                        job_info[TEST_SCRIPT_INDEX], job_info[TEST_SCRIPT_ARGS_INDEX],
                                        trial_dir, job[TEST_NAME_INDEX], mode_1,
                                        time_stamp)))
                                else:
                                    process_list.append(Process(target=self.run_one_script,
                                                                args=(job_info[TRAIN_SCRIPT_INDEX],
                                                                      job_info[TRAIN_SCRIPT_ARGS_INDEX],
                                                                      trial_dir,
                                                                      job[TEST_NAME_INDEX], mode_1,
                                                                      time_stamp, 'TRAIN')))
                                    process_list.append(Process(target=self.run_one_script,
                                                                args=(job_info[TEST_SCRIPT_INDEX],
                                                                      job_info[TEST_SCRIPT_ARGS_INDEX],
                                                                      trial_dir,
                                                                      job[TEST_NAME_INDEX], mode_1,
                                                                      time_stamp, 'TEST')))
                                log_files.append('{}_{}_{}_{}.log'.format(job[TEST_NAME_INDEX].replace('/', '_'),
                                                                        mode_2, mode_1, time_stamp))
        # execute all 'job's in parallel
        for process in process_list:
            process.start()
        ended_processes = set()
        exit_code = 0
        while True:
            if len(ended_processes) == len(process_list):
                break
            for process in process_list:
                if process not in ended_processes and not process.is_alive():
                    ended_processes.add(process)
                    exit_code += process.exitcode
                    logger.info('Process {} ended with exit code {}'.format(process.name, process.exitcode))
                    process.join()
            sleep(2)
        if exit_code > 0:
            # upload all the files to s3 
            self.upload_log_to_s3(upload_time_str) 
            msg = "exit code of pytest run non zero. Please check logs in s3://{}{}".format(BUCKET, upload_time_str)
            assert False, msg
        # once all jobs are finished, delete the outputs on local and s3
        #self.delete_local_trials(local_trials)
        #self.delete_s3_trials(s3_trials)
        # upload log files to s3
        #self.upload_log_to_s3(log_files)
        # delete local log files
        #self.delete_local_log()

# only for codebuilding test
# enable args string with pytest
def test_test_rules(request):
    mode = request.config.getoption('mode')
    path_to_config = request.config.getoption('path_to_config')
    # user may specify environment variable value
    env_dict = {}
    env_dict['tf_path'] = request.config.getoption('tf_path')
    env_dict['pytorch_path'] = request.config.getoption('pytorch_path')
    env_dict['mxnet_path'] = request.config.getoption('mxnet_path')
    env_dict['rules_path'] = request.config.getoption('rules_path')
    env_dict['core_path'] = request.config.getoption('core_path')
    # user may specify ci or local testing mode
    env_dict['CI_OR_LOCAL'] = request.config.getoption('CI_OR_LOCAL')
    # $CODEBUILD_SRC_DIR is used for one-for-all testing
    env_dict['CODEBUILD_SRC_DIR'] = request.config.getoption('CODEBUILD_SRC_DIR')
    # get the test_case_list if user specifies
    test_case_list = request.config.getoption('test_case')
    # get the test_case_regex if user specifies
    test_case_regex = request.config.getoption('test_case_regex')

    TestRules(framework=mode,
              path_to_config=path_to_config,
              env_dict=env_dict,
              test_case_list=test_case_list,
              test_case_regex=test_case_regex).run_jobs()

# test on local machine
# TestRules(framework='pytorch').run_jobs()