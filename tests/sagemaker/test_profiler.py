# Standard Library
import json
import shutil
import time
from os import environ, makedirs, walk
from os.path import abspath, dirname, exists, getsize, join

# Third Party
import boto3
import yaml
from sagemaker.debugger import Rule, rule_configs
from sagemaker.profiler import ProfilerConfig
from sagemaker.tensorflow import TensorFlow

# First Party
from smdebug.core.logger import get_logger
from smdebug.profiler.algorithm_metrics_reader import S3AlgorithmMetricsReader

JOB_NAME_INDEX = 0
ENABLE_INDEX = 1
JOB_SCRIPT_INDEX = 2
ROLE_INDEX = 3
INSTANCE_TYPE_INDEX = 4
INSTANCE_COUNT_INDEX = 5
PROFILER_PARAMS_INDEX = 6

logger = get_logger()


def _download_artifacts(s3_path):
    bucket_name = s3_path.split("/")[2]
    directory_name = "/".join(s3_path.split("/")[2:]) + "/framework/pevents/"

    s3_resource = boto3.resource("s3")
    s3_bucket = s3_resource.Bucket(bucket_name)
    for obj in s3_bucket.objects.filter(Prefix=directory_name):
        if not exists(dirname(obj.key)):
            makedirs(dirname(obj.key))
        s3_bucket.download_file(obj.key, obj.key)
    return abspath(directory_name)


def _cleanup_folder(dir_path):
    if exists(dir_path):
        shutil.rmtree(dir_path)


def _get_estimator_list(index, job_type):
    fhandle = open("tests/sagemaker/profiler_tests_config.yaml")
    config_file = yaml.load(fhandle, Loader=yaml.FullLoader)
    fhandle.close()
    estimator_list = []
    profiler_params_list = []

    for job in config_file[index][job_type]:
        job_name = job[JOB_NAME_INDEX]
        is_enabled = job[ENABLE_INDEX]
        if not is_enabled:
            continue
        job_script = job[JOB_SCRIPT_INDEX]
        job_role = job[ROLE_INDEX]
        instance_type = job[INSTANCE_TYPE_INDEX]
        instance_count = job[INSTANCE_COUNT_INDEX]
        profiler_params = job[PROFILER_PARAMS_INDEX]["profiler_params"]

        profiler_config = None
        if profiler_params:
            profiler_config = ProfilerConfig(
                profiling_interval_millis=500, profiling_parameters=profiler_params
            )
        profiler_params_list.append(profiler_params)

        distributions = None
        if job_type == "cpu_jobs":
            image_uri = environ["ENV_CPU_TRAIN_IMAGE"]
            if instance_count > 1:
                distributions = {"mpi": {"enabled": False, "processes_per_host": 1}}
        else:
            image_uri = environ["ENV_GPU_TRAIN_IMAGE"]
            if instance_count > 1:
                distributions = {
                    "mpi": {
                        "enabled": True,
                        "processes_per_host": 1,
                        "custom_mpi_options": "-verbose --NCCL_DEBUG=INFO -x OMPI_MCA_btl_vader_single_copy_mechanism=none",
                    }
                }

        rules = [Rule.sagemaker(rule_configs.vanishing_gradient())]
        estimator = TensorFlow(
            role=job_role,
            base_job_name=job_name,
            train_instance_count=instance_count,
            train_instance_type=instance_type,
            image_name=image_uri,
            entry_point=job_script,
            framework_version="2.2",
            profiler_config=profiler_config,
            py_version="py3",
            script_mode=True,
            rules=rules,
            distributions=distributions,
        )
        estimator_list.append(estimator)
    return estimator_list, profiler_params_list


def _validate_trace_files(tracefiles, profiler_config):
    """
    Checks for three things -
    1. size of every file is less than the size_limit
    2. The difference in timestamps of two consecutive files are not greater than rotation_limit
    3. The end time of every event in a given file does not exceed the timestamp on the respective filenames.
    """
    file_size_limit = profiler_config.get("RotateMaxFileSizeInBytes", 10485760)
    file_rotation_limit = profiler_config.get("RotateFileCloseIntervalInSeconds", 60 * 10 ** 6)
    timestamps = []
    for cur_file in tracefiles:
        with open(cur_file) as fhandle:
            data = json.load(fhandle)
        # checks if size of every file is less than the size_limit
        if getsize(cur_file) > file_size_limit:
            return False
        file_timestamp = int(cur_file.split("/")[-1].split("_")[0])
        timestamps.append(file_timestamp)
        start_time = data[0]["args"]["start_time_since_epoch_in_micros"]
        # Checks if the end time of every event in a given file does not exceed the timestamp on the respective filenames.
        for item in data:
            if "ph" not in item:
                continue
            if item.get("ph") is not "X":
                continue
            event_start = start_time + item.get("ts")
            event_end = event_start + item.get("dur")
            if event_end > file_timestamp:
                return False
    # checks if the difference in timestamps of two consecutive files are not greater than rotation_limit
    timestamps.sort()
    for i in range(1, len(timestamps)):
        if (timestamps[i] - timestamps[i - 1]) > file_rotation_limit:
            return False
    return True


def _run_verify_job(estimator, profiler_config):
    sagemaker_client = boto3.client("sagemaker")
    estimator.fit(wait=False)

    job_name = estimator.latest_training_job.name
    print(f"Running training job - {job_name}")
    path = estimator.latest_job_profiler_artifacts_path()
    print(f"Profiler metrics for the current training job will be uploaded here - {path}")
    client = estimator.sagemaker_session.sagemaker_client
    description = client.describe_training_job(TrainingJobName=job_name)

    while description["TrainingJobStatus"] == "InProgress":
        if path:
            framework_metrics_reader = S3AlgorithmMetricsReader(path)
            framework_metrics_reader.refresh_event_file_list()
            current_timestamp = framework_metrics_reader.get_timestamp_of_latest_available_file()
        description = client.describe_training_job(TrainingJobName=job_name)
        time.sleep(20)

    if description["TrainingJobStatus"] == "Failed":
        assert False

    if profiler_config:
        assert path
        dir_path = _download_artifacts(path)
        python_tracefiles = [
            join(root, name)
            for root, _, files in walk(dir_path)
            for name in files
            if name.endswith("pythontimeline.json")
        ]
        print(f"Number of generated python trace files {len(python_tracefiles)}")

        framework_tracefiles = [
            join(root, name)
            for root, _, files in walk(dir_path)
            for name in files
            if name.endswith("model_timeline.json")
        ]
        print(f"Number of generated framework trace files {len(framework_tracefiles)}")

        assert _validate_trace_files(python_tracefiles, profiler_config)
        assert _validate_trace_files(framework_tracefiles, profiler_config)
        _cleanup_folder(dir_path.split("/")[0])
    else:
        assert not path


def test_cpu_jobs():
    estimator_list, profiler_params_list = _get_estimator_list(1, "cpu_jobs")

    for estimator, profiler_params in zip(estimator_list, profiler_params_list):
        _run_verify_job(estimator, profiler_params)


def test_gpu_jobs():
    estimator_list, profiler_params_list = _get_estimator_list(2, "gpu_jobs")

    for estimator, profiler_params in zip(estimator_list, profiler_params_list):
        _run_verify_job(estimator, profiler_params)
