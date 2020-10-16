# Standard Library
import json
import os
import pstats
import time
from os import environ, makedirs, walk
from os.path import abspath, basename, dirname, exists, getsize, join

# Third Party
import boto3
import pytest
import yaml
from sagemaker.debugger import Rule, rule_configs
from sagemaker.profiler import ProfilerConfig
from sagemaker.pytorch import PyTorch

# First Party
from smdebug.core.logger import get_logger
from smdebug.profiler.algorithm_metrics_reader import S3AlgorithmMetricsReader
from smdebug.profiler.analysis.python_profile_analysis import PyinstrumentAnalysis, cProfileAnalysis
from smdebug.profiler.profiler_constants import (
    CPROFILE_NAME,
    CPROFILE_STATS_FILENAME,
    PYINSTRUMENT_HTML_FILENAME,
    PYINSTRUMENT_JSON_FILENAME,
    PYINSTRUMENT_NAME,
)

# YAML files INDICES
CPU_JOBS_INDEX = 1
GPU_JOBS_INDEX = 2
JOB_NAME_INDEX = 0
ENABLE_INDEX = 1
JOB_SCRIPT_INDEX = 2
ROLE_INDEX = 3
INSTANCE_TYPE_INDEX = 4
INSTANCE_COUNT_INDEX = 5
PROFILER_PARAMS_INDEX = 6
EXPECTED_FILE_NUM_INDEX = 7
HYPERPARAMS_INDEX = 8

# Other CONSTANTS and INDICES
S3_PATH_BUCKET_INDEX = 2
S3_PATH_FOLDER_INDEX = 3
FILE_TIMESTAMP_OFFSET = 20
FILE_SIZE_OFFSET = 500
FILE_ROTATION_OFFSET = 1
TRACE_FILE_NUM_OFFSET = 1
TRACE_FILE_TIMESTAMP_INDEX = 0
TRACE_FILE_NODE_ID_INDEX = 1
MICROS_FACTOR = 10 ** 6

# Profiler Constants
DEFAULT_FILE_SIZE_LIMIT = 10485760  # 10 MB
DEFAULT_FILE_ROTATION_LIMIT = 60 * (10 ** 6)  # 60 seconds

logger = get_logger()

framework = "pytorch"


def _download_artifacts(s3_path, out_dir):
    bucket_name = s3_path.split("/")[S3_PATH_BUCKET_INDEX]
    directory_name = "/".join(s3_path.split("/")[S3_PATH_FOLDER_INDEX:]) + "/framework/"

    s3_resource = boto3.resource("s3")
    s3_bucket = s3_resource.Bucket(bucket_name)
    for obj in s3_bucket.objects.filter(Prefix=directory_name):

        obj_file_path = join(out_dir, obj.key)
        if not exists(dirname(obj_file_path)):
            makedirs(dirname(obj_file_path))
        s3_bucket.download_file(obj.key, obj_file_path)

    local_framework_dir = join(out_dir, directory_name)
    print(f"Training Job artifacts are downloaded to here - {abspath(local_framework_dir)}")
    return local_framework_dir


def _get_estimator_list(index, job_type):
    print(f"Getting estimators for job type {job_type}")
    fhandle = open("tests/sagemaker/pytorch_profiler_tests_config.yaml")
    config_file = yaml.load(fhandle, Loader=yaml.FullLoader)
    fhandle.close()
    estimator_list = []
    profiler_params_list = []
    expected_num_trace_file_list = []
    hyper_params = {}
    for job in config_file[index][job_type]:
        job_name = job[JOB_NAME_INDEX]
        is_enabled = job[ENABLE_INDEX]
        if not is_enabled:
            continue
        job_script = job[JOB_SCRIPT_INDEX]
        job_role = job[ROLE_INDEX]
        instance_type = job[INSTANCE_TYPE_INDEX]
        instance_count = job[INSTANCE_COUNT_INDEX]["instance_count"]
        profiler_params = job[PROFILER_PARAMS_INDEX]["profiler_params"]
        expected_num_trace_file_list.append(
            job[EXPECTED_FILE_NUM_INDEX]["expected_values_in_test_artifacts"]
        )

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
        hyper_params = job[HYPERPARAMS_INDEX]["hyper_params"]
        estimator = PyTorch(
            role=job_role,
            base_job_name=job_name,
            train_instance_count=instance_count,
            train_instance_type=instance_type,
            image_name=image_uri,
            entry_point=job_script,
            framework_version="1.5.1",
            profiler_config=profiler_config,
            py_version="py3",
            rules=rules,
            source_dir=f"{os.getcwd()}/tests/sagemaker/scripts/pytorch",
            hyperparameters=hyper_params,
        )
        estimator.fit(wait=False)
        estimator_list.append(estimator)
    print(f"Starting {len(estimator_list)} jobs for job type {job_type}")
    return zip(estimator_list, profiler_params_list, expected_num_trace_file_list)


def _validate_trace_files(tracefiles, profiler_config):
    """
    Checks for three things -
    1. size of every file is less than the size_limit
    2. The difference in timestamps of two consecutive files are not greater than rotation_limit
    3. The end time of every event in a given file does not exceed the timestamp on the respective filenames.
    """
    file_size_limit = int(
        float(profiler_config.get("RotateMaxFileSizeInBytes", DEFAULT_FILE_SIZE_LIMIT))
    )
    file_rotation_limit = int(
        float(profiler_config.get("RotateFileCloseIntervalInSeconds", DEFAULT_FILE_ROTATION_LIMIT))
    )

    timestamps = {}
    for cur_file in tracefiles:
        # If there are badly formed json trace files, this try catch block will catch it.
        try:
            with open(cur_file) as fhandle:
                data = json.load(fhandle)
        except:
            print(f"Badly formed json trace file: {cur_file}")
            return False

        # Files are rotated when size of the current file exceeds the file_size_limit.
        # So here we check if size of the file is less than or equal to the file_size_limit + 500 bytes offset
        cur_file_size = getsize(cur_file)
        assert (
            cur_file_size <= file_size_limit + FILE_SIZE_OFFSET
        ), f"{cur_file} has file size {cur_file_size} bytes, it has exceeded file size limit {file_size_limit} bytes."
        file_timestamp = int(cur_file.split("/")[-1].split("_")[TRACE_FILE_TIMESTAMP_INDEX])
        file_node_id = cur_file.split("/")[-1].split("_")[TRACE_FILE_NODE_ID_INDEX]
        timestamps[file_node_id] = timestamps.get(file_node_id, []) + [file_timestamp]
        start_time = data[0]["args"]["start_time_since_epoch_in_micros"]

        # Checks if the end time of every event in a given file does not exceed the timestamp on the
        # respective filenames.
        for item in data:
            if "ph" not in item:
                continue
            if item.get("ph") is not "X":
                continue
            event_start = start_time + item.get("ts")
            event_end = event_start + item.get("dur")

            assert event_end <= file_timestamp + FILE_TIMESTAMP_OFFSET, (
                f"In file: {cur_file} \nfor event {item}, \nthe event end time: {event_end} exceeds the filename "
                f"timestamp {file_timestamp}"
            )

        # Files are rotated when the duration of the current file exceeds the file_rotation_limit.
        # So here we check if in each of the training nodes the difference in timestamps of two consecutive
        # files are not greater than rotation_limit + FILE_ROTATION_OFFSET second offset
        for k, v in timestamps.items():
            v.sort()
            for i in range(1, len(v)):
                assert (
                    v[i] - v[i - 1]
                ) / MICROS_FACTOR <= file_rotation_limit + FILE_ROTATION_OFFSET, f"Timestamps of the trace files: {timestamps}. \nFile timestamps are incorrect, it has breached the file rotation limit: {file_rotation_limit} for the timestamps: {v[i - 1]} and {v[i]}"


def _validate_python_stats_files(python_profile_stats):
    """
    For each python stats file downloaded, validate it by doing the following:
    1. If it is a file generated by cProfile (python_stats), then load it into memory as a pStats object to ensure that
        it is valid.
    2. If it is a file generated by pyinstrument (python_stats.json), then load it into memory as a JSON dictionary to
        ensure that it is valid.
    3. Otherwise, an unexpected file was dumped and the test should fail.
    """
    for stats_file in python_profile_stats:
        stats_file_path = stats_file.stats_path
        if basename(stats_file_path) == CPROFILE_STATS_FILENAME:
            assert pstats.Stats(
                stats_file_path
            ), f"cProfile stats at {stats_file_path} failed validation!"
        elif basename(stats_file_path) == PYINSTRUMENT_JSON_FILENAME:
            with open(stats_file_path, "r") as f:
                assert json.load(f), f"Pyinstrument stats at {stats_file_path} failed validation!"
        elif basename(stats_file_path) == PYINSTRUMENT_HTML_FILENAME:
            # We don't really have an easy way to validate HTML right now, since HTML can be rendered even with errors
            # and the HTML generated by pyinstrument is no exception.
            continue
        else:
            assert (
                False
            ), f"Found an unexpected file when validating python stats: {stats_file_path}"


def _run_verify_job(estimator, profiler_config, expected_num_trace_file, out_dir):
    job_name = estimator.latest_training_job.name
    print(f"\nRunning training job - {job_name}")
    path = estimator.latest_job_profiler_artifacts_path()
    print(f"Profiler metrics for the current training job will be uploaded here - {path}")
    client = estimator.sagemaker_session.sagemaker_client
    description = client.describe_training_job(TrainingJobName=job_name)

    while description["TrainingJobStatus"] == "InProgress":
        if path:
            framework_metrics_reader = S3AlgorithmMetricsReader(path)
            framework_metrics_reader.refresh_event_file_list()
            current_timestamp = framework_metrics_reader.get_timestamp_of_latest_available_file()
            assert current_timestamp is not None
        description = client.describe_training_job(TrainingJobName=job_name)
        time.sleep(20)

    # If the training job has failed or been stopped, then fail the test.
    assert description["TrainingJobStatus"] not in ("Failed", "Stopped")

    if profiler_config:
        assert path

        framework_dir = _download_artifacts(path, out_dir)
        pevents_dir = join(framework_dir, "pevents")
        python_stats_dir = join(framework_dir, framework, "{profiler_name}")

        python_tracefiles = [
            join(root, name)
            for root, _, files in walk(pevents_dir)
            for name in files
            if name.endswith("pythontimeline.json")
        ]
        print(f"Number of generated python trace files {len(python_tracefiles)}")
        if expected_num_trace_file["python_trace_file_count"] == 0:
            assert len(python_tracefiles) == 0
        else:
            assert len(python_tracefiles) > 0
        python_tracefiles.sort()

        if "herring_trace_file_count" in expected_num_trace_file:
            herring_tracefiles = [
                join(root, name)
                for root, _, files in walk(pevents_dir)
                for name in files
                if name.endswith("herring_timeline.json")
            ]
            print(f"Number of herring timeline files {len(herring_tracefiles)}")
            assert len(herring_tracefiles) >= expected_num_trace_file["herring_trace_file_count"]
            herring_tracefiles.sort()
            print("Validating the generated herring trace files...")
            _validate_trace_files(herring_tracefiles, profiler_config)

        if "horovod_trace_file_count" in expected_num_trace_file:
            horovod_tracefiles = [
                join(root, name)
                for root, _, files in walk(pevents_dir)
                for name in files
                if name.endswith("horovod_timeline.json")
            ]
            print(f"Number of horovod timeline files {len(horovod_tracefiles)}")
            assert len(horovod_tracefiles) >= expected_num_trace_file["horovod_trace_file_count"]
            horovod_tracefiles.sort()
            print("Validating the generated horovod trace files...")
            _validate_trace_files(horovod_tracefiles, profiler_config)

        python_analysis_class = (
            cProfileAnalysis
            if eval(profiler_config.get("PythonProfilingConfig", "{}")).get(
                "ProfilerName", CPROFILE_NAME
            )
            == CPROFILE_NAME
            else PyinstrumentAnalysis
        )
        python_stats_dir = python_stats_dir.format(profiler_name=python_analysis_class.name)
        makedirs(
            python_stats_dir, exist_ok=True
        )  # still needs to be created if python profiling is disabled
        python_analysis = python_analysis_class(local_profile_dir=python_stats_dir)
        python_profile_stats = python_analysis.python_profile_stats
        print(f"Number of generated python stats files {len(python_profile_stats)}")
        assert len(python_profile_stats) == expected_num_trace_file["python_stats_file_count"]

        print("Validating the generated trace files...")
        _validate_trace_files(python_tracefiles, profiler_config)
        print("Validating the generated python profile stats files...")
        _validate_python_stats_files(python_profile_stats)
        print("SMProfiler trace files validated.")
    else:
        assert not path


@pytest.mark.parametrize(
    "cpu_estimator_and_config", _get_estimator_list(CPU_JOBS_INDEX, "cpu_jobs")
)
def test_cpu_jobs(cpu_estimator_and_config, out_dir):
    estimator, profiler_params, expected_num_trace_file = cpu_estimator_and_config
    _run_verify_job(estimator, profiler_params, expected_num_trace_file, out_dir)


@pytest.mark.parametrize(
    "gpu_estimator_and_config", _get_estimator_list(GPU_JOBS_INDEX, "gpu_jobs")
)
def test_gpu_jobs(gpu_estimator_and_config, out_dir):
    estimator, profiler_params, expected_num_trace_file = gpu_estimator_and_config
    _run_verify_job(estimator, profiler_params, expected_num_trace_file, out_dir)
