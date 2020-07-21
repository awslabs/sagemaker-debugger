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


def _download_artifacts(s3_path):
    bucket_name = s3_path.split("/")[S3_PATH_BUCKET_INDEX]
    directory_name = "/".join(s3_path.split("/")[S3_PATH_FOLDER_INDEX:]) + "/framework/"

    s3_resource = boto3.resource("s3")
    s3_bucket = s3_resource.Bucket(bucket_name)
    for obj in s3_bucket.objects.filter(Prefix=directory_name):
        if not exists(dirname(obj.key)):
            makedirs(dirname(obj.key))
        s3_bucket.download_file(obj.key, obj.key)
    print(f"Training Job artifacts are downloaded to here - {abspath(directory_name)}")
    return abspath(directory_name)


def _cleanup_folder(dir_path):
    print(f"Cleaning up this path - {dir_path}\n")
    if exists(dir_path):
        shutil.rmtree(dir_path)


def _compare_with_tolerance(val1, val2, comparison_op, tolerance):
    if comparison_op == "GreaterThan":
        return val1 > (val2 + tolerance)
    elif comparison_op == "EqualTo":
        return abs(val1 - val2) <= tolerance
    else:
        raise Exception("Invalid comparison operator")


def _get_estimator_list(index, job_type):
    fhandle = open("tests/sagemaker/profiler_tests_config.yaml")
    config_file = yaml.load(fhandle, Loader=yaml.FullLoader)
    fhandle.close()
    estimator_list = []
    profiler_params_list = []
    expected_num_trace_file_list = []

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
    return estimator_list, profiler_params_list, expected_num_trace_file_list


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
            # TODO: replace continue with return False after the framework bug is fixed
            # https://github.com/leleamol/sagemaker-debugger-private/issues/72
            continue
            # return False

        # Files are rotated when size of the current file exceeds the file_size_limit.
        # So here we check if size of the file is less than or equal to the file_size_limit + 500 bytes offset
        cur_file_size = getsize(cur_file)
        if _compare_with_tolerance(cur_file_size, file_size_limit, "GreaterThan", FILE_SIZE_OFFSET):
            print(
                f"{cur_file} has file size {cur_file_size} bytes, it has exceeded file size limit {file_size_limit} bytes."
            )
            return False
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
            # TODO : There is flakiness in some cases where there is a difference of <5 microsecond between
            # event_end time and file_timestamp. I am adding an offset of 5. But this has to be fixed on
            # framework side. The occurence of this bug is flaky.
            if _compare_with_tolerance(
                event_end, file_timestamp, "GreaterThan", FILE_TIMESTAMP_OFFSET
            ):
                print(
                    f"In file: {cur_file} \nfor event {item}, \nthe event end time: {event_end} exceeds the filename timestamp {file_timestamp}"
                )
                return False

    # Files are rotated when the duration of the current file exceeds the file_rotation_limit.
    # So here we check if in each of the training nodes the difference in timestamps of two consecutive
    # files are not greater than rotation_limit + FILE_ROTATION_OFFSET second offset
    for k, v in timestamps.items():
        v.sort()
        for i in range(1, len(v)):
            if _compare_with_tolerance(
                (v[i] - v[i - 1]) / MICROS_FACTOR,
                file_rotation_limit,
                "GreaterThan",
                FILE_ROTATION_OFFSET,
            ):
                print(
                    f"Timestamps of the trace files: {timestamps}. \nFile timestamps are incorrect, it has breached the file rotation limit: {file_rotation_limit} for the timestamps: {v[i - 1]} and {v[i]}"
                )
                return False
    return True


def _run_verify_job(estimator, profiler_config, expected_num_trace_file):
    sagemaker_client = boto3.client("sagemaker")
    estimator.fit(wait=False)

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
    if (
        description["TrainingJobStatus"] == "Failed"
        or description["TrainingJobStatus"] == "Stopped"
    ):
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
        if expected_num_trace_file["python_trace_file_count"] == 0:
            assert len(python_tracefiles) == 0
        else:
            assert len(python_tracefiles) > 0

        framework_tracefiles = [
            join(root, name)
            for root, _, files in walk(dir_path)
            for name in files
            if name.endswith("model_timeline.json")
        ]
        print(f"Number of generated framework trace files {len(framework_tracefiles)}")
        if expected_num_trace_file["framework_trace_file_count"] == 0:
            assert len(framework_tracefiles) == 0
        else:
            assert len(framework_tracefiles) > 0

        try:
            print("Validating the generated trace files...")
            assert _validate_trace_files(python_tracefiles, profiler_config)
            assert _validate_trace_files(framework_tracefiles, profiler_config)
            print("SMProfiler trace files validated.")
        except Exception as e:
            print("Validating trace files failed with error: ", e)
            assert False
        finally:
            # this will clean up all the downloaded test artifacts.
            _cleanup_folder(dir_path[: dir_path.index("/profiler-output")])
    else:
        assert not path


def test_cpu_jobs():
    estimator_list, profiler_params_list, expected_values_in_test_artifacts = _get_estimator_list(
        CPU_JOBS_INDEX, "cpu_jobs"
    )

    for estimator, profiler_params, expected_num_trace_file in zip(
        estimator_list, profiler_params_list, expected_values_in_test_artifacts
    ):
        _run_verify_job(estimator, profiler_params, expected_num_trace_file)


def test_gpu_jobs():
    estimator_list, profiler_params_list, expected_values_in_test_artifacts = _get_estimator_list(
        GPU_JOBS_INDEX, "gpu_jobs"
    )

    for estimator, profiler_params, expected_num_trace_file in zip(
        estimator_list, profiler_params_list, expected_values_in_test_artifacts
    ):
        _run_verify_job(estimator, profiler_params, expected_num_trace_file)


if __name__ == "__main__":
    test_cpu_jobs()
    test_gpu_jobs()
