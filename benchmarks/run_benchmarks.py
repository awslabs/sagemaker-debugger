# Third Party
# Standard Library
import argparse
import json
import os
import tempfile
import time
from datetime import datetime

import boto3
from sagemaker.analytics import TrainingJobAnalytics
from sagemaker.profiler import ProfilerConfig
from sagemaker.tensorflow import TensorFlow

instance_types = ["ml.p3.16xlarge", "ml.p2.8xlarge"]
batch_size = 1024
epoch = 100


def run_perf_jobs():
    role = os.environ["ENV_SAGEMAKER_ROLE"]  # "AmazonSageMaker-ExecutionRole-20200205T150348"
    # 072677473360.dkr.ecr.us-east-1.amazonaws.com/smprofiler-gpu:latest
    image_uri = os.environ["ENV_GPU_TRAIN_IMAGE"]

    experiment_configs = {
        "profiler_fully_enabled": {},
        "system_profiler_enabled": {},
        "framework_profiler_enabled": {},
        "profiler_disabled": {},
        "smdebug_enabled": {},
    }

    for instance_type in instance_types:
        ### System profiler and framework profiler enabled
        estimator = TensorFlow(
            base_job_name="perf-benchmark-sys-fw-profiler-enabled",
            role=role,
            image_name=image_uri,
            train_instance_count=1,
            train_instance_type=instance_type,
            entry_point="resnet50.py",
            source_dir="benchmarks/scripts",
            framework_version="2.2.0",
            py_version="py37",
            profiler_config=ProfilerConfig(
                profiling_interval_millis=500, profiling_parameters={"ProfilerEnabled": "True"}
            ),
            script_mode=True,
            hyperparameters={"batch_size": batch_size, "epoch": epoch},
            metric_definitions=[{"Name": "train:duration", "Regex": "Total_Train_Duration=(.*?);"}],
        )
        estimator.fit(wait=False)
        experiment_configs["profiler_fully_enabled"][
            instance_type
        ] = estimator.latest_training_job.job_name

        ### Only system profiler enabled
        estimator = TensorFlow(
            base_job_name="perf-benchmark-only-sys-profiler-enabled",
            role=role,
            train_instance_count=1,
            train_instance_type=instance_type,
            entry_point="resnet50.py",
            source_dir="benchmarks/scripts",
            framework_version="2.2.0",
            py_version="py37",
            profiler_config=ProfilerConfig(
                profiling_interval_millis=500, profiling_parameters={"ProfilerEnabled": "True"}
            ),
            script_mode=True,
            debugger_hook_config=False,
            hyperparameters={"batch_size": batch_size, "epoch": epoch},
            metric_definitions=[{"Name": "train:duration", "Regex": "Total_Train_Duration=(.*?);"}],
        )
        estimator.fit(wait=False)
        experiment_configs["system_profiler_enabled"][
            instance_type
        ] = estimator.latest_training_job.job_name

        ### Only framework profiler enabled
        estimator = TensorFlow(
            base_job_name="perf-benchmark-only-fw-profiler-enabled",
            role=role,
            image_name=image_uri,
            train_instance_count=1,
            train_instance_type=instance_type,
            entry_point="resnet50.py",
            source_dir="benchmarks/scripts",
            framework_version="2.2.0",
            py_version="py37",
            script_mode=True,
            hyperparameters={
                "batch_size": batch_size,
                "epoch": epoch,
                "write_profiler_config": True,
            },
            metric_definitions=[{"Name": "train:duration", "Regex": "Total_Train_Duration=(.*?);"}],
        )
        estimator.fit(wait=False)
        experiment_configs["framework_profiler_enabled"][
            instance_type
        ] = estimator.latest_training_job.job_name

        ### Baseline1: Run without profiler and with ZCC (default)
        estimator = TensorFlow(
            base_job_name="perf-benchmark-zcc-enabled-profiler-disabled",
            role=role,
            image_name=image_uri,
            train_instance_count=1,
            train_instance_type=instance_type,
            entry_point="resnet50.py",
            source_dir="benchmarks/scripts",
            framework_version="2.2.0",
            py_version="py37",
            script_mode=True,
            hyperparameters={"batch_size": batch_size, "epoch": epoch},
            metric_definitions=[{"Name": "train:duration", "Regex": "Total_Train_Duration=(.*?);"}],
        )
        estimator.fit(wait=False)
        experiment_configs["smdebug_enabled"][
            instance_type
        ] = estimator.latest_training_job.job_name

        ### Baseline2: Run without profiler and smdebug
        estimator = TensorFlow(
            base_job_name="perf-benchmark-sys-fw-profiler-disabled",
            role=role,
            image_name=image_uri,
            train_instance_count=1,
            train_instance_type=instance_type,
            entry_point="resnet50.py",
            source_dir="benchmarks/scripts",
            framework_version="2.2.0",
            py_version="py37",
            script_mode=True,
            debugger_hook_config=False,
            hyperparameters={"batch_size": batch_size, "epoch": epoch},
            metric_definitions=[{"Name": "train:duration", "Regex": "Total_Train_Duration=(.*?);"}],
        )
        estimator.fit(wait=False)
        experiment_configs["profiler_disabled"][
            instance_type
        ] = estimator.latest_training_job.job_name

    def check_job_status(name, results, fhandle):
        client = boto3.client("sagemaker")
        description = client.describe_training_job(TrainingJobName=name)

        while True:
            if description["TrainingJobStatus"] == "Completed":
                start_time = description["CreationTime"]
                end_time = description["TrainingEndTime"]

                # Total time taken by the sagemaker job.
                sagemaker_job_time = end_time - start_time
                start_time_training = description["SecondaryStatusTransitions"][2]["StartTime"]
                end_time_training = description["SecondaryStatusTransitions"][2]["EndTime"]

                # Time taken by sagemaker during training phase.
                # includes - downloading training image and running the script.
                # we do not use this time metric to directly benchmark performance,
                # but we do upload these values as csv file into the S3 bucket in the env var - S3_BUCKET_ARTIFACTS
                sagemaker_training_time = end_time_training - start_time_training

                # Time taken by the script to run training.
                # (This is the time that we will benchmark.)
                script_training_time = (
                    TrainingJobAnalytics(training_job_name=name, metric_names=["train:duration"])
                    .dataframe()["value"]
                    .values[0]
                )
                results[name] = [instance_type, script_training_time]
                print(
                    f"Training Job: {name} with: {experiment_config} run on: {instance_type} with batch size: {batch_size}  and with epochs: {epoch}. Train Script Time: {script_training_time} SageMaker Training Phase Time: {sagemaker_training_time.seconds} SageMaker Training Job Time: {sagemaker_job_time.seconds}"
                )
                fhandle.write(
                    f"{name},{instance_type},{batch_size},{epoch},{script_training_time},{sagemaker_training_time.seconds},{sagemaker_job_time.seconds}\n"
                )
                break
            else:
                print(
                    "Training Job:", name, " Training Job Status", description["TrainingJobStatus"]
                )
                description = client.describe_training_job(TrainingJobName=name)
                if description["TrainingJobStatus"] in ["Stopped", "Failed"]:
                    print(f"Training Job: {name}, has failed.")
                    results[name] = [instance_type, None]
                    fhandle.write(
                        f"{name},{instance_type},{batch_size},{epoch},{None},{None},{None}\n"
                    )
                    break
                time.sleep(100)

    results = {}
    fhandle = tempfile.NamedTemporaryFile(mode="a", delete=False)
    fhandle.write(
        f"Training_Job,instance_type,batch_size,epochs,Training_Loop_Time,SageMaker_Training_Time,SageMaker_Job_Time\n"
    )
    for experiment_config in experiment_configs:
        for instance_type in experiment_configs[experiment_config]:
            name = experiment_configs[experiment_config][instance_type]
            check_job_status(name, results, fhandle)
    fhandle.close()
    s3_client = boto3.client("s3")
    s3_path = os.path.join(
        str(datetime.today().strftime("%Y-%m-%d")), str(time.time()), "performance_metrics.csv"
    )
    print("Uploading metrics to S3 bucket...")  # smprofiler-perf-test-artifacts
    s3_client.upload_file(fhandle.name, os.environ["S3_BUCKET_ARTIFACTS"], s3_path)
    print(results)
    return experiment_configs, results


def upload_metrics(experiment_configs, results):
    """
    This function uploads metrics to Cloudwatch of Tornasole's Prod account.
    """
    cloudwatch_client = boto3.client("cloudwatch")
    run_types = ["profiler_fully_enabled", "system_profiler_enabled", "framework_profiler_enabled"]
    baselines = ["profiler_disabled", "smdebug_enabled"]
    timestamp = datetime.utcnow()
    for instance_type in instance_types:
        comparison_metrics = []
        for run_type in run_types:
            job_name = experiment_configs[run_type][instance_type]
            trainloop_time = results[job_name][1]
            for baseline in baselines:
                baseline_job_name = experiment_configs[baseline][instance_type]
                try:
                    baseline_time = float(results[baseline_job_name][1])
                    metric_value = (trainloop_time / baseline_time - 1) * 100
                    metric_name = (
                        f"Change in TrainingTimeInSeconds for {run_type} compared to {baseline}"
                    )
                    comparison_metrics.append(
                        {"MetricName": metric_name, "Value": metric_value, "Timestamp": timestamp}
                    )
                except:
                    print(
                        f"Training Job: {baseline_job_name} failed, no statistics is available for this job."
                    )

        print(comparison_metrics)
        namespace = f"smprofiler-TF2-resnet50-{instance_type}"
        cloudwatch_client.put_metric_data(Namespace=namespace, MetricData=comparison_metrics)


def create_and_check_alarms(alarm_retries, alarm_interval):
    """
    This function compares the recorded training time, picks up threshold values from alarm_spec.jaon,
    and creates and rises alarms if there is a breach in threshold.
    """
    with open("benchmarks/alarm_spec.json") as json_data:
        parameters = json.load(json_data)

    cloudwatch_client = boto3.client("cloudwatch")
    for alarm_params in parameters:
        put_metric_alarm_response = cloudwatch_client.put_metric_alarm(
            AlarmName=alarm_params["AlarmName"],
            AlarmDescription=alarm_params["AlarmDescription"]
            if "AlarmDescription" in alarm_params
            else alarm_params["AlarmName"],
            ActionsEnabled=True,
            AlarmActions=[alarm_params["AlarmActions"]] if "AlarmActions" in alarm_params else [],
            EvaluationPeriods=alarm_params["EvaluationPeriods"],
            Threshold=float(alarm_params["Threshold"]),
            ComparisonOperator=alarm_params["ComparisonOperator"],
            Period=alarm_params["Period"],
            Statistic=alarm_params["Statistic"],
            MetricName=alarm_params["MetricName"],
            Namespace=alarm_params["Namespace"],
        )
        print(put_metric_alarm_response)

    alarm_names = [alarm_params["AlarmName"] for alarm_params in parameters]
    build_status = 0
    print("Waiting to check the alarm status of the different metrics...")
    for retry in range(alarm_retries):
        build_status = 0
        time.sleep(alarm_interval)
        current_timestamp_str = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
        response = cloudwatch_client.describe_alarms(AlarmNames=alarm_names)
        for status in response["MetricAlarms"]:
            alarm_name = status["AlarmName"]
            alarm_status = status["StateValue"]
            threshold = status["Threshold"]
            print(
                f"Trial {retry}: Alarm {alarm_name} is in {alarm_status} state at {current_timestamp_str}  "
                f"Threshold {threshold}"
            )
            if alarm_status != "OK":
                build_status |= 1
        if build_status == 0:
            return build_status
    return build_status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Cloudwatch can take upto 15 minutes to update its alarms, so we will use the following two params to wait and retry while checking for the alarm state of the different metrics.
    parser.add_argument(
        "--retries", type=int, default=3, help="Number of retries to check the alarm state"
    )
    parser.add_argument(
        "--interval", type=int, default=300, help="Interval in seconds in between retries"
    )
    args = parser.parse_args()

    s3_client = boto3.client("s3")
    s3_client.download_file(
        "smdebug-testing", "datasets/cifar-10-python.tar.gz", "/tmp/cifar-10-batches-py.tar.gz"
    )
    os.system("mv /tmp/cifar-10-batches-py.tar.gz benchmarks/scripts/")
    exit_status = 1
    try:
        print("Running the performance benchmarking jobs ...")
        experiment_configs, results = run_perf_jobs()

        print("Uploading the generated metrics to cloudwatch ...")
        upload_metrics(experiment_configs, results)

        print("Checking the status of the alarms in cloudwatch ...")
        exit_status = create_and_check_alarms(args.retries, args.interval)
    except Exception as e:
        print(f"Error : Exception {str(e)}")
    finally:
        exit(exit_status)
