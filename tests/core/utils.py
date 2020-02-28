# First Party
# Standard Library
import glob
import json
import logging
import os
import shutil
from os.path import join

from smdebug.core.collection import Collection
from smdebug.core.collection_manager import CollectionManager
from smdebug.core.config_constants import (
    DEFAULT_COLLECTIONS_FILE_NAME,
    DEFAULT_SAGEMAKER_METRICS_PATH,
)
from smdebug.core.modes import ModeKeys
from smdebug.core.reader import FileReader
from smdebug.core.sagemaker_utils import is_sagemaker_job
from smdebug.trials import create_trial


def write_dummy_collection_file(trial):
    cm = CollectionManager()
    cm.create_collection("default")
    cm.add(Collection(trial))
    cm.export(trial, DEFAULT_COLLECTIONS_FILE_NAME)


def delete_local_trials(local_trials):
    for trial in local_trials:
        shutil.rmtree(trial)


# need this to seek to the right file offset for test output verification
metrics_file_position = 0


def check_metrics_file(save_steps, saved_scalars=None):
    """
    Check the SageMaker metrics file to ensure that all the scalars saved using
    save_scalar(sm_metrics=True) or mentioned through SM_METRICS collections, have been saved.
    """
    global metrics_file_position
    if is_sagemaker_job():
        METRICS_DIR = os.environ.get(DEFAULT_SAGEMAKER_METRICS_PATH)
        if not METRICS_DIR:
            logging.warning("SageMaker Metric Directory not specified")
            return
        file_name = "{}/{}.json".format(METRICS_DIR, str(os.getpid()))
        scalarnames = set()

        import collections

        train_metric = collections.defaultdict(list)
        eval_metric = collections.defaultdict(list)

        with open(file_name) as fp:
            # since SM metrics expects all metrics to be written in 1 file, seeking to
            # the right offset for the purpose of this test - so that the metrics logged in
            # the corresponding test are verified
            fp.seek(metrics_file_position)
            for line in fp:
                data = json.loads(line)
                assert data["IterationNumber"] != -1  # iteration number should not be -1
                metric_name = data["MetricName"]
                if "TRAIN" in metric_name:
                    train_metric[metric_name].append(data["IterationNumber"])
                    scalarnames.add(metric_name.rstrip("_TRAIN"))
                elif "EVAL" in metric_name:
                    eval_metric[metric_name].append(data["IterationNumber"])
                    scalarnames.add(metric_name.rstrip("_EVAL"))
                else:
                    scalarnames.add(
                        metric_name.rstrip("_GLOBAL")
                    )  # check the scalar saved using save_scalar()
            metrics_file_position = fp.tell()
        assert scalarnames

        if saved_scalars:
            assert len(set(saved_scalars) & set(scalarnames)) > 0

        # check if all metrics have been written at the expected step number
        for train_data in train_metric:
            assert len(set(save_steps["TRAIN"]) & set(train_metric[train_data])) == len(
                save_steps["TRAIN"]
            )
        for eval_data in eval_metric:
            assert len(set(save_steps["EVAL"]) & set(eval_metric[eval_data])) == len(
                save_steps["EVAL"]
            )


def check_trials(out_dir, save_steps, saved_scalars=None):
    """
    Create trial to check if non-scalar data is written as per save config and
    check whether all the scalars written through save_scalar have been saved.
    """
    trial = create_trial(path=out_dir, name="test output")
    assert trial
    tensor_list = trial.tensor_names()
    for tname in tensor_list:
        if tname not in saved_scalars:
            train_steps = trial.tensor(tname).steps(mode=ModeKeys.TRAIN)
            eval_steps = trial.tensor(tname).steps(mode=ModeKeys.EVAL)

            # check if all tensors have been saved according to save steps
            assert len(set(save_steps["TRAIN"]) & set(train_steps)) == len(save_steps["TRAIN"])
            if eval_steps:  # need this check for bias and gradients
                assert len(set(save_steps["EVAL"]) & set(eval_steps)) == len(save_steps["EVAL"])
    scalar_list = trial.tensor_names(regex="^scalar")
    if saved_scalars:
        assert len(set(saved_scalars) & set(scalar_list)) == len(saved_scalars)


def verify_files(out_dir, save_config, saved_scalars=None):
    """
    Analyze the tensors saved and verify that metrics are stored correctly in the
    SM metrics json file
    """

    # Retrieve save_step for verification in the trial and the JSON file
    save_config_train_steps = save_config.get_save_config(ModeKeys.TRAIN).save_steps
    if not save_config_train_steps:
        save_interval = save_config.get_save_config(ModeKeys.TRAIN).save_interval
        save_config_train_steps = [i for i in range(0, 10, save_interval)]
    save_config_eval_steps = save_config.get_save_config(ModeKeys.EVAL).save_steps
    if not save_config_eval_steps:
        save_interval = save_config.get_save_config(ModeKeys.EVAL).save_interval
        save_config_eval_steps = [i for i in range(0, 10, save_interval)]

    save_steps = {"TRAIN": save_config_train_steps, "EVAL": save_config_eval_steps}

    check_trials(out_dir, save_steps, saved_scalars)
    check_metrics_file(save_steps, saved_scalars)


"""
    Read the scalar events from tfevents files.
    Test and assert on following:
    1. The names of scalars in 'saved_scalars' match with the names in tfevents.
    2. The timestamps along with the 'saved_scalars' match with timestamps saved in tfevents
    3. The values of 'saved_scalars' match with the values saved in tfevents.
"""


def check_tf_events(out_dir, saved_scalars=None):
    # Read the events from all the saved steps
    fs = glob.glob(join(out_dir, "events", "*", "*.tfevents"), recursive=True)
    events = list()
    for f in fs:
        fr = FileReader(f)
        events += fr.read_events(regex_list=["scalar"])

    # Create a dict of scalar events.
    scalar_events = dict()
    for x in events:
        event_name = str(x["name"])
        if event_name not in scalar_events:
            scalar_events[event_name] = list()
        scalar_events[event_name].append((x["timestamp"], x["value"]))

    for scalar_name in saved_scalars:
        assert scalar_name in scalar_events
        (stored_timestamp, stored_value) = scalar_events[scalar_name][0]
        (recorded_timestamp, recorded_value) = saved_scalars[scalar_name]
        assert recorded_timestamp == stored_timestamp
        assert recorded_value == stored_value[0]
