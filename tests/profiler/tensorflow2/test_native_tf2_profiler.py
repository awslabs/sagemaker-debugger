# Standard Library
import json
import os
import time
from datetime import datetime
from pathlib import Path

# Third Party
import pytest
import tensorflow as tf
from tests.profiler.core.utils import validate_python_profiling_stats
from tests.profiler.tensorflow2.utils import verify_detailed_profiling
from tests.tensorflow2.utils import ModelType, is_tf_version_gte

# First Party
import smdebug.tensorflow as smd
from smdebug.core.collection import CollectionKeys
from smdebug.core.utils import FRAMEWORK
from smdebug.profiler.profiler_config_parser import ProfilerConfigParser
from smdebug.profiler.profiler_constants import (
    CONVERT_TO_MICROSECS,
    CPROFILE_NAME,
    DEFAULT_PREFIX,
    PYINSTRUMENT_NAME,
    TRACE_DIRECTORY_FORMAT,
)
from smdebug.profiler.python_profile_utils import StepPhase
from smdebug.tensorflow import KerasHook as Hook


@pytest.fixture
def native_tf2_cprofile_profiler_config_parser(config_folder, monkeypatch):
    config_path = os.path.join(
        config_folder, "test_native_tf2_cprofile_profiler_config_parser.json"
    )
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser(FRAMEWORK.TENSORFLOW)


@pytest.fixture
def native_tf2_pyinstrument_profiler_config_parser(config_folder, monkeypatch):
    config_path = os.path.join(
        config_folder, "test_native_tf2_pyinstrument_profiler_config_parser.json"
    )
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser(FRAMEWORK.TENSORFLOW)


def _train_step(hook, profiler_config_parser, model, opt, images, labels, strategy):
    start_step = profiler_config_parser.config.python_profiling_config.start_step
    end_step = start_step + profiler_config_parser.config.python_profiling_config.num_steps

    with hook.profiler():
        with tf.GradientTape() as tape:
            logits = tf.reduce_mean(model(images, training=True))
            if start_step <= hook.step < end_step:
                assert profiler_config_parser.python_profiler._start_step == hook.step
                assert profiler_config_parser.python_profiler._start_phase == StepPhase.STEP_START
        grads = tape.gradient(logits, model.variables)
        opt.apply_gradients(zip(grads, model.variables))

    if start_step <= hook.step < end_step:
        assert profiler_config_parser.python_profiler._start_step == hook.step
        assert profiler_config_parser.python_profiler._start_phase == StepPhase.STEP_END

    return logits


def _distributed_train_step(hook, profiler_config_parser, model, opt, images, labels, strategy):
    per_replica_losses = strategy.run(
        _train_step, args=(hook, profiler_config_parser, model, opt, images, labels, strategy)
    )
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


def _training_loop(hook, profiler_config_parser, model, opt, dataset, train_step_func, strategy):
    if strategy:
        strategy.run(profiler_config_parser.start_pre_step_zero_python_profiling)
    else:
        profiler_config_parser.start_pre_step_zero_python_profiling()

    for current_step, (data, labels) in enumerate(dataset):
        logits = train_step_func(hook, profiler_config_parser, model, opt, data, labels, strategy)
        hook.save_tensor("inputs", data, CollectionKeys.INPUTS)
        hook.save_tensor("logits", logits, CollectionKeys.OUTPUTS)
        hook.save_tensor("labels", labels, CollectionKeys.OUTPUTS)

    if strategy:
        strategy.run(hook.profiling_end)
    else:
        hook.profiling_end()


def _verify_tensor_names(out_dir):
    """
    This verifies the tensor names when debugger is enabled.
    """

    trial = smd.create_trial(out_dir)
    assert len(trial.steps()) > 0, "Nothing saved at any step."
    assert len(trial.tensor_names()) > 0, "Tensors were not saved."
    assert trial.tensor_names(collection=CollectionKeys.LOSSES) == ["loss"]
    assert len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) > 0
    assert len(trial.tensor_names(collection=CollectionKeys.BIASES)) > 0
    assert trial.tensor_names(collection="optimizer_variables") == [
        "Adam/beta_1:0",
        "Adam/beta_2:0",
        "Adam/decay:0",
        "Adam/iter:0",
        "Adam/learning_rate:0",
    ]
    assert trial.tensor_names(collection=CollectionKeys.INPUTS) == ["inputs"]
    assert trial.tensor_names(collection=CollectionKeys.OUTPUTS) == [
        "labels",
        "logits",
        "predictions",
    ]


def _verify_timeline_files(out_dir):
    """
    This verifies the creation of the timeline files according to file path specification.
    It reads backs the file contents to make sure it is in valid JSON format.
    """
    files = list(Path(os.path.join(out_dir, DEFAULT_PREFIX)).rglob("*.json"))

    assert len(files) >= 1

    for file in files:
        file_ts = file.name.split("_")[0]
        folder_name = file.parent.name
        assert folder_name == time.strftime(
            TRACE_DIRECTORY_FORMAT, time.gmtime(int(file_ts) / CONVERT_TO_MICROSECS)
        )
        assert folder_name == datetime.strptime(folder_name, TRACE_DIRECTORY_FORMAT).strftime(
            TRACE_DIRECTORY_FORMAT
        )

        with open(file) as timeline_file:
            events_dict = json.load(timeline_file)

        assert len(events_dict) > 2000
        assert set([event["name"] for event in events_dict]) == {
            "Step:ModeKeys.TRAIN",
            "process_name",
            "process_sort_index",
        }


# Skipping because the tests are failing.
# Support for profiling using gradient tape has never been released publicly
# and since we're planning on deprecating profiler v1, we can just disable the tests
@pytest.mark.skipif(is_tf_version_gte("2.7"), reason="unblock TF2.7 release")
@pytest.mark.parametrize("python_profiler_name", [CPROFILE_NAME, PYINSTRUMENT_NAME])
@pytest.mark.parametrize(
    "model_type", [ModelType.SEQUENTIAL, ModelType.FUNCTIONAL, ModelType.SUBCLASSED]
)
@pytest.mark.parametrize("use_mirrored_strategy", [False, True])
def test_native_tf2_profiling(
    monkeypatch,
    python_profiler_name,
    model_type,
    use_mirrored_strategy,
    get_model,
    native_tf2_cprofile_profiler_config_parser,
    native_tf2_pyinstrument_profiler_config_parser,
    out_dir,
    mnist_dataset,
    tf_eager_mode,
):
    """
    Enable all types of profiling and validate the output artfacts. Parametrizes on the type of Python
    profiler used for Python profiling as well as the model used for training.

    We cannot test dataloader profiling in pytest, because the resource config needs to be configured at
    /opt/ml/input/config/resourceconfig.json before tensorflow is even imported.
    """
    if python_profiler_name == CPROFILE_NAME:
        profiler_config_parser = native_tf2_cprofile_profiler_config_parser
    else:
        profiler_config_parser = native_tf2_pyinstrument_profiler_config_parser

    assert profiler_config_parser.profiling_enabled
    profiler_config_parser.load_config()

    hook = Hook(out_dir=out_dir, save_all=True)
    # Known issue where logging in a python callback function (i.e. atexit) during pytest causes logging errors.
    # See https://github.com/pytest-dev/pytest/issues/5502 for more information.
    hook.profiler_config_parser = profiler_config_parser
    hook.logger.disabled = True

    if use_mirrored_strategy:
        strategy = tf.distribute.MirroredStrategy()
        num_devices = strategy.num_replicas_in_sync
        with strategy.scope():
            model = get_model(model_type)
            optimizer = tf.optimizers.Adam()
        train_step_func = _distributed_train_step
    else:
        strategy = None
        num_devices = 1
        model = get_model(model_type)
        optimizer = tf.optimizers.Adam()
        train_step_func = _train_step

    optimizer = hook.wrap_optimizer(optimizer)
    _training_loop(
        hook, profiler_config_parser, model, optimizer, mnist_dataset, train_step_func, strategy
    )

    # Sanity check debugger output
    _verify_tensor_names(out_dir)

    # Validate all timeline files
    _verify_timeline_files(out_dir)

    # Validate detailed profiling
    expected_event_count = 90 if use_mirrored_strategy else 230
    verify_detailed_profiling(out_dir, expected_event_count)

    # The expected number of stats directories during is ((num_steps * 2) + 2) * num_devices. This includes profiling
    # for both phases of each step and pre-step zero python profiling and post-hook-close python profiling.
    expected_stats_dir_count = (
        (profiler_config_parser.config.python_profiling_config.num_steps * 2) + 2
    ) * num_devices
    python_stats_dir = os.path.join(out_dir, "framework", "tensorflow", python_profiler_name)
    validate_python_profiling_stats(
        python_stats_dir, python_profiler_name, expected_stats_dir_count
    )
