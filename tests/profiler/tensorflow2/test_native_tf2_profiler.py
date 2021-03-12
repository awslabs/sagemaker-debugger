# Standard Library
import atexit
import json
import os
import pstats

# Third Party
import pytest
import tensorflow as tf
from tests.profiler.resources.profiler_config_parser_utils import build_metrics_config

# First Party
import smdebug.tensorflow as smd
from smdebug.core.collection import CollectionKeys
from smdebug.profiler.profiler_config_parser import ProfilerConfigParser
from smdebug.profiler.profiler_constants import (
    CPROFILE_NAME,
    CPROFILE_STATS_FILENAME,
    PYINSTRUMENT_HTML_FILENAME,
    PYINSTRUMENT_JSON_FILENAME,
    PYINSTRUMENT_NAME,
)
from smdebug.profiler.python_profile_utils import StepPhase
from smdebug.profiler.python_profiler import PythonProfiler
from smdebug.tensorflow import KerasHook as Hook


@pytest.fixture
def profiler_config_path(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "profiler_config.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    yield config_path
    if os.path.isfile(config_path):
        os.remove(config_path)


def generate_profiler_config_parser(profiling_type, profiler_config_path, profiling_parameters):
    python_profiling_config, detailed_profiling_config = "{}", "{}"

    if profiling_type == "PythonProfiling":
        start_step, num_steps, profiler_name, cprofile_timer = profiling_parameters
        python_profiling_config = build_metrics_config(
            StartStep=start_step,
            NumSteps=num_steps,
            ProfilerName=profiler_name,
            cProfileTimer=cprofile_timer,
        )

    full_config = {
        "ProfilingParameters": {
            "ProfilerEnabled": True,
            "LocalPath": "/tmp/test",
            "PythonProfilingConfig": python_profiling_config,
        }
    }

    with open(profiler_config_path, "w") as f:
        json.dump(full_config, f)

    profiler_config_parser = ProfilerConfigParser()
    assert profiler_config_parser.profiling_enabled

    return profiler_config_parser


def generate_profiler_config_parser_all_params(profiler_config_path, python_profiling_parameters):

    start_step_1, num_steps_1, profiler_name, cprofile_timer = python_profiling_parameters

    python_profiling_config = build_metrics_config(
        StartStep=start_step_1,
        NumSteps=num_steps_1,
        ProfilerName=profiler_name,
        cProfileTimer=cprofile_timer,
    )

    full_config = {
        "ProfilingParameters": {
            "ProfilerEnabled": True,
            "LocalPath": "/tmp/test",
            "PythonProfilingConfig": python_profiling_config,
        }
    }

    with open(profiler_config_path, "w") as f:
        json.dump(full_config, f)

    profiler_config_parser = ProfilerConfigParser()
    assert profiler_config_parser.profiling_enabled

    return profiler_config_parser


def set_up_profiling(profiler_config):
    profiler_config_parser = profiler_config
    python_profiler = None
    if profiler_config_parser.profiling_enabled:
        config = profiler_config_parser.config
        if config.python_profiling_config.is_enabled():
            python_profiler = PythonProfiler.get_python_profiler(config, "tensorflow")
            python_profiler.start_profiling(StepPhase.START)
            atexit.register(python_profiler.stop_profiling, StepPhase.END)
    return profiler_config_parser, python_profiler


def create_model():
    model = tf.keras.models.Sequential(
        [
            # WA for TF issue https://github.com/tensorflow/tensorflow/issues/36279
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return model


def prepare_dataset():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), _ = mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_train[..., tf.newaxis] / 255, tf.float32), tf.cast(y_train, tf.int64))
    )
    dataset = dataset.shuffle(1000).batch(64)
    return dataset


def helper_native_tf2_gradtape(
    hook, tf_eager_mode, python_profiler, enable_debugger, start_step, end_step
):
    def get_grads(images, labels):
        return model(images, training=True)

    @tf.function
    def train_step_noneager(images, labels):
        return tf.reduce_mean(get_grads(images, labels))

    def train_step_eager(images, labels):
        return tf.reduce_mean(get_grads(images, labels))

    train_step = train_step_eager if tf_eager_mode else train_step_noneager

    dataset = prepare_dataset()
    model = create_model()
    opt = tf.keras.optimizers.Adam()
    hook.wrap_optimizer(opt)

    current_step = 0
    n_epochs = 1
    for epoch in range(n_epochs):
        for data, labels in dataset:
            with hook.profiler():
                labels = tf.one_hot(labels, depth=10)
                if enable_debugger:
                    with hook.wrap_tape(tf.GradientTape()) as tape:
                        logits = train_step(data, labels)
                        if python_profiler and start_step <= current_step < end_step:
                            assert python_profiler._start_step == current_step
                            assert python_profiler._start_phase == StepPhase.STEP_START
                    grads = tape.gradient(logits, model.variables)
                    opt.apply_gradients(zip(grads, model.variables))
                    hook.save_tensor("inputs", data, CollectionKeys.INPUTS)
                    hook.save_tensor("logits", logits, CollectionKeys.OUTPUTS)
                    hook.save_tensor("labels", labels, CollectionKeys.OUTPUTS)
                else:
                    with tf.GradientTape() as tape:
                        logits = train_step(data, labels)
                        if python_profiler and start_step <= current_step < end_step:
                            assert python_profiler._start_step == current_step
                            assert python_profiler._start_phase == StepPhase.STEP_START
                    grads = tape.gradient(logits, model.variables)
                    opt.apply_gradients(zip(grads, model.variables))
            if python_profiler and start_step <= current_step < end_step:
                assert python_profiler._start_step == current_step
                assert python_profiler._start_phase == StepPhase.STEP_END
            current_step += 1
    # required for these tests since this normally gets called in the cleanup process and we can't test for artifacts
    # at that point.
    hook.profiling_end()
    if python_profiler:
        assert python_profiler._start_step == current_step - 1
        assert python_profiler._start_phase == StepPhase.STEP_END


def _initiate_python_profiling(profiler_config):
    assert profiler_config.profiling_enabled
    profiler_config_parser, python_profiler = set_up_profiling(profiler_config)
    config = profiler_config_parser.config
    start_step = config.python_profiling_config.start_step
    num_steps = config.python_profiling_config.num_steps
    end_step = start_step + num_steps
    return python_profiler, start_step, end_step


def _train_loop(out_dir, tf_eager_mode, python_profiler, enable_debugger, start_step, end_step):
    hook = Hook(out_dir=out_dir, save_all=True)
    hook.logger.disabled = True
    if python_profiler:
        hook.python_profiler = python_profiler
    helper_native_tf2_gradtape(
        hook, tf_eager_mode, python_profiler, enable_debugger, start_step, end_step
    )
    # Known issue where logging in a python callback function (i.e. atexit) during pytest causes logging errors.
    # See https://github.com/pytest-dev/pytest/issues/5502 for more information.
    hook.logger.disabled = True


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
    assert trial.tensor_names(collection=CollectionKeys.OUTPUTS) == ["labels", "logits"]


def _verify_python_profiling(profiler_name, out_dir, num_steps):
    """
    This executes a TF2 native training script with profiler or both profiler and debugger,
    enables python profiling by step, and verifies the python profiling's steps and expected output files.
    """

    if profiler_name == CPROFILE_NAME:
        allowed_files = [CPROFILE_STATS_FILENAME]

    if profiler_name == PYINSTRUMENT_NAME:
        allowed_files = [PYINSTRUMENT_JSON_FILENAME, PYINSTRUMENT_HTML_FILENAME]

    python_stats_dir = os.path.join(out_dir, "framework/", "tensorflow/", profiler_name)

    assert os.path.isdir(python_stats_dir)

    for node_id in os.listdir(python_stats_dir):
        node_dir_path = os.path.join(python_stats_dir, node_id)
        stats_dirs = os.listdir(node_dir_path)

        # Since python_profiler.stop_profiling for the posthookclose step automatically executed
        # upon normal interpreter termination,
        # the number of the files is num_steps * 2 + 2 - 1.
        assert len(stats_dirs) == num_steps * 2 + 1

        for stats_dir in stats_dirs:
            # Validate that the expected files are in the stats dir
            stats_dir_path = os.path.join(node_dir_path, stats_dir)
            stats_files = os.listdir(stats_dir_path)
            assert set(stats_files) == set(allowed_files)

            # Validate the actual stats files
            for stats_file in stats_files:
                stats_path = os.path.join(stats_dir_path, stats_file)
                if stats_file == CPROFILE_STATS_FILENAME:
                    assert pstats.Stats(stats_path)
                elif stats_file == PYINSTRUMENT_JSON_FILENAME:
                    with open(stats_path, "r") as f:
                        assert json.load(f)


@pytest.mark.parametrize("enable_python_profiling", [CPROFILE_NAME, PYINSTRUMENT_NAME])
@pytest.mark.parametrize("enable_debugger", [False, True])
def test_native_tf2_profiling_debugger(
    enable_python_profiling, enable_debugger, profiler_config_path, out_dir, tf_eager_mode
):
    if enable_python_profiling == CPROFILE_NAME:
        profiler_config_parser = generate_profiler_config_parser(
            "PythonProfiling", profiler_config_path, (5, 2, CPROFILE_NAME, None)
        )
    if enable_python_profiling == PYINSTRUMENT_NAME:
        profiler_config_parser = generate_profiler_config_parser(
            "PythonProfiling", profiler_config_path, (10, 3, PYINSTRUMENT_NAME, None)
        )
    python_profiler, start_step, end_step = _initiate_python_profiling(profiler_config_parser)
    _train_loop(out_dir, tf_eager_mode, python_profiler, enable_debugger, start_step, end_step)
    _verify_python_profiling(enable_python_profiling, out_dir, num_steps=end_step - start_step)

    if enable_debugger:
        _verify_tensor_names(out_dir)
