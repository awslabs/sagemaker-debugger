# Standard Library
import json
import os
import pstats

# Third Party
import pytest
import tensorflow as tf

# First Party
import smdebug.tensorflow as smd
from smdebug.core.collection import CollectionKeys
from smdebug.core.utils import Framework
from smdebug.profiler.profiler_config_parser import ProfilerConfigParser
from smdebug.profiler.profiler_constants import (
    CPROFILE_NAME,
    CPROFILE_STATS_FILENAME,
    PYINSTRUMENT_HTML_FILENAME,
    PYINSTRUMENT_JSON_FILENAME,
    PYINSTRUMENT_NAME,
)
from smdebug.profiler.python_profile_utils import StepPhase
from smdebug.tensorflow import KerasHook as Hook


@pytest.fixture
def native_tf2_cprofile_profiler_config_parser(config_folder, monkeypatch):
    config_path = os.path.join(
        config_folder, "test_native_tf2_cprofile_profiler_config_parser.json"
    )
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser(Framework.TENSORFLOW)


@pytest.fixture
def native_tf2_pyinstrument_profiler_config_parser(config_folder, monkeypatch):
    config_path = os.path.join(
        config_folder, "test_native_tf2_pyinstrument_profiler_config_parser.json"
    )
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser(Framework.TENSORFLOW)


def _helper_native_tf2_gradtape(out_dir, model, dataset, tf_eager_mode, profiler_config_parser):
    def get_grads(images, labels):
        return model(images, training=True)

    @tf.function
    def train_step_noneager(images, labels):
        return tf.reduce_mean(get_grads(images, labels))

    train_step = train_step_noneager

    hook = Hook(out_dir=out_dir, save_all=True)
    # Known issue where logging in a python callback function (i.e. atexit) during pytest causes logging errors.
    # See https://github.com/pytest-dev/pytest/issues/5502 for more information.
    hook.logger.disabled = True
    hook.profiler_config_parser = profiler_config_parser

    start_step = profiler_config_parser.config.python_profiling_config.start_step
    end_step = start_step + profiler_config_parser.config.python_profiling_config.num_steps

    opt = tf.keras.optimizers.Adam()
    hook.wrap_optimizer(opt)

    for current_step, (data, labels) in enumerate(dataset):
        with hook.profiler():
            labels = tf.one_hot(labels, depth=10)
            with tf.GradientTape() as tape:
                logits = train_step(data, labels)
                if start_step <= current_step < end_step:
                    assert profiler_config_parser.python_profiler._start_step == current_step
                    assert (
                        profiler_config_parser.python_profiler._start_phase == StepPhase.STEP_START
                    )
            grads = tape.gradient(logits, model.variables)
            opt.apply_gradients(zip(grads, model.variables))

            hook.save_tensor("inputs", data, CollectionKeys.INPUTS)
            hook.save_tensor("logits", logits, CollectionKeys.OUTPUTS)
            hook.save_tensor("labels", labels, CollectionKeys.OUTPUTS)

        if start_step <= current_step < end_step:
            assert profiler_config_parser.python_profiler._start_step == current_step
            assert profiler_config_parser.python_profiler._start_phase == StepPhase.STEP_END
    # required for these tests since this normally gets called in the cleanup process and we need to stop any ongoing
    # profiling and collect post-hook-close Python profiling stats
    hook.profiling_end()
    _verify_tensor_names(out_dir, tf_eager_mode)


def _verify_tensor_names(out_dir, tf_eager_mode):
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


def _verify_python_profiling(profiler_name, out_dir, profiler_config_parser):
    """
    This executes a TF2 native training script with profiler or both profiler and debugger,
    enables python profiling by step, and verifies the python profiling's steps and expected output files.
    """

    if profiler_name == CPROFILE_NAME:
        allowed_files = [CPROFILE_STATS_FILENAME]

    if profiler_name == PYINSTRUMENT_NAME:
        allowed_files = [PYINSTRUMENT_JSON_FILENAME, PYINSTRUMENT_HTML_FILENAME]

    python_stats_dir = os.path.join(out_dir, "framework", "tensorflow", profiler_name)

    assert os.path.isdir(python_stats_dir)

    for node_id in os.listdir(python_stats_dir):
        node_dir_path = os.path.join(python_stats_dir, node_id)
        stats_dirs = os.listdir(node_dir_path)

        for stats_dir in stats_dirs:
            print(stats_dir)

        # The expected number of stats directories during is (num_steps * 2) + 1. This includes profiling for both
        # phases of each step and pre-step zero python profiling and post-hook-close python profiling.
        assert (
            len(stats_dirs)
            == profiler_config_parser.config.python_profiling_config.num_steps * 2 + 2
        )

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


@pytest.mark.parametrize("python_profiler_name", [CPROFILE_NAME, PYINSTRUMENT_NAME])
@pytest.mark.parametrize("model_type", ["sequential", "functional", "subclassed"])
def test_native_tf2_profiling(
    python_profiler_name,
    model_type,
    tf2_mnist_sequential_model,
    tf2_mnist_functional_model,
    tf2_mnist_subclassed_model,
    native_tf2_cprofile_profiler_config_parser,
    native_tf2_pyinstrument_profiler_config_parser,
    out_dir,
    mnist_dataset,
    tf_eager_mode,
):
    if model_type == "sequential":
        model = tf2_mnist_sequential_model
    elif model_type == "functional":
        model = tf2_mnist_functional_model
    else:
        model = tf2_mnist_subclassed_model

    if python_profiler_name == CPROFILE_NAME:
        profiler_config_parser = native_tf2_cprofile_profiler_config_parser
    else:
        profiler_config_parser = native_tf2_pyinstrument_profiler_config_parser

    assert profiler_config_parser.profiling_enabled
    profiler_config_parser.load_config()
    profiler_config_parser.start_pre_step_zero_python_profiling()

    _helper_native_tf2_gradtape(
        out_dir, model, mnist_dataset, tf_eager_mode, profiler_config_parser
    )
    _verify_python_profiling(python_profiler_name, out_dir, profiler_config_parser)
