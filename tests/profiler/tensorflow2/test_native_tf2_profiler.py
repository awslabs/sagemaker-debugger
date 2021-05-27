# Standard Library
import os

# Third Party
import pytest
import tensorflow as tf
from tests.profiler.core.utils import validate_python_profiling_stats
from tests.tensorflow2.utils import ModelType

# First Party
import smdebug.tensorflow as smd
from smdebug.core.collection import CollectionKeys
from smdebug.core.utils import FRAMEWORK
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
    return ProfilerConfigParser(FRAMEWORK.TENSORFLOW)


@pytest.fixture
def native_tf2_pyinstrument_profiler_config_parser(config_folder, monkeypatch):
    config_path = os.path.join(
        config_folder, "test_native_tf2_pyinstrument_profiler_config_parser.json"
    )
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser(FRAMEWORK.TENSORFLOW)


def _helper_native_tf2_gradtape(out_dir, model, dataset, profiler_config_parser):
    def get_grads(images, labels):
        return model(images, training=True)

    @tf.function
    def train_step(images, labels):
        return tf.reduce_mean(get_grads(images, labels))

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


@pytest.mark.parametrize("python_profiler_name", [CPROFILE_NAME, PYINSTRUMENT_NAME])
@pytest.mark.parametrize(
    "model_type", [ModelType.SEQUENTIAL, ModelType.FUNCTIONAL, ModelType.SUBCLASSED]
)
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
    if model_type == ModelType.SEQUENTIAL:
        model = tf2_mnist_sequential_model
    elif model_type == ModelType.FUNCTIONAL:
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

    _helper_native_tf2_gradtape(out_dir, model, mnist_dataset, profiler_config_parser)

    # Sanity check debugger output
    _verify_tensor_names(out_dir)

    # The expected number of stats directories during is (num_steps * 2) + 2. This includes profiling for both
    # phases of each step and pre-step zero python profiling and post-hook-close python profiling.
    expected_stats_dir_count = (
        profiler_config_parser.config.python_profiling_config.num_steps * 2
    ) + 2
    python_stats_dir = os.path.join(out_dir, "framework", "tensorflow", python_profiler_name)
    validate_python_profiling_stats(
        python_stats_dir, python_profiler_name, expected_stats_dir_count
    )
