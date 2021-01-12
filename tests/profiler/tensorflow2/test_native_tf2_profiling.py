# Standard Library
import os
import json
import time
from datetime import datetime
from pathlib import Path
import pstats

# Third Party
import tensorflow as tf
import pytest

# First Party
import smdebug.tensorflow as smd
from smdebug.core.collection import CollectionKeys
from smdebug.core.tfevent.timeline_file_writer import TimelineFileWriter
from smdebug.profiler.profiler_config_parser import ProfilerConfigParser
from smdebug.profiler.profiler_constants import TENSORBOARDTIMELINE_SUFFIX
from smdebug.profiler.tf_profiler_parser import TensorboardProfilerEvents
from smdebug.tensorflow import KerasHook as Hook
from smdebug.profiler.profiler_constants import (
    CONVERT_TO_MICROSECS,
    DEFAULT_PREFIX,
    TRACE_DIRECTORY_FORMAT,
    CPROFILE_NAME,
    CPROFILE_STATS_FILENAME,
    PYINSTRUMENT_HTML_FILENAME,
    PYINSTRUMENT_JSON_FILENAME,
    PYINSTRUMENT_NAME,
)
from smdebug.profiler.python_profiler import PythonProfiler
from smdebug.profiler.python_profiler import (
    PyinstrumentPythonProfiler,
    cProfilePythonProfiler,
    cProfileTimer,
)
from smdebug.profiler.python_profile_utils import PythonProfileModes, StepPhase

@pytest.fixture()
def tf2_profiler_config_parser_by_step(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "test_tf2_profiler_config_parser_by_step.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser()


@pytest.fixture()
def tf2_python_cprofiler_config_parser_by_step(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "test_tf2_python_profiler_cprofiler_config_parser_by_step.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser()


@pytest.fixture()
def tf2_python_pyinstrument_config_parser_by_step(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "test_tf2_python_profiler_pyinstrument_config_parser_by_step.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser()


@pytest.fixture()
def tf2_profiler_config_parser_by_time(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "test_tf2_profiler_config_parser_by_time.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser()


@pytest.fixture
def test_framework():
    return "test-framework"


@pytest.fixture()
def cprofile_python_profiler(out_dir, test_framework):
    return cProfilePythonProfiler(out_dir, test_framework, cProfileTimer.TOTAL_TIME)


@pytest.fixture()
def pyinstrument_python_profiler(out_dir, test_framework):
    return PyinstrumentPythonProfiler(out_dir, test_framework)


@pytest.fixture()
def framework_dir(out_dir, test_framework):
    return "{0}/framework/{1}".format(out_dir, test_framework)


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


def helper_native_tf2_profiler_debugger(trial_dir, hook):

    def get_grads(images, labels):
        return model(images, training=True)

    @tf.function
    def train_step(images, labels):
        return tf.reduce_mean(get_grads(images, labels))

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), _ = mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_train[..., tf.newaxis] / 255, tf.float32), tf.cast(y_train, tf.int64))
    )
    dataset = dataset.shuffle(1000).batch(128)
    model = create_model()
    opt = tf.keras.optimizers.Adam()
    hook.wrap_optimizer(opt)

    n_epochs = 1
    for epoch in range(n_epochs):
        for data, labels in dataset:
            labels = tf.one_hot(labels, depth=10)
            hook.start_profiling_start_train_batch()
            with hook.wrap_tape(tf.GradientTape()) as tape:
                logits = train_step(data, labels)
            grads = tape.gradient(logits, model.variables)
            opt.apply_gradients(zip(grads, model.variables))
            hook.save_tensor("inputs", data, CollectionKeys.INPUTS)
            hook.save_tensor("logits", logits, CollectionKeys.OUTPUTS)
            hook.save_tensor("labels", labels, CollectionKeys.OUTPUTS)
            hook.start_profiling_end_train_batch()
    hook.stop_profiling_end_of_training()

    model.save(trial_dir, save_format="tf")

    trial = smd.create_trial(trial_dir)
    assert trial.tensor_names(collection=CollectionKeys.LOSSES) == ["loss"]
    assert trial.tensor_names(collection=CollectionKeys.WEIGHTS) == [
        "weights/dense/kernel:0",
        "weights/dense_1/kernel:0",
    ]
    assert trial.tensor_names(collection=CollectionKeys.BIASES) == [
        "weights/dense/bias:0",
        "weights/dense_1/bias:0",
    ]
    assert trial.tensor_names(collection=CollectionKeys.OPTIMIZER_VARIABLES) == [
        "Adam/beta_1:0",
        "Adam/beta_2:0",
        "Adam/decay:0",
        "Adam/iter:0",
        "Adam/learning_rate:0",
    ]
    assert trial.tensor_names(collection=CollectionKeys.INPUTS) == ["inputs"]
    assert trial.tensor_names(collection=CollectionKeys.OUTPUTS) == ["labels", "logits"]


def helper_native_tf2_profiler(trial_dir, hook):

    def get_grads(images, labels):
        return model(images, training=True)

    @tf.function
    def train_step(images, labels):
        return tf.reduce_mean(get_grads(images, labels))

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), _ = mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_train[..., tf.newaxis] / 255, tf.float32), tf.cast(y_train, tf.int64))
    )
    dataset = dataset.shuffle(1000).batch(128)
    model = create_model()
    opt = tf.keras.optimizers.Adam()
    hook.wrap_optimizer(opt)


    # print('\nTraining script: ', hook.python_profiler)
    n_epochs = 1
    for epoch in range(n_epochs):
        for data, labels in dataset:
            labels = tf.one_hot(labels, depth=10)
            hook.start_profiling_start_train_batch()
            with tf.GradientTape() as tape:
                logits = train_step(data, labels)
            grads = tape.gradient(logits, model.variables)
            opt.apply_gradients(zip(grads, model.variables))
            hook.start_profiling_end_train_batch()
    hook.stop_profiling_end_of_training()


@pytest.mark.skip_if_non_eager
def test_native_tf2_profiler_by_step_profiler_debugger(tf2_profiler_config_parser_by_step, out_dir):
    """
    This test executes a TF2 native training script, enables detailed TF profiling by step, and
    verifies the number of events.
    """
    assert tf2_profiler_config_parser_by_step.profiling_enabled

    hook = Hook(out_dir=out_dir)
    helper_native_tf2_profiler_debugger(trial_dir=out_dir, hook=hook)

    t_events = TensorboardProfilerEvents()

    # get tensorboard timeline files
    files = []
    for path in Path(tf2_profiler_config_parser_by_step.config.local_path + "/framework").rglob(
        f"*{TENSORBOARDTIMELINE_SUFFIX}"
    ):
        files.append(path)

    assert len(files) == 1

    trace_file = str(files[0])
    t_events.read_events_from_file(trace_file)

    all_trace_events = t_events.get_all_events()
    num_trace_events = len(all_trace_events)

    print(f"Number of events read = {num_trace_events}")

    # The number of events is varying by a small number on
    # consecutive runs. Hence, the approximation in the below asserts.
    assert num_trace_events >= 230


def test_native_tf2_profiler_by_step_profiler(tf2_profiler_config_parser_by_step, out_dir):
    """
    This test executes a TF2 native training script, enables detailed TF profiling by step, and
    verifies the number of events.
    """
    assert tf2_profiler_config_parser_by_step.profiling_enabled

    hook = Hook(out_dir=out_dir)
    helper_native_tf2_profiler(trial_dir=out_dir, hook=hook)

    t_events = TensorboardProfilerEvents()

    # get tensorboard timeline files
    files = []
    for path in Path(tf2_profiler_config_parser_by_step.config.local_path + "/framework").rglob(
        f"*{TENSORBOARDTIMELINE_SUFFIX}"
    ):
        files.append(path)

    assert len(files) == 1

    trace_file = str(files[0])
    t_events.read_events_from_file(trace_file)

    all_trace_events = t_events.get_all_events()
    num_trace_events = len(all_trace_events)

    print(f"Number of events read = {num_trace_events}")

    # The number of events is varying by a small number on
    # consecutive runs. Hence, the approximation in the below asserts.
    assert num_trace_events >= 230


@pytest.mark.skip_if_non_eager
def test_native_tf2_profiler_by_time_profiler_debugger(tf2_profiler_config_parser_by_time, out_dir):
    """
    This test executes a TF2 native training script, enables detailed TF profiling by time, and
    verifies the number of events.
    """
    assert tf2_profiler_config_parser_by_time.profiling_enabled

    hook = Hook(out_dir=out_dir)
    helper_native_tf2_profiler_debugger(trial_dir=out_dir, hook=hook)

    # get tensorboard timeline files
    files = []
    for path in Path(tf2_profiler_config_parser_by_time.config.local_path + "/framework").rglob(
        f"*{TENSORBOARDTIMELINE_SUFFIX}"
    ):
        files.append(path)

    assert len(files) == 1

    trace_file = str(files[0])
    t_events = TensorboardProfilerEvents()

    t_events.read_events_from_file(trace_file)

    all_trace_events = t_events.get_all_events()
    num_trace_events = len(all_trace_events)

    print(f"Number of events read = {num_trace_events}")

    # The number of events is varying by a small number on
    # consecutive runs. Hence, the approximation in the below asserts.
    assert num_trace_events >= 700

@pytest.mark.skip_if_non_eager
def test_native_tf2_profiler_by_time_profiler(tf2_profiler_config_parser_by_time, out_dir):
    """
    This test executes a TF2 native training script, enables detailed TF profiling by time, and
    verifies the number of events.
    """
    assert tf2_profiler_config_parser_by_time.profiling_enabled

    hook = Hook(out_dir=out_dir)
    helper_native_tf2_profiler(trial_dir=out_dir, hook=hook)

    # get tensorboard timeline files
    files = []
    for path in Path(tf2_profiler_config_parser_by_time.config.local_path + "/framework").rglob(
        f"*{TENSORBOARDTIMELINE_SUFFIX}"
    ):
        files.append(path)

    assert len(files) == 1

    trace_file = str(files[0])
    t_events = TensorboardProfilerEvents()

    t_events.read_events_from_file(trace_file)

    all_trace_events = t_events.get_all_events()
    num_trace_events = len(all_trace_events)

    print(f"Number of events read = {num_trace_events}")

    # The number of events is varying by a small number on
    # consecutive runs. Hence, the approximation in the below asserts.
    assert num_trace_events >= 700


# @pytest.mark.parametrize("use_pyinstrument", [False, True])
# @pytest.mark.parametrize("steps", [(1, 2), (1, 5)])
def test_native_python_profiling_cprofiler(
    out_dir, tf2_python_cprofiler_config_parser_by_step
):
    assert tf2_python_cprofiler_config_parser_by_step.profiling_enabled

    config = tf2_python_cprofiler_config_parser_by_step.config
    print('\npath: ', os.environ["SMPROFILER_CONFIG_PATH"])
    print('\nname: ', config.python_profiling_config.name)
    print('\nstart_step: ', config.python_profiling_config.start_step)
    print('\nnum_steps: ', config.python_profiling_config.num_steps)

    python_profiler = PythonProfiler.get_python_profiler(config, "tensorflow")
    print('\ntest function: ', python_profiler)
    start_step = config.python_profiling_config.start_step
    num_steps = config.python_profiling_config.num_steps
    end_step = start_step + num_steps

    profiler_name = CPROFILE_NAME
    allowed_files = [CPROFILE_STATS_FILENAME]
    python_stats_dir = os.path.join(out_dir, 'framework/', 'tensorflow/', profiler_name)

    hook = Hook(out_dir=out_dir)
    hook.python_profiler = python_profiler
    helper_native_tf2_profiler(trial_dir=out_dir, hook=hook)

    # Test that directory and corresponding files exist.
    assert os.path.isdir(python_stats_dir)

    for node_id in os.listdir(python_stats_dir):
        node_dir_path = os.path.join(python_stats_dir, node_id)
        stats_dirs = os.listdir(node_dir_path)
        assert len(stats_dirs) == (end_step - start_step) * 2 + 1

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


def test_native_python_profiling_pyinstrument(
    out_dir, tf2_python_pyinstrument_config_parser_by_step
):
    assert tf2_python_pyinstrument_config_parser_by_step.profiling_enabled

    config = tf2_python_pyinstrument_config_parser_by_step.config
    print('\npath: ', os.environ["SMPROFILER_CONFIG_PATH"])
    print('\nname: ', config.python_profiling_config.profiler_name)
    print('\nstart_step: ', config.python_profiling_config.start_step)
    print('\nnum_steps: ', config.python_profiling_config.num_steps)

    # python_profiler = PythonProfiler.get_python_profiler(config, "tensorflow")
    start_step = config.python_profiling_config.start_step
    num_steps = config.python_profiling_config.num_steps
    end_step = start_step + num_steps

    profiler_name = PYINSTRUMENT_NAME
    allowed_files = [PYINSTRUMENT_JSON_FILENAME, PYINSTRUMENT_HTML_FILENAME]
    python_stats_dir = os.path.join(out_dir, 'framework/', 'tensorflow/', profiler_name)

    hook = Hook(out_dir=out_dir)
    helper_native_tf2_profiler(trial_dir=out_dir, hook=hook)

    # Test that directory and corresponding files exist.
    assert os.path.isdir(python_stats_dir)

    for node_id in os.listdir(python_stats_dir):
        node_dir_path = os.path.join(python_stats_dir, node_id)
        stats_dirs = os.listdir(node_dir_path)
        assert len(stats_dirs) == (end_step - start_step) * 2 + 1

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


def test_create_timeline_file(simple_profiler_config_parser, out_dir):
    """
    This test is meant to test successful creation of the timeline file according to file path specification.
    $ENV_BASE_FOLDER/framework/pevents/$START_TIME_YYMMDDHR/$FILEEVENTSTARTTIMEUTCINEPOCH_
    {$ENV_NODE_ID_4digits0padded}_pythontimeline.json
    It reads backs the file contents to make sure it is in valid JSON format.
    """
    assert simple_profiler_config_parser.profiling_enabled

    hook = Hook(out_dir=out_dir)
    helper_native_tf2_profiler(trial_dir=out_dir, hook=hook)

    files = []
    for path in Path(out_dir + "/" + DEFAULT_PREFIX).rglob("*.json"):
        files.append(path)

    assert len(files) == 1

    file_ts = files[0].name.split("_")[0]
    folder_name = files[0].parent.name
    assert folder_name == time.strftime(
        TRACE_DIRECTORY_FORMAT, time.gmtime(int(file_ts) / CONVERT_TO_MICROSECS)
    )
    assert folder_name == datetime.strptime(folder_name, TRACE_DIRECTORY_FORMAT).strftime(
        TRACE_DIRECTORY_FORMAT
    )

    with open(files[0]) as timeline_file:
        events_dict = json.load(timeline_file)

    assert events_dict