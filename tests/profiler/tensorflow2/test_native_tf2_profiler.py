# Standard Library
import atexit
import json
import os
import pstats
import time
from datetime import datetime
from pathlib import Path

# Third Party
import pytest
import tensorflow as tf
from tests.profiler.resources.profiler_config_parser_utils import build_metrics_config

# First Party
import smdebug.tensorflow as smd
from smdebug.core.collection import CollectionKeys
from smdebug.profiler.profiler_config_parser import ProfilerConfigParser
from smdebug.profiler.profiler_constants import (
    CONVERT_TO_MICROSECS,
    CPROFILE_NAME,
    CPROFILE_STATS_FILENAME,
    DEFAULT_PREFIX,
    PYINSTRUMENT_HTML_FILENAME,
    PYINSTRUMENT_JSON_FILENAME,
    PYINSTRUMENT_NAME,
    TENSORBOARDTIMELINE_SUFFIX,
    TRACE_DIRECTORY_FORMAT,
)
from smdebug.profiler.python_profile_utils import StepPhase
from smdebug.profiler.python_profiler import PythonProfiler
from smdebug.profiler.tf_profiler_parser import TensorboardProfilerEvents
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

    if profiling_type == "DetailedProfiling":
        start_step, num_steps, start_time, duration = profiling_parameters
        detailed_profiling_config = build_metrics_config(
            StartStep=start_step,
            NumSteps=num_steps,
            StartTimeInSecSinceEpoch=start_time,
            DurationInSeconds=duration,
        )

    full_config = {
        "ProfilingParameters": {
            "ProfilerEnabled": True,
            "LocalPath": "/tmp/test",
            "PythonProfilingConfig": python_profiling_config,
            "DetailedProfilingConfig": detailed_profiling_config,
        }
    }

    with open(profiler_config_path, "w") as f:
        json.dump(full_config, f)

    profiler_config_parser = ProfilerConfigParser()
    assert profiler_config_parser.profiling_enabled

    return profiler_config_parser


def generate_profiler_config_parser_all_params(
    profiler_config_path, python_profiling_parameters, detailed_profiling_parameters
):

    start_step_1, num_steps_1, profiler_name, cprofile_timer = python_profiling_parameters
    start_step_2, num_steps_2, start_time, duration = detailed_profiling_parameters

    python_profiling_config = build_metrics_config(
        StartStep=start_step_1,
        NumSteps=num_steps_1,
        ProfilerName=profiler_name,
        cProfileTimer=cprofile_timer,
    )

    detailed_profiling_config = build_metrics_config(
        StartStep=start_step_2,
        NumSteps=num_steps_2,
        StartTimeInSecSinceEpoch=start_time,
        DurationInSeconds=duration,
    )

    full_config = {
        "ProfilingParameters": {
            "ProfilerEnabled": True,
            "LocalPath": "/tmp/test",
            "PythonProfilingConfig": python_profiling_config,
            "DetailedProfilingConfig": detailed_profiling_config,
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
    hook, debugger=False, python_profiler=None, start_step=None, end_step=None
):
    def get_grads(images, labels):
        return model(images, training=True)

    @tf.function
    def train_step(images, labels):
        return tf.reduce_mean(get_grads(images, labels))

    dataset = prepare_dataset()
    model = create_model()
    opt = tf.keras.optimizers.Adam()
    hook.wrap_optimizer(opt)

    current_step = 0
    n_epochs = 1
    for epoch in range(n_epochs):
        for data, labels in dataset:
            hook.profiling_start_batch()
            labels = tf.one_hot(labels, depth=10)
            if debugger:
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
            hook.profiling_end_batch()
            if python_profiler and start_step <= current_step < end_step:
                assert python_profiler._start_step == current_step
                assert python_profiler._start_phase == StepPhase.STEP_END
            current_step += 1
    hook.profiling_end()
    if python_profiler:
        assert python_profiler._start_step == current_step - 1
        assert python_profiler._start_phase == StepPhase.STEP_END


def initiate_python_profiling(profiler_config):
    assert profiler_config.profiling_enabled
    profiler_config_parser, python_profiler = set_up_profiling(profiler_config)
    config = profiler_config_parser.config
    start_step = config.python_profiling_config.start_step
    num_steps = config.python_profiling_config.num_steps
    end_step = start_step + num_steps
    return python_profiler, start_step, end_step


def train_loop(out_dir, debugger=False, python_profiler=None, start_step=None, end_step=None):
    hook = Hook(out_dir=out_dir, save_all=True)
    if python_profiler:
        hook.python_profiler = python_profiler
    helper_native_tf2_gradtape(
        hook=hook, debugger=debugger, start_step=start_step, end_step=end_step
    )


def verify_num_trace_events(profiler_config):
    """
    This verifies the number of events when detailed profiling is enabled.
    """
    t_events = TensorboardProfilerEvents()

    # get tensorboard timeline files
    files = []

    for path in Path(os.path.join(profiler_config.config.local_path + "/framework")).rglob(
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


def verify_tensor_names(out_dir):
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


def verify_timeline_file(out_dir):
    """
    This verifies the creation of the timeline file according to file path specification.
    It reads backs the file contents to make sure it is in valid JSON format.
    """
    files = []
    for path in Path(os.path.join(out_dir + "/" + DEFAULT_PREFIX)).rglob("*.json"):
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


def verify_python_profiling(profiler_name, out_dir, num_steps):
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


@pytest.mark.skip_if_non_eager
@pytest.mark.parametrize("enable_detailed_profiling", [False, True])
@pytest.mark.parametrize("enable_python_profiling", [False, CPROFILE_NAME, PYINSTRUMENT_NAME])
@pytest.mark.parametrize("enable_debugger", [False, True])
def test_native_tf2_profiling_debugger(
    enable_detailed_profiling,
    enable_python_profiling,
    enable_debugger,
    profiler_config_path,
    out_dir,
):
    if not enable_debugger:
        if enable_detailed_profiling and not enable_python_profiling:
            profiler_config_parser = generate_profiler_config_parser(
                "DetailedProfiling", profiler_config_path, (8, 4, None, None)
            )
            train_loop(out_dir)
            verify_num_trace_events(profiler_config_parser)
            verify_timeline_file(out_dir)
        elif not enable_detailed_profiling and enable_python_profiling:
            if enable_python_profiling == CPROFILE_NAME:
                profiler_config_parser = generate_profiler_config_parser(
                    "PythonProfiling", profiler_config_path, (5, 2, CPROFILE_NAME, None)
                )
            if enable_python_profiling == PYINSTRUMENT_NAME:
                profiler_config_parser = generate_profiler_config_parser(
                    "PythonProfiling", profiler_config_path, (10, 3, PYINSTRUMENT_NAME, None)
                )
            python_profiler, start_step, end_step = initiate_python_profiling(
                profiler_config_parser
            )
            train_loop(
                out_dir, python_profiler=python_profiler, start_step=start_step, end_step=end_step
            )
            verify_python_profiling(
                enable_python_profiling, out_dir, num_steps=end_step - start_step
            )
            verify_timeline_file(out_dir)
        elif enable_detailed_profiling and enable_python_profiling:
            profiler_config_parser = generate_profiler_config_parser_all_params(
                profiler_config_path, (4, 2, enable_python_profiling, None), (8, 1, None, None)
            )
            python_profiler, start_step, end_step = initiate_python_profiling(
                profiler_config_parser
            )
            train_loop(
                out_dir, python_profiler=python_profiler, start_step=start_step, end_step=end_step
            )
            verify_python_profiling(
                enable_python_profiling, out_dir, num_steps=end_step - start_step
            )
            verify_num_trace_events(profiler_config_parser)
            verify_timeline_file(out_dir)
        else:
            pass
    else:
        if enable_detailed_profiling and not enable_python_profiling:
            profiler_config_parser = generate_profiler_config_parser(
                "DetailedProfiling", profiler_config_path, (8, 4, None, None)
            )
            train_loop(out_dir, debugger=True)
            verify_num_trace_events(profiler_config_parser)
            verify_timeline_file(out_dir)
            verify_tensor_names(out_dir)
        elif not enable_detailed_profiling and enable_python_profiling:
            if enable_python_profiling == CPROFILE_NAME:
                profiler_config_parser = generate_profiler_config_parser(
                    "PythonProfiling", profiler_config_path, (5, 2, CPROFILE_NAME, None)
                )
            if enable_python_profiling == PYINSTRUMENT_NAME:
                profiler_config_parser = generate_profiler_config_parser(
                    "PythonProfiling", profiler_config_path, (10, 3, PYINSTRUMENT_NAME, None)
                )
            python_profiler, start_step, end_step = initiate_python_profiling(
                profiler_config_parser
            )
            train_loop(
                out_dir,
                debugger=True,
                python_profiler=python_profiler,
                start_step=start_step,
                end_step=end_step,
            )
            verify_python_profiling(
                enable_python_profiling, out_dir, num_steps=end_step - start_step
            )
            verify_timeline_file(out_dir)
            verify_tensor_names(out_dir)
        elif enable_detailed_profiling and enable_python_profiling:
            profiler_config_parser = generate_profiler_config_parser_all_params(
                profiler_config_path, (4, 2, enable_python_profiling, None), (8, 1, None, None)
            )
            python_profiler, start_step, end_step = initiate_python_profiling(
                profiler_config_parser
            )
            train_loop(
                out_dir,
                debugger=True,
                python_profiler=python_profiler,
                start_step=start_step,
                end_step=end_step,
            )
            verify_python_profiling(
                enable_python_profiling, out_dir, num_steps=end_step - start_step
            )
            verify_num_trace_events(profiler_config_parser)
            verify_timeline_file(out_dir)
            verify_tensor_names(out_dir)
        else:
            pass
