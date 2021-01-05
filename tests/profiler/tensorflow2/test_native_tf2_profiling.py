# Standard Library
import os
from pathlib import Path

# Third Party
import tensorflow as tf
import pytest
# from tests.tensorflow2.test_keras import helper_keras_fit

# First Party
import smdebug.tensorflow as smd
from smdebug.core.collection import CollectionKeys
from smdebug.profiler.profiler_config_parser import ProfilerConfigParser
from smdebug.profiler.profiler_constants import TENSORBOARDTIMELINE_SUFFIX
from smdebug.profiler.tf_profiler_parser import TensorboardProfilerEvents
from smdebug.tensorflow import KerasHook as Hook

@pytest.fixture()
def tf2_profiler_config_parser_by_step(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "test_tf2_profiler_config_parser_by_step.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser()


@pytest.fixture()
def tf2_profiler_config_parser_by_time(config_folder, monkeypatch):
    config_path = os.path.join(config_folder, "test_tf2_profiler_config_parser_by_time.json")
    monkeypatch.setenv("SMPROFILER_CONFIG_PATH", config_path)
    return ProfilerConfigParser()


def create_hook(trial_dir):
    hook = smd.KerasHook(trial_dir, save_all=True)
    return hook


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

def helper_native_tf2():


def test_gradtape_tf_function(out_dir):
    def get_grads(images, labels):
        # with tf.GradientTape() as tape:
        return model(images, training=True)

    @tf.function
    def train_step(images, labels):
        return tf.reduce_mean(get_grads(images, labels))

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), _ = mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_train[..., tf.newaxis] / 255, tf.float32), tf.cast(y_train, tf.int64))
    )
    dataset = dataset.shuffle(1000).batch(64)
    model = create_model()
    hook = create_hook(out_dir)
    opt = tf.keras.optimizers.Adam()
    hook.wrap_optimizer(opt)

    n_epochs = 1
    for epoch in range(n_epochs):
        for data, labels in dataset:
            dataset_labels = labels
            labels = tf.one_hot(labels, depth=10)
            hook.start_profiling_start_train_batch()
            with hook.wrap_tape(tf.GradientTape()) as tape:
                logits = train_step(data, labels)
            grads = tape.gradient(logits, model.variables)
            opt.apply_gradients(zip(grads, model.variables))
            # hook.save_tensor("inputs", data, CollectionKeys.INPUTS)
            # hook.save_tensor("logits", logits, CollectionKeys.OUTPUTS)
            # hook.save_tensor("labels", labels, CollectionKeys.OUTPUTS)
            hook.start_profiling_end_train_batch()
    hook.stop_profiling_end_of_training()

    model.save(out_dir, save_format="tf")


    trial = smd.create_trial(out_dir)
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


def test_native_tf2_profiler_by_step(set_up_resource_config, tf2_profiler_config_parser_by_step, out_dir):
    """
    This test executes a TF2 native training script, enables detailed TF profiling by step, and
    verifies the number of events.
    """
    assert tf2_profiler_config_parser_by_step.profiling_enabled

    hook = Hook(out_dir=out_dir)
    helper_native_tf2(trial_dir=out_dir, hook=hook, eager=True, steps=["train", "eval", "predict"])
    hook.close()

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