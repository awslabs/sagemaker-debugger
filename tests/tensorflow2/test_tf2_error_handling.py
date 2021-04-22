# Standard Library
import logging
import os
from pathlib import Path

# Third Party
import pytest
from tests.tensorflow2.test_keras import helper_keras_fit

# First Party
from smdebug.core.error_handler import BASE_ERROR_MESSAGE
from smdebug.core.logger import get_logger
from smdebug.core.utils import error_handler


def _get_on_train_batch_begin_error_fn(error_message):
    @error_handler.catch_smdebug_errors()
    def on_train_batch_begin_error(self, batch, logs=None):
        raise RuntimeError(error_message)

    return on_train_batch_begin_error


def _get_wrap_model_with_input_output_saver_error_fn(
    _wrap_model_with_input_output_saver, error_message
):
    def _wrap_model_with_input_output_saver_error(self):
        def _get_layer_call_fn_error(layer):
            # assert False, 2
            old_call_fn = layer.old_call

            @error_handler.catch_smdebug_layer_call_errors(old_call_fn=old_call_fn)
            def call(inputs, *args, **kwargs):
                raise ValueError(error_message)

            return call

        _wrap_model_with_input_output_saver(self)

        for layer in self.model.layers:
            layer.call = _get_layer_call_fn_error(layer)

    return _wrap_model_with_input_output_saver_error


@pytest.fixture
def stack_trace_filepath(out_dir):
    return f"{out_dir}/tmp.log"


@pytest.fixture(autouse=True)
def set_up(out_dir, stack_trace_filepath):
    logger = get_logger()
    os.makedirs(out_dir)
    Path(stack_trace_filepath).touch()
    file_handler = logging.FileHandler(filename=stack_trace_filepath)
    logger.addHandler(file_handler)
    error_handler.reset()


@pytest.fixture
def runtime_error_message():
    return (
        "If this RuntimeError causes the test to fail, the error handler failed to catch the error!"
    )


@pytest.fixture
def value_error_message():
    return (
        "If this ValueError causes the test to fail, the error handler failed to catch the error!"
    )


@pytest.fixture
def dummy_clean_function():
    @error_handler.catch_smdebug_errors()
    def do_nothing():
        pass

    return do_nothing


@pytest.fixture(autouse=True)
def set_up(out_dir):
    logger = get_logger()
    os.makedirs(out_dir)
    file_handler = logging.FileHandler(filename=f"{out_dir}/tmp.log")
    logger.addHandler(file_handler)
    error_handler.reset()


@pytest.fixture
def hook_with_keras_callback_error(out_dir, runtime_error_message):
    from smdebug.tensorflow import KerasHook as Hook

    old_on_train_batch_begin = Hook.on_train_batch_begin
    Hook.on_train_batch_begin = _get_on_train_batch_begin_error_fn(runtime_error_message)
    yield Hook(out_dir=out_dir)
    Hook.on_train_batch_begin = old_on_train_batch_begin


@pytest.fixture
def hook_with_layer_callback_error(out_dir, value_error_message):
    from smdebug.tensorflow import KerasHook as Hook

    old_wrap_model_with_input_output_saver = Hook._wrap_model_with_input_output_saver
    Hook._wrap_model_with_input_output_saver = _get_wrap_model_with_input_output_saver_error_fn(
        Hook._wrap_model_with_input_output_saver, value_error_message
    )
    yield Hook(out_dir=out_dir)
    Hook._wrap_model_with_input_output_saver = old_wrap_model_with_input_output_saver


def test_tf2_keras_callback_error_handling(
    hook_with_keras_callback_error, out_dir, stack_trace_filepath, runtime_error_message
):
    """
    This test executes a TF2 training script, enables detailed TF profiling by step, and
    verifies the number of events.
    """
    # Hook.on_train_batch_begin = _get_on_train_batch_begin_error_fn(Hook.on_train_batch_begin)
    assert error_handler.disabled is False

    helper_keras_fit(
        trial_dir=out_dir,
        hook=hook_with_keras_callback_error,
        eager=True,
        steps=["train", "eval", "predict"],
    )
    hook_with_keras_callback_error.close()

    # assert error_handler.disabled is True
    with open(stack_trace_filepath) as logs:
        stack_trace_logs = logs.read()
        assert BASE_ERROR_MESSAGE in stack_trace_logs
        assert runtime_error_message in stack_trace_logs


def test_tf2_layer_callback_error_handling(
    hook_with_layer_callback_error, out_dir, stack_trace_filepath, value_error_message
):
    """
    This test executes a TF2 training script, enables detailed TF profiling by step, and
    verifies the number of events.
    """
    # Hook.on_train_batch_begin = _get_on_train_batch_begin_error_fn(Hook.on_train_batch_begin)
    assert error_handler.disabled is False

    helper_keras_fit(
        trial_dir=out_dir,
        hook=hook_with_layer_callback_error,
        eager=True,
        steps=["train", "eval", "predict"],
    )
    hook_with_layer_callback_error.close()

    # assert error_handler.disabled is True
    with open(stack_trace_filepath) as logs:
        stack_trace_logs = logs.read()
        assert BASE_ERROR_MESSAGE in stack_trace_logs
        assert value_error_message in stack_trace_logs
