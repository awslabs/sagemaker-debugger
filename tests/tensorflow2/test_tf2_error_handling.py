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
from smdebug.tensorflow import KerasHook as Hook


def _get_on_train_batch_begin_error_fn(on_train_batch_begin):
    @error_handler.catch_smdebug_errors
    def on_train_batch_begin_error(self, batch, logs=None):
        on_train_batch_begin(batch, logs=logs)
        assert False

    return on_train_batch_begin_error


def _get_wrap_model_with_input_output_saver_error_fn(
    _wrap_model_with_input_output_saver, error_message
):
    def _wrap_model_with_input_output_saver_error(self):
        def _get_layer_call_fn_error(layer):
            # assert False, 2
            old_call_fn = layer.old_call

            # @error_handler.catch_smdebug_layer_call_errors(old_call_fn=old_call_fn)
            def call(inputs, *args, **kwargs):
                assert False, 1
                layer_input = inputs
                layer_output = old_call_fn(inputs, *args, **kwargs)
                for hook in layer._hooks:
                    hook_result = hook(inputs, layer_input=layer_input, layer_output=layer_output)
                    if hook_result is not None:
                        layer_output = hook_result
                raise RuntimeError(error_message)
                return layer_output

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
def error_message():
    return (
        "If this RuntimeError causes the test to fail, the error handler failed to catch the error!"
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


def test_tf2_error_handling(out_dir, stack_trace_filepath, error_message):
    """
    This test executes a TF2 training script, enables detailed TF profiling by step, and
    verifies the number of events.
    """
    # Hook.on_train_batch_begin = _get_on_train_batch_begin_error_fn(Hook.on_train_batch_begin)
    assert error_handler.disabled is False
    Hook._wrap_model_with_input_output_saver = _get_wrap_model_with_input_output_saver_error_fn(
        Hook._wrap_model_with_input_output_saver, error_message
    )
    hook = Hook(out_dir=out_dir)
    helper_keras_fit(trial_dir=out_dir, hook=hook, eager=True, steps=["train", "eval", "predict"])
    hook.close()

    # assert error_handler.disabled is True
    with open(stack_trace_filepath) as logs:
        stack_trace_logs = logs.read()
        assert BASE_ERROR_MESSAGE in stack_trace_logs
        assert error_message in stack_trace_logs
