# Standard Library
import logging
import os

# Third Party
import pytest
from tests.tensorflow2.test_keras import helper_keras_fit

# First Party
from smdebug.core.error_handler import BASE_ERROR_MESSAGE
from smdebug.core.logger import get_logger
from smdebug.core.utils import error_handler
from smdebug.tensorflow import KerasHook as Hook


@pytest.fixture
def stack_trace_filepath(out_dir):
    return f"{out_dir}/tmp.log"


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
def hook_with_keras_callback_error(out_dir, runtime_error_message):
    class HookWithBadKerasCallback(Hook):
        @error_handler.catch_smdebug_errors()
        def on_train_batch_begin(self, batch, logs=None):
            raise RuntimeError(runtime_error_message)

    return HookWithBadKerasCallback(out_dir=out_dir)


@pytest.fixture
def hook_with_layer_callback_error(out_dir, value_error_message):
    class HookWithBadLayerCallback(Hook):
        def _wrap_model_with_input_output_saver(self):
            def _get_layer_call_fn_error(layer):
                old_call_fn = layer.old_call

                @error_handler.catch_smdebug_errors(
                    return_type="layer_call", old_call_fn=old_call_fn
                )
                def call(inputs, *args, **kwargs):
                    raise ValueError(value_error_message)

                return call

            for layer in self.model.layers:
                layer.call = _get_layer_call_fn_error(layer)

    return HookWithBadLayerCallback(out_dir=out_dir)


@pytest.fixture(autouse=True)
def set_up(out_dir):
    logger = get_logger()
    os.makedirs(out_dir)
    file_handler = logging.FileHandler(filename=f"{out_dir}/tmp.log")
    logger.addHandler(file_handler)
    yield
    error_handler.disable_smdebug = False


def test_tf2_keras_callback_error_handling(
    hook_with_keras_callback_error, out_dir, stack_trace_filepath, runtime_error_message
):
    """
    Test that an error thrown by an smdebug Keras callback is caught and logged correctly by the error handler.
    """
    assert error_handler.disable_smdebug is False

    print(hook_with_keras_callback_error._collections_to_save)
    hook_with_keras_callback_error._prepare_collections_for_tf2()
    print(hook_with_keras_callback_error._collections_to_save)
    print(
        hook_with_keras_callback_error.has_default_hook_configuration(),
        hook_with_keras_callback_error.has_default_profiler_configuration(),
    )
    assert hook_with_keras_callback_error.has_default_configuration()

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
    Test that an error thrown by an smdebug layer callback is caught and logged correctly by the error handler.
    """
    assert error_handler.disable_smdebug is False

    hook_with_layer_callback_error._prepare_collections_for_tf2()
    assert hook_with_layer_callback_error.has_default_configuration()

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
