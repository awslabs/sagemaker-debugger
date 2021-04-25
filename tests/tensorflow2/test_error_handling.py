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
from smdebug.tensorflow.collection import CollectionKeys


@pytest.fixture
def stack_trace_filepath(out_dir):
    return f"{out_dir}/tmp.log"


@pytest.fixture
def keras_callback_error_message():
    return "If this Keras callback error causes the test to fail, the error handler failed to catch the error!"


@pytest.fixture
def layer_callback_error_message():
    return "If this layer callback error causes the test to fail, the error handler failed to catch the error!"


@pytest.fixture
def custom_configuration_error_message():
    return "This error should have been thrown and not caught by the error handler!"


@pytest.fixture
def hook_class_with_keras_callback_error(keras_callback_error_message):
    class HookWithBadKerasCallback(Hook):
        def __init__(self, keras_error_message=keras_callback_error_message, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.keras_callback_error_message = keras_error_message

        @error_handler.catch_smdebug_errors()
        def on_train_batch_begin(self, batch, logs=None):
            raise RuntimeError(self.keras_callback_error_message)

    return HookWithBadKerasCallback


@pytest.fixture
def hook_class_with_layer_callback_error(layer_callback_error_message):
    class HookWithBadLayerCallback(Hook):
        def __init__(self, layer_error_message=layer_callback_error_message, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.layer_callback_error_message = layer_error_message

        def _wrap_model_with_input_output_saver(self):
            def _get_layer_call_fn_error(layer):
                old_call_fn = layer.call

                @error_handler.catch_smdebug_errors(
                    return_type="layer_call", old_call_fn=old_call_fn
                )
                def call(inputs, *args, **kwargs):
                    raise RuntimeError(self.layer_callback_error_message)

                return call

            for layer in self.model.layers:
                layer.call = _get_layer_call_fn_error(layer)

    return HookWithBadLayerCallback


@pytest.fixture
def hook_class_with_keras_and_layer_callback_error(
    out_dir,
    hook_class_with_keras_callback_error,
    hook_class_with_layer_callback_error,
    keras_callback_error_message,
    layer_callback_error_message,
):
    class HookWithBadKerasAndLayerCallback(
        hook_class_with_keras_callback_error, hook_class_with_layer_callback_error
    ):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    return HookWithBadKerasAndLayerCallback


@pytest.fixture
def profiler_config_path(config_folder):
    return os.path.join(config_folder, "test_tf2_profiler_config_parser_by_time.json")


@pytest.fixture(autouse=True)
def set_up(out_dir, stack_trace_filepath):
    logger = get_logger()
    os.makedirs(out_dir)
    file_handler = logging.FileHandler(filename=f"{out_dir}/tmp.log")
    logger.addHandler(file_handler)
    yield
    error_handler.disable_smdebug = False


@pytest.mark.parametrize("hook_type", ["keras_callback_error", "layer_callback_error", "both"])
def test_tf2_keras_callback_error_handling(
    hook_type,
    hook_class_with_keras_callback_error,
    hook_class_with_layer_callback_error,
    hook_class_with_keras_and_layer_callback_error,
    out_dir,
    stack_trace_filepath,
    keras_callback_error_message,
    layer_callback_error_message,
):
    """
    Test that an error thrown by an smdebug callback is caught and logged correctly by the error handler. This test
    has three test cases:

    keras_callback_error: The Keras hook's `on_train_batch_begin` is overridden to always fail. The error handler should
    catch this error and log it once, and then disable smdebug for the rest of training.

    layer_callback_error: The Keras hook's layer callback `call` is overridden to always fail. The error handler should
    catch this error and log it once, and then disable smdebug for the rest of training.

    both: Both of the above hook functions are overridden to always fail. However, the `on_train_batch_begin` function
    is called first, so the error handler should only catch that error. Then because smdebug is disabled, the error
    raised by `call` should not be caught because the function shouldn't be even be called in the first place.

    Each hook needs to be initialized during its corresponding test, because the error handler is configured to a hook
    during the hook initialization.
    """
    assert error_handler.disable_smdebug is False

    if hook_type == "keras_callback_error":
        hook_class = hook_class_with_keras_callback_error
        error_message = keras_callback_error_message
    elif hook_type == "layer_callback_error":
        hook_class = hook_class_with_layer_callback_error
        error_message = layer_callback_error_message
    else:
        hook_class = hook_class_with_keras_and_layer_callback_error
        error_message = (
            keras_callback_error_message
        )  # only on_train_batch_begin should error and get caught

    hook = hook_class(out_dir=out_dir)
    hook._prepare_collections_for_tf2()
    assert (
        hook.has_default_configuration()
    )  # error handler should only catch errors for default smdebug configuration

    helper_keras_fit(trial_dir=out_dir, hook=hook, eager=True, steps=["train", "eval", "predict"])
    hook.close()

    assert error_handler.disable_smdebug is True
    with open(stack_trace_filepath) as logs:
        stack_trace_logs = logs.read()

        # check that one error was caught
        assert stack_trace_logs.count(BASE_ERROR_MESSAGE) == 1

        # check that the right error was caught (printed twice for each time caught)
        assert stack_trace_logs.count(error_message) == 2

        # `call` should not have errored if `on_train_batch_begin` already errored.
        if hook_type == "both":
            assert layer_callback_error_message not in stack_trace_logs


@pytest.mark.parametrize("custom_configuration", ["debugger", "profiler"])
def test_non_default_smdebug_configuration(
    custom_configuration,
    monkeypatch,
    profiler_config_path,
    out_dir,
    hook_class_with_keras_callback_error,
    custom_configuration_error_message,
    stack_trace_filepath,
):
    """
    Test that the error handler does not catch errors when a custom smdebug configuration of smdebug is used.

    This hook needs to be initialized during its corresponding test, because the error handler is configured to a hook
    during the hook initialization.
    """
    if custom_configuration == "debugger":
        hook = hook_class_with_keras_callback_error(
            keras_error_message=custom_configuration_error_message,
            out_dir=out_dir,
            include_collections=[CollectionKeys.ALL],
        )
    else:
        monkeypatch.setenv("SMPROFILER_CONFIG_PATH", profiler_config_path)
        hook = hook_class_with_keras_callback_error(
            keras_error_message=custom_configuration_error_message, out_dir=out_dir
        )

    # Verify the correct error gets thrown and doesnt get caught.
    with pytest.raises(RuntimeError, match=custom_configuration_error_message):
        helper_keras_fit(
            trial_dir=out_dir, hook=hook, eager=True, steps=["train", "eval", "predict"]
        )
    assert error_handler.disable_smdebug is False
    with open(stack_trace_filepath) as logs:
        stack_trace_logs = logs.read()
        assert stack_trace_logs.count(BASE_ERROR_MESSAGE) == 0
        assert stack_trace_logs.count(custom_configuration_error_message) == 0
