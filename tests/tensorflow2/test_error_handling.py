# Standard Library
import functools
import logging
import os

# Third Party
import pytest
from tests.tensorflow2.utils import helper_gradtape_tf, helper_keras_fit

# First Party
from smdebug.core.error_handler import BASE_ERROR_MESSAGE
from smdebug.core.logger import DuplicateLogFilter, get_logger
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
def gradient_tape_callback_error_message():
    return "If this GradientTape callback error causes the test to fail, the error handler failed to catch the error!"


@pytest.fixture
def custom_configuration_error_message():
    return "This error should have been thrown and not caught by the error handler!"


@pytest.fixture
def hook_class_with_keras_callback_error(out_dir, keras_callback_error_message):
    class HookWithBadKerasCallback(Hook):
        def __init__(self, keras_error_message=keras_callback_error_message, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.keras_callback_error_message = keras_error_message

        @error_handler.catch_smdebug_errors()
        def on_train_batch_begin(self, batch, logs=None):
            raise RuntimeError(self.keras_callback_error_message)

        @classmethod
        def create_from_json_file(cls, json_file_path=None):
            return HookWithBadKerasCallback(out_dir=out_dir)

    return HookWithBadKerasCallback


@pytest.fixture
def hook_class_with_layer_callback_error(out_dir, layer_callback_error_message):
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

        @classmethod
        def create_from_json_file(cls, json_file_path=None):
            return HookWithBadLayerCallback(out_dir=out_dir)

    return HookWithBadLayerCallback


@pytest.fixture
def hook_class_with_gradient_tape_callback_error(out_dir, gradient_tape_callback_error_message):
    class HookWithBadGradientTapeCallbacks(Hook):
        def __init__(
            self, gradient_tape_message=gradient_tape_callback_error_message, *args, **kwargs
        ):
            super().__init__(*args, **kwargs)
            self.gradient_tape_callback_error_message = gradient_tape_message

        def _wrap_push_tape(self, function):
            @functools.wraps(function)
            @error_handler.catch_smdebug_errors(return_type="tape", function=function)
            def run(*args, **kwargs):
                raise RuntimeError(self.gradient_tape_callback_error_message)

            return run

        def _wrap_tape_gradient(self, function):
            @functools.wraps(function)
            @error_handler.catch_smdebug_errors(return_type="tape", function=function)
            def run(*args, **kwargs):
                raise RuntimeError(self.gradient_tape_callback_error_message)

            return run

        def _wrap_pop_tape(self, function):
            @functools.wraps(function)
            @error_handler.catch_smdebug_errors(return_type="tape", function=function)
            def run(*args, **kwargs):
                raise RuntimeError(self.gradient_tape_callback_error_message)

            return run

        @classmethod
        def create_from_json_file(cls, json_file_path=None):
            return HookWithBadGradientTapeCallbacks(out_dir=out_dir)

    return HookWithBadGradientTapeCallbacks


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

        @classmethod
        def create_from_json_file(cls, json_file_path=None):
            return HookWithBadKerasAndLayerCallback(out_dir=out_dir)

    return HookWithBadKerasAndLayerCallback


@pytest.fixture
def hook_class_with_keras_callback_error_and_custom_debugger_configuration(
    out_dir, custom_configuration_error_message, hook_class_with_keras_callback_error
):
    class HookWithBadKerasCallbackAndCustomDebuggerConfiguration(
        hook_class_with_keras_callback_error
    ):
        def __init__(self, *args, **kwargs):
            super().__init__(
                keras_callback_error_message=custom_configuration_error_message, *args, **kwargs
            )

        @classmethod
        def create_from_json_file(cls, json_file_path=None):
            return HookWithBadKerasCallbackAndCustomDebuggerConfiguration(
                out_dir=out_dir, include_collections=[CollectionKeys.ALL]
            )

    return HookWithBadKerasCallbackAndCustomDebuggerConfiguration


@pytest.fixture
def profiler_config_path(config_folder):
    return os.path.join(config_folder, "test_tf2_profiler_config_parser_by_time.json")


@pytest.fixture
def hook_class_with_keras_callback_error_and_custom_profiler_configuration(
    out_dir,
    custom_configuration_error_message,
    monkeypatch,
    profiler_config_path,
    hook_class_with_keras_callback_error,
):
    class HookWithBadKerasCallbackAndCustomDebuggerConfiguration(
        hook_class_with_keras_callback_error
    ):
        def __init__(self, *args, **kwargs):
            super().__init__(
                keras_callback_error_message=custom_configuration_error_message, *args, **kwargs
            )

        @classmethod
        def create_from_json_file(cls, json_file_path=None):
            monkeypatch.setenv("SMPROFILER_CONFIG_PATH", profiler_config_path)
            return HookWithBadKerasCallbackAndCustomDebuggerConfiguration(out_dir=out_dir)

    return HookWithBadKerasCallbackAndCustomDebuggerConfiguration


@pytest.fixture(autouse=True)
def set_up_logging_and_error_handler(out_dir, stack_trace_filepath):
    """
    Setup up each test to:
        - Add a logging handler to write all logs to a file (which will be used to verify caught errors in the tests)
        - Remove the duplicate logging filter
        - Reset the error handler after the test so it that it is reenabled.
    """
    old_create_from_json = Hook.create_from_json_file
    logger = get_logger()
    os.makedirs(out_dir)
    file_handler = logging.FileHandler(filename=stack_trace_filepath)
    logger.addHandler(file_handler)
    duplicate_log_filter = None
    for log_filter in logger.filters:
        if isinstance(log_filter, DuplicateLogFilter):
            duplicate_log_filter = log_filter
            break
    logger.removeFilter(duplicate_log_filter)
    yield
    logger.removeHandler(file_handler)
    logger.addFilter(duplicate_log_filter)
    error_handler.disable_smdebug = False
    Hook.create_from_json_file = old_create_from_json


@pytest.mark.parametrize(
    "hook_type",
    [
        "keras_callback_error",
        "layer_callback_error",
        "gradient_tape_callback_error",
        "keras_and_layer_callback_error",
    ],
)
def test_tf2_callback_error_handling(
    hook_type,
    hook_class_with_keras_callback_error,
    hook_class_with_layer_callback_error,
    hook_class_with_gradient_tape_callback_error,
    hook_class_with_keras_and_layer_callback_error,
    out_dir,
    stack_trace_filepath,
    keras_callback_error_message,
    layer_callback_error_message,
    gradient_tape_callback_error_message,
):
    """
    Test that an error thrown by an smdebug callback is caught and logged correctly by the error handler. This test
    has three test cases:

    keras_callback_error: The Keras hook's `on_train_batch_begin` is overridden to always fail. The error handler should
    catch this error and log it once, and then disable smdebug for the rest of training.

    layer_callback_error: The Keras hook's layer callback `call` is overridden to always fail. The error handler should
    catch this error and log it once, and then disable smdebug for the rest of training.

    gradient_tape_error: The Keras hook's GradientTape callbacks `_wrap_tape_*` are overriden to always fail. The error
    handler should catch this error and log it once, and then disable smdebug for the rest of training.

    keras_and_layer_callback_error: Both the keras and layer callback functions are overridden to always fail. However,
    the `on_train_batch_begin` function is called first, so the error handler should only catch that error. Then
    because smdebug is disabled, the error raised by `call` should not be caught because the function shouldn't be even
    be called in the first place.

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
    elif hook_type == "gradient_tape_callback_error":
        hook_class = hook_class_with_gradient_tape_callback_error
        error_message = gradient_tape_callback_error_message
    else:
        hook_class = hook_class_with_keras_and_layer_callback_error
        error_message = (
            keras_callback_error_message
        )  # only on_train_batch_begin should error and get caught

    Hook.create_from_json_file = hook_class.create_from_json_file

    # hook = hook_class(out_dir=out_dir)
    # hook._prepare_collections_for_tf2()
    # assert (
    #     hook.has_default_configuration()
    # )  # error handler should only catch errors for default smdebug configuration

    helper = helper_gradtape_tf if hook_type == "gradient_tape_callback_error" else helper_keras_fit
    helper(out_dir)
    # hook.close()

    assert error_handler.disable_smdebug is True
    with open(stack_trace_filepath) as logs:
        stack_trace_logs = logs.read()

        # check that one error was caught
        assert stack_trace_logs.count(BASE_ERROR_MESSAGE) == 1

        # check that the right error was caught (printed twice for each time caught)
        assert stack_trace_logs.count(error_message) == 2

        # `call` should not have errored if `on_train_batch_begin` already errored.
        if hook_type == "keras_and_layer_callback_error":
            assert layer_callback_error_message not in stack_trace_logs


@pytest.mark.parametrize("custom_configuration", ["debugger", "profiler"])
def test_non_default_smdebug_configuration(
    custom_configuration,
    monkeypatch,
    profiler_config_path,
    out_dir,
    hook_class_with_keras_callback_error_and_custom_debugger_configuration,
    hook_class_with_keras_callback_error_and_custom_profiler_configuration,
    custom_configuration_error_message,
    stack_trace_filepath,
):
    """
    Test that the error handler does not catch errors when a custom smdebug configuration of smdebug is used.

    This hook needs to be initialized during its corresponding test, because the error handler is configured to a hook
    during the hook initialization.
    """
    if custom_configuration == "debugger":
        hook_class = hook_class_with_keras_callback_error_and_custom_debugger_configuration
    else:
        hook_class = hook_class_with_keras_callback_error_and_custom_profiler_configuration

    Hook.create_from_json_file = hook_class.create_from_json_file

    # Verify the correct error gets thrown and doesnt get caught.
    with pytest.raises(RuntimeError, match=custom_configuration_error_message):
        helper_keras_fit(out_dir)
    assert error_handler.disable_smdebug is False
    with open(stack_trace_filepath) as logs:
        stack_trace_logs = logs.read()
        assert stack_trace_logs.count(BASE_ERROR_MESSAGE) == 0
        assert stack_trace_logs.count(custom_configuration_error_message) == 0
