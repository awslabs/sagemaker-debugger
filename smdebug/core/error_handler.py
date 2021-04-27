# Standard Library
import functools

# First Party
from smdebug.core.logger import get_logger

# Base message logged when the error handler has caught an error.
BASE_ERROR_MESSAGE = (
    "SMDebug error has occurred, disabling SMDebug for the rest of the training job. Stack trace:"
)


class SMDebugErrorHandler(object):
    """
    Error handler to catch all errors originating from smdebug and its dependencies. This is instantiated as a
    global util object and wrapped around smdebug functions.

    Currently, the error handler is designed to catch all errors that could come up for the default smdebug
    configuration, or in other words, the default debugger and profiler configuration. The error handler is wrapped
    around all externally facing smdebug functions (i.e. called from the ML framework).

    Only one instance of the error handler is allowed. The error handler cannot be instantiated directly. To get the
    error handler, `get_error_handler` must be called:

    ```
    error_handler = SMDebugErrorHandler.get_error_handler()
    ```

    If an error handler catches an error, smdebug functionality is disabled for the rest of training.

    At this time, the error handler is only implemented for TF2.

    TODO: Wrap the error handler around all smdebug functions called from all frameworks.
    """

    _error_handler = None

    class _SMDebugErrorHandler(object):
        def __init__(self):
            self.disable_smdebug = False
            self.logger = get_logger()
            self.hook = None

        def set_hook(self, hook):
            """
            Set the hook to be used by the error handler. The hook is used to determine whether the ongoing training is
            using the default smdebug configuration or a custom smdebug configuration.

            This is meant to be called in the constructor of the relevant hook used for training. If an error occurs before
            this function is called, the error handler will catch the error.
            """
            self.hook = hook

        def catch_smdebug_errors(self, return_type=None, **handler_kwargs):
            """
            This function is designed to be wrapped around all smdebug functions that are called externally, so that any
            errors arising from the wrapped functions or the resulting smdebug or third party functions called are caught
            here.

            The return type of the function being wrapped must be specified in `return_type` if it isn't valid for the
            wrapped function to return `None`. Based on the return type specified, a default return value is determined. If
            the error handler has caught an error or smdebug has already been disabled, the default return value is
            returned.

            Currently, the error handler will only catch errors if the default smdebug configuration is being used.
            Otherwise, the error will be raised normally. When an error is caught, the stack trace of the error will still
            be logged for tracking purposes.

            Examples:

            ```
            error_handler = SMDebugErrorHandler.get_error_handler()
            ...
            @error_handler.catch_smdebug_errors()
            def foo():
                ...
                return

            @error_handler.catch_smdebug_errors(return_value=bool)
            def bar():
                ...
                return True
            ```
            """

            def wrapper(func):
                @functools.wraps(func)
                def error_handler(*args, **kwargs):
                    # Determine default return value based on the return type.
                    if return_type == bool:
                        return_val = False
                    elif return_type == "layer_call":
                        return_val = handler_kwargs["old_call_fn"](*args, **kwargs)
                    else:
                        return_val = None

                    # Return immediately if smdebug is disabled.
                    if self.disable_smdebug:
                        return return_val

                    try:
                        # Attempt calling the smdebug function and returning the output
                        return func(*args, **kwargs)
                    except Exception as e:
                        # If an smdebug error occurred with the default configuration or it occurred before the
                        # configuration can even be determined (i.e. the constructor), catch the error and log it.
                        if self.hook is None or self.hook.has_default_configuration():
                            if return_type == "tape":
                                self.hook._unwrap_tape()
                                return handler_kwargs["function"](*args, **kwargs)
                            self.logger.error(BASE_ERROR_MESSAGE)
                            self.logger.exception(e)  # Log stack trace.
                            self.disable_smdebug = True  # Disable smdebug
                            return return_val
                        else:
                            raise e  # Raise the error normally

                return error_handler

            return wrapper

    @classmethod
    def get_error_handler(cls):
        if cls._error_handler is None:
            cls._error_handler = cls._SMDebugErrorHandler()

        return cls._error_handler
