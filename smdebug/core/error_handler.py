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

        def catch_smdebug_errors(self, default_return_val=None, return_type=None):
            """
            This function is designed to be wrapped around all smdebug functions that are called externally, so that any
            errors arising from the wrapped functions or the resulting smdebug or third party functions called are
            caught here.

            When an error is caught during the execution of the wrapped function, smdebug is disabled for
            the rest of training. A default return value is returned when an error is caught or a wrapped function is
            called when smdebug is already disabled.

            The default return value of the wraooed function (in the event of an error) must be specified in
            `default_return_value` if it isn't valid for the wrapped function to return `None`.

            If the default return value can only be determined at runtime (i.e. layer or tape callback), a function
            can be provided in `default_return_value` and the default return value will be determined dynamically by
            calling that function with the inputs provided to the wrapped function.

            Currently, the error handler will only catch errors if the default smdebug configuration is being used.
            Otherwise, the error will be raised normally. When an error is caught, the stack trace of the error will still
            be logged for tracking purposes.

            Examples:

            ```
            error_handler = SMDebugErrorHandler.get_error_handler()
            ...
            @error_handler.catch_smdebug_errors()
            def foo(*args, **kwargs):
                ...
                return

            @error_handler.catch_smdebug_errors(default_return_val=False)
            def bar(*args, **kwargs):
                ...
                return True

            def foobar(*args, **kwargs):
                default_func = lambda *args, **kwargs: {"args": args, "kwargs": kwargs}

                @error_handler.catch_smdebug_errors(default_return_val=default_func)
                def baz()
                    ...
                return baz
            ```
            """

            def wrapper(func):
                @functools.wraps(func)
                def error_handler(*args, **kwargs):
                    # Return immediately if smdebug is disabled.
                    if self.disable_smdebug:
                        if callable(default_return_val):
                            return default_return_val(*args, **kwargs)
                        return default_return_val

                    try:
                        # Attempt calling the smdebug function and returning the output
                        return func(*args, **kwargs)
                    except Exception as e:
                        # If an smdebug error occurred with the default configuration or it occurred before the
                        # configuration can even be determined (i.e. the constructor), catch the error and log it.

                        if self.hook is None or self.hook.has_default_configuration():
                            self.logger.error(BASE_ERROR_MESSAGE)
                            self.logger.exception(e)  # Log stack trace.
                            self.disable_smdebug = True  # Disable smdebug

                            if callable(default_return_val):
                                return default_return_val(*args, **kwargs)
                            return default_return_val
                        else:
                            raise e  # Raise the error normally

                return error_handler

            return wrapper

    @classmethod
    def get_error_handler(cls):
        if cls._error_handler is None:
            cls._error_handler = cls._SMDebugErrorHandler()

        return cls._error_handler
