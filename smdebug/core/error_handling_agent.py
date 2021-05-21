# Standard Library
import functools

# First Party
from smdebug.core.logger import get_logger

# Base message logged when the error handling agent has caught an error.
BASE_ERROR_MESSAGE = (
    "SMDebug error has occurred, disabling SMDebug for the rest of the training job. Stack trace:"
)


class ErrorHandlingAgent(object):
    """
    Error handling agent to catch all errors originating from smdebug and its dependencies. This is instantiated as a
    global util object and wrapped around smdebug functions.

    Currently, the error handling agent is designed to catch all errors that could come up for the default smdebug
    configuration, or in other words, the default debugger and profiler configuration. The agent is wrapped
    around all externally facing smdebug functions (i.e. called from the ML framework).

    Only one instance of the error handling agent is allowed. The agent cannot be instantiated directly. To get an
    object of the error handling agent, `get_error_handling_agent` must be called:

    ```
    error_handling_agent = ErrorHandlingAgent.get_error_handling_agent()
    ```

    If the error handling agent catches an error, smdebug functionality is disabled for the rest of training.

    TODO: Wrap the error handling agent around all smdebug functions called from all frameworks.
    """

    _error_handling_agent = None

    class _ErrorHandlingAgent(object):
        def __init__(self):
            self.disable_smdebug = False
            self.logger = get_logger()
            self.hook = None

        def set_hook(self, hook):
            """
            Set the hook to be used by the error handling agent. The hook is used to determine whether the ongoing
            training is using the default smdebug configuration or a custom smdebug configuration.

            This is meant to be called in the constructor of the relevant hook used for training. If an error occurs
            before this function is called, the error handling agent will catch the error.
            """
            self.hook = hook

        def catch_smdebug_errors(self, default_return_val=None):
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

            Currently, the error handling agent will only catch errors if the default smdebug configuration is being
            used. Otherwise, the error will be raised normally. When an error is caught, the stack trace of the error
            will still be logged for tracking purposes.

            Examples:

            ```
            error_handling_agent = ErrorHandlingAgent.get_error_handling_agent()
            ...
            @error_handling_agent.catch_smdebug_errors()
            def foo(*args, **kwargs):
                ...
                return

            @error_handling_agent.catch_smdebug_errors(default_return_val=False)
            def bar(*args, **kwargs):
                ...
                return True

            def foobar(*args, **kwargs):
                default_func = lambda *args, **kwargs: {"args": args, "kwargs": kwargs}

                @error_handling_agent.catch_smdebug_errors(default_return_val=default_func)
                def baz()
                    ...
                return baz
            ```
            """

            def wrapper(func):
                @functools.wraps(func)
                def error_handling_agent(*args, **kwargs):
                    # Return immediately if smdebug is disabled.
                    if self.disable_smdebug:
                        # If `default_return_val` is a function, call it with the inputs and return the output.
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

                            # If `default_return_val` is a function, call it with the inputs and return the output.
                            if callable(default_return_val):
                                return default_return_val(*args, **kwargs)
                            return default_return_val
                        else:
                            raise e  # Raise the error normally

                return error_handling_agent

            return wrapper

    @classmethod
    def get_error_handling_agent(cls):
        if cls._error_handling_agent is None:
            cls._error_handling_agent = cls._ErrorHandlingAgent()

        return cls._error_handling_agent
