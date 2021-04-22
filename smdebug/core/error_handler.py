# Standard Library
import functools

# First Party
from smdebug.core.logger import get_logger

BASE_ERROR_MESSAGE = (
    "SMDebug error has occurred, disabling SMDebug for the rest of the training job. Stack trace:"
)


class ErrorHandler(object):
    def __init__(self):
        self.disabled = False
        self.logger = get_logger()
        self.hook = None

    def set_hook(self, hook):
        self.hook = hook

    def reset(self):
        self.disabled = False

    def catch_smdebug_errors(self, return_type=None, *args, **kwargs):
        def wrapper(func, *wrapper_args, **wrapper_kwargs):
            @functools.wraps(func)
            def error_handler(*handler_args, **handler_kwargs):
                a = wrapper_args + handler_args
                kw = {**wrapper_kwargs, **handler_kwargs}

                if return_type == bool:
                    return_val = False
                else:
                    return_val = None

                # Return immediately if smdebug is disabled.
                if self.disabled:
                    return return_val

                try:
                    return func(*a, **kw)
                except Exception as e:
                    # If an smdebug error occurred with the default configuration or it occurred before the configuration
                    # can even be determined (i.e. the constructor), catch the error and log it.
                    if self.hook is None or self.hook.has_default_configuration():
                        self.logger.error(BASE_ERROR_MESSAGE)
                        self.logger.exception(e)  # Log stack trace.
                        self.disabled = True  # Disable smdebug
                        return return_val
                    else:
                        raise e

            return error_handler

        return wrapper

    def catch_smdebug_layer_call_errors(self, *args, **kwargs):
        def wrapper(func):
            @functools.wraps(func)
            def error_handler(*a, **kw):
                return_val = kwargs["old_call_fn"](*a, **kw)

                # Return immediately if smdebug is disabled.
                if self.disabled:
                    return return_val

                try:
                    return func(*a, **kw)
                except Exception as e:
                    # If an smdebug error occurred with the default configuration or it occurred before the configuration
                    # can even be determined (i.e. the constructor), catch the error and log it.
                    if self.hook is None or self.hook.has_default_configuration():
                        self.logger.error(BASE_ERROR_MESSAGE)
                        self.logger.exception(e)  # Log stack trace.
                        self.disabled = True  # Disable smdebug
                        return return_val
                    else:
                        raise e

            return error_handler

        return wrapper

    def catch_smdebug_constructor_errors(self, func, return_type=None, *args, **kwargs):
        @functools.wraps(func)
        def error_handler(*a, **kw):
            # Return immediately if smdebug is disabled.
            if self.disabled:
                return

            try:
                func(*a, **kw)
            except Exception as e:
                # If an smdebug error occurred with the default configuration or it occurred before the configuration
                # can even be determined (i.e. the constructor), catch the error and log it.
                if self.hook is None or self.hook.has_default_configuration():
                    self.logger.error(BASE_ERROR_MESSAGE)
                    self.logger.exception(e)  # Log stack trace.
                    self.disabled = True  # Disable smdebug
                else:
                    raise e

        return error_handler


# class GetHookErrorHandler(ErrorHandler):
#     def catch_smdebug_errors(self, func):
#         @functools.wraps(func)
#         def error_handler(*a, **kw):
#             # Return immediately if smdebug is disabled.
#             if self.disabled:
#                 return
#
#             try:
#                 func(*a, **kw)
#             except Exception as e:
#                 # If an smdebug error occurred with the default configuration or it occurred before the configuration
#                 # can even be determined (i.e. the constructor), catch the error and log it.
#                 if self.hook is None or self.hook.has_default_configuration():
#                     self.logger.error(BASE_ERROR_MESSAGE)
#                     self.logger.exception(e)  # Log stack trace.
#                     self.disabled = True  # Disable smdebug
#                 else:
#                     raise e
#
#         return error_handler
