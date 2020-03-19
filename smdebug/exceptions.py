# First Party
from smdebug.core.logger import get_logger
from smdebug.core.modes import ModeKeys as modes


class SMDebugError(Exception):
    def __init__(self):
        logger = get_logger("exceptions")
        logger.error("%s: %s", self.__class__.__name__, str(self))
        for h in logger.handlers:
            h.flush()


class InvalidCollectionConfiguration(SMDebugError):
    def __init__(self, c_name):
        self.c_name = c_name
        super().__init__()

    def __str__(self):
        return f"Collection {self.c_name} has not been configured. Please fill in tensor_name or include_regex"


class StepNotYetAvailable(SMDebugError):
    def __init__(self, step, mode):
        self.step = step
        self.mode = mode
        super().__init__()

    def __str__(self):
        return "Step {} of mode {} not yet available".format(self.step, self.mode.name)


class MissingCollectionFiles(SMDebugError):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Training job has ended. All the collection files could not be loaded"


class IndexReaderException(SMDebugError):
    def __init__(self, message):
        self.message = message
        super().__init__()

    def __str__(self):
        return self.message


class StepUnavailable(SMDebugError):
    def __init__(self, step, mode):
        self.step = step
        self.mode = mode
        super().__init__()

    def __str__(self):
        return "Step {} of mode {} is not available as it was not saved".format(
            self.step, self.mode.name
        )


class TensorUnavailableForStep(SMDebugError):
    def __init__(self, tname, step, mode=modes.GLOBAL, has_reductions=False):
        self.step = step
        self.mode = mode
        self.tname = tname
        self.has_reductions = has_reductions
        super().__init__()

    def __str__(self):
        msg = (
            "Value for tensor {} is not available for step {} "
            "with mode {} as it was not saved."
            "".format(self.tname, self.step, self.mode.name)
        )
        if self.has_reductions:
            msg += (
                "This tensor has reductions saved for this step. "
                "You might want to query for the reductions."
            )
        return msg


class TensorUnavailable(SMDebugError):
    def __init__(self, tname):
        self.tname = tname
        super().__init__()

    def __str__(self):
        return "Tensor {} was not saved.".format(self.tname)


class InvalidWorker(SMDebugError):
    def __init__(self, worker):
        self.worker = worker
        super().__init__()

    def __str__(self):
        return "Invalid Worker: {}".format(self.worker)


class NoMoreData(SMDebugError):
    def __init__(self, step, mode, last_step):
        self.step = step
        self.mode = mode
        self.last_step = last_step

        self.msg = (
            "Looking for step {} of mode {} and reached "
            "end of training. Max step available is {}".format(
                self.step, self.mode.name, self.last_step
            )
        )
        super().__init__()

    def __str__(self):
        return self.msg


class RuleEvaluationConditionMet(SMDebugError):
    def __init__(self, rule_name, step):
        self.rule_name = rule_name
        self.step = step
        super().__init__()

    def __str__(self):
        return "Evaluation of the rule {} at step {} resulted in the condition being met".format(
            self.rule_name, self.step
        )


class InsufficientInformationForRuleInvocation(SMDebugError):
    def __init__(self, rule_name, message):
        self.rule_name = rule_name
        self.message = mesage
        super().__init__()

    def __str__(self):
        return "Insufficient information to invoke rule {}: {}".format(self.rule_name, self.message)
