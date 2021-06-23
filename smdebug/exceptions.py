# First Party
from smdebug.core.modes import ModeKeys as modes


class InvalidCollectionConfiguration(Exception):
    def __init__(self, c_name):
        self.c_name = c_name

    def __str__(self):
        return f"Collection {self.c_name} has not been configured. \
        Please fill in tensor_name or include_regex"


class StepNotYetAvailable(Exception):
    """This means that the step has not yet been
      seen from the training job. It may be available in the future if the
      training is still going on. We automatically load new data as and
      when it becomes available. This step may either become available in
      the future, or the exception might change to ``StepUnavailable``.

    """

    def __init__(self, step, mode):
        self.step = step
        self.mode = mode

    def __str__(self):
        return "Step {} of mode {} not yet available".format(self.step, self.mode.name)


class MissingCollectionFiles(Exception):
    """This is raised when no data was saved by
      the training job. Check that the ``Hook`` was configured correctly
      before starting the training job.

    """

    def __init__(self):
        pass

    def __str__(self):
        return "Training job has ended. All the collection files could not be loaded"


class IndexReaderException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class StepUnavailable(Exception):
    """This means that the step was not saved from the
      training job. No tensor will be available for this step.

    """

    def __init__(self, step, mode):
        self.step = step
        self.mode = mode

    def __str__(self):
        return "Step {} of mode {} is not available as it was not saved".format(
            self.step, self.mode.name
        )


class TensorUnavailableForStep(Exception):
    """This is raised when the tensor requested
      is not available for the step. It may have been or will be saved for
      a different step number. You can check which steps tensor is saved
      for by ``trial.tensor('tname').steps()``
      `api <https://github.com/awslabs/sagemaker-debugger/blob/master/docs/analysis.md#steps-1>`__.
      Note that this exception implies that the requested tensor will never
      become available for this step in the future.
    """

    def __init__(self, tname, step, mode=modes.GLOBAL, has_reductions=False):
        self.step = step
        self.mode = mode
        self.tname = tname
        self.has_reductions = has_reductions

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


class ShapeUnavailableForStep(Exception):
    def __init__(self, tname, step, mode=modes.GLOBAL):
        self.step = step
        self.mode = mode
        self.tname = tname

    def __str__(self):
        msg = (
            "Shape for tensor {} is not available for step {} "
            "with mode {} as it was not saved."
            "".format(self.tname, self.step, self.mode.name)
        )
        return msg


class TensorUnavailable(Exception):
    """This means that this tensor has not been
      saved from the training job. Note that if you have a ``SaveConfig``
      which saves a certain tensor only after the time you queried for the
      tensor, you might get a ``TensorUnavailable`` exception even if the
      tensor may become available later for some step.

    """

    def __init__(self, tname):
        self.tname = tname

    def __str__(self):
        return "Tensor {} was not saved.".format(self.tname)


class InvalidWorker(Exception):
    def __init__(self, worker):
        self.worker = worker

    def __str__(self):
        return "Invalid Worker: {}".format(self.worker)


class NoMoreProfilerData(Exception):
    """This will be raised when the training ends. Once you
      see this, you will know that there will be no more steps and no more
      tensors saved.

    """

    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.msg = "Looking for timestamp {} and reached " "end of training.".format(timestamp)

    def __str__(self):
        return self.msg


class NoMoreData(Exception):
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

    def __str__(self):
        return self.msg


class RuleEvaluationConditionMet(Exception):
    """This is raised when the rule
      invocation returns ``True`` for some step.

    """

    def __init__(self, rule_name, step, end_of_rule=False):
        self.rule_name = rule_name
        self.step = step
        self.end_of_rule = end_of_rule

    def __str__(self):
        return "Evaluation of the rule {} at step {} resulted in the condition being met".format(
            self.rule_name, self.step
        )


class InsufficientInformationForRuleInvocation(Exception):
    def __init__(self, rule_name, message):
        self.rule_name = rule_name
        self.message = message

    def __str__(self):
        return "Insufficient information to invoke rule {}: {}".format(self.rule_name, self.message)
