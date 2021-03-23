# Standard Library
from abc import ABC, abstractmethod

# First Party
from smdebug.analysis.utils import no_refresh
from smdebug.core.logger import get_logger
from smdebug.exceptions import RuleEvaluationConditionMet
from smdebug.rules.action import Actions

# Local
from .req_tensors import RequiredTensors


# This is Rule interface
class Rule(ABC):
    def __init__(self, base_trial, action_str, other_trials=None):
        self.base_trial = base_trial
        self.other_trials = other_trials

        self.trials = [base_trial]
        if self.other_trials is not None:
            self.trials += [x for x in self.other_trials]

        self.req_tensors = RequiredTensors(self.base_trial, self.other_trials)

        self.logger = get_logger()
        self.rule_name = self.__class__.__name__
        self._actions = Actions(action_str, rule_name=self.rule_name)
        self.report = {
            "RuleTriggered": 0,
            "Violations": 0,
            "Details": {},
            "Datapoints": 0,
            "RuleParameters": "",
        }

    def set_required_tensors(self, step):
        pass

    # step here is global step
    @abstractmethod
    def invoke_at_step(self, step):
        # implementation check for tensor
        # do checkpoint if needed at periodic interval
        # --> storage_handler.save("last_processed_tensor",(tensor_name,step))
        # check-pointing is needed if execution is longer duration,
        # so that we don't lose the work done in certain step
        pass

    # step specific for which global step this rule was invoked
    # storage_handler is used to save & get states across different invocations
    def invoke(self, step):
        self.logger.debug("Invoking rule {} for step {}".format(self.rule_name, step))
        self.base_trial.wait_for_steps([step])

        # do not refresh during invoke at step
        # since we have already waited till the current step
        # this will ensure that the step numbers seen
        # by required_tensors are the same as seen by invoke
        with no_refresh(self.trials):
            self.req_tensors.clear()
            self.set_required_tensors(step)
            self.req_tensors.fetch()
            val = self.invoke_at_step(step)

        if val:
            self._actions.invoke()
            raise RuleEvaluationConditionMet(self.rule_name, step)
