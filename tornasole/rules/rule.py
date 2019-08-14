from tornasole.exceptions import TensorUnavailable
from tornasole.core.utils import get_logger
from tornasole.analysis.utils import no_refresh
from tornasole.exceptions import RuleEvaluationConditionMet
from abc import ABC, abstractmethod

logger = get_logger()

class RequiredTensors:
    def __init__(self, trial):
        self.trial = trial
        self.tensor_names = {}
        self.logger = logger
        self.should_match_regex = {}

    def need_tensor(self, name, steps, should_match_regex=False):
        if name not in self.tensor_names:
            self.tensor_names[name] = steps
        else:
            self.tensor_names[name].extend(steps)

        if should_match_regex:
            self.should_match_regex[name] = True

    def _check_if_steps_available(self, tname, steps):
        t = self.trial.tensor(tname)
        for st in steps:
            t.value(st)

    # returns number of arrays fetched for this rule
    def _fetch_tensors(self):
        required_steps = set()
        for steps in self.tensor_names.values():
            required_steps = required_steps.union(set(steps))
        required_steps = sorted(required_steps)
        if required_steps:
            self.logger.debug(f"Waiting for required_steps: {required_steps}")
        self.trial.wait_for_steps(required_steps)
        self.trial.get_tensors(self.tensor_names,
                               should_regex_match=self.should_match_regex)
        for tensorname, steps in self.tensor_names.items():
            # check whether we should match regex for this tensorname
            # False refers to the default value if the key does not exist in the dictionary
            if self.should_match_regex.get(tensorname, False):
                regex = tensorname
                tnames = self.trial.tensors_matching_regex([regex])
            else:
                tnames = [tensorname]
            for tname in tnames:
                if not self.trial.has_tensor(tname):
                    raise TensorUnavailable(tensorname)
                else:
                    self._check_if_steps_available(tname, steps)

# This is Rule interface
class Rule(ABC):
    def __init__(self, base_trial, other_trials=None):
        self.base_trial = base_trial
        self.other_trials = other_trials

        self.trials = [base_trial]
        if self.other_trials is not None:
            self.trials += [x for x in self.other_trials]

        self.actions = None
        self.logger = logger
        pass

    @abstractmethod
    # returns a list of RequiredTensor objects, one for each trial
    def required_tensors(self, step, **kwargs):
        # TODO implement to read from jsonconfig file
        pass

    # step here is global step
    @abstractmethod
    def invoke_at_step(self, step, storage_handler=None, **kwargs):
        # implementation check for tensor
        # do checkpoint if needed at periodic interval --> storage_handler.save("last_processed_tensor",(tensorname,step))
        # checkpoiniting is needed if execution is longer duration, so that we don't
        # lose the work done in certain step
        pass

    @staticmethod
    def _fetch_tensors_for_trials(req_tensors_requests):
        for req_tensors_request in req_tensors_requests:
            req_tensors_request._fetch_tensors()

    # step specific for which global step this rule was invoked
    # storage_handler is used to save & get states across different invocations
    def invoke(self, step, storage_handler=None, **kwargs):
        self.logger.debug('Invoking rule {} for step {}'.format(self.__class__.__name__, step))
        self.base_trial.wait_for_steps([step])
        req_tensors_requests = self.required_tensors(step)
        self._fetch_tensors_for_trials(req_tensors_requests)

        # do not refresh during invoke at step since required tensors are already here
        with no_refresh(self.trials):
            val = self.invoke_at_step(step)

        if val:
            self.run_actions()
            raise RuleEvaluationConditionMet

    def register_action(self, actions):
        self.actions = actions

    def run_actions(self):
        if self.actions is not None:
            for action in self.actions:
                action.run(rule_name=self.__class__.__name__)
