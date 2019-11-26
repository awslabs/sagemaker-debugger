# First Party
from smdebug.exceptions import TensorUnavailable


class RequiredTensors:
    def __init__(self, base_trial, other_trials):
        self._per_trial = {}
        self._per_trial[base_trial] = RequiredTensorsForTrial(base_trial)
        if other_trials is not None:
            for trial in other_trials:
                self._per_trial[trial] = RequiredTensorsForTrial(trial)
        self.base_trial = base_trial

    def add(self, name, steps, trial=None, should_match_regex=False):
        if trial is None:
            trial = self.base_trial
        self._per_trial[trial].add(name, steps, should_match_regex=should_match_regex)

    def get_tensor_steps(self, name, trial=None):
        if trial is None:
            trial = self.base_trial
        return self._per_trial[trial].get_tensor_steps(name)

    def get_names(self, trial=None):
        if trial is None:
            trial = self.base_trial
        return self._per_trial[trial].get_names()

    def get(self, trial=None):
        if trial is None:
            trial = self.base_trial
        return self._per_trial[trial].get()

    def clear(self):
        for trial in self._per_trial:
            self._per_trial[trial].clear()

    def fetch(self):
        for trial in self._per_trial:
            self._per_trial[trial].fetch()


class RequiredTensorsForTrial:
    def __init__(self, trial):
        self.trial = trial
        self._tensor_steps = {}
        self._tensors = {}
        self.logger = self.trial.logger

    def _add_steps_for_tensor(self, name, steps):
        if name not in self._tensor_steps:
            self._tensor_steps[name] = steps
        else:
            self._tensor_steps[name].extend(steps)
        if name not in self._tensors:
            self._tensors[name] = self.trial.tensor(name)

    def add(self, name, steps, should_match_regex=False):
        if should_match_regex:
            names = self.trial.tensor_names(regex=[name])
            for name in names:
                self._add_steps_for_tensor(name, steps)
        else:
            if not self.trial.has_tensor(name):
                raise TensorUnavailable(name)

            self._add_steps_for_tensor(name, steps)

    def get_tensor_steps(self, name):
        return self._tensor_steps[name]

    def get_names(self):
        return self._tensor_steps.keys()

    def get(self):
        return self._tensors.values()

    def clear(self):
        self._tensor_steps.clear()
        self._tensors.clear()

    def fetch(self):
        required_steps = set()
        for steps in self._tensor_steps.values():
            required_steps = required_steps.union(set(steps))
        required_steps = sorted(required_steps)
        if required_steps:
            self.logger.debug(f"Waiting for required_steps: {required_steps}")
        self.trial.wait_for_steps(required_steps)

        # fetch in bulk
        # self.trial.get_tensors(self._tensor_steps)

        for name, steps in self._tensor_steps.items():
            # this will raise exception if step is unavailable
            t = self._tensors[name]
            for st in steps:
                t.value(st)
