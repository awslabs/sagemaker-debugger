from tornasole.core.modes import ModeKeys as modes


class StepNotYetAvailable(Exception):
  def __init__(self, step, mode):
    self.step = step
    self.mode = mode

  def __str__(self):
    return 'Step {} of mode {} not yet available'.format(self.step, self.mode)


class StepUnavailable(Exception):
  def __init__(self, step, mode):
    self.step = step
    self.mode = mode

  def __str__(self):
    return 'Step {} of mode {} is not available as it was not saved'\
      .format(self.step, self.mode)


class TensorUnavailableForStep(Exception):
  def __init__(self, tname, step, mode=modes.GLOBAL, has_reductions=False):
    self.step = step
    self.mode = mode
    self.tname = tname
    self.has_reductions = has_reductions

  def __str__(self):
    msg = 'Value for tensor {} is not available for step {} ' \
          'with mode {} as it was not saved.' \
           ''.format(self.tname, self.step, self.mode.name)
    if self.has_reductions:
      msg += 'This tensor has reductions saved for this step. ' \
             'You might want to query for the reductions.'
    return msg

class TensorUnavailable(Exception):
  def __init__(self, tname):
    self.tname = tname

  def __str__(self):
    return 'Tensor {} can not be satisfied. Tornasole does ' \
           'not know about this tensor.'.format(self.tname)


class NoMoreData(Exception):
  pass


class RuleEvaluationConditionMet(Exception):
  def __init__(self, rule_name, step):
    self.rule_name = rule_name
    self.step = step

  def __str__(self):
    return 'Evaluation of the rule {} at step {} resulted in the condition being met'\
      .format(self.rule_name, self.step)
