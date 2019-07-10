SAVE_CONFIG_VERSION_NUM = 'v0'

class SaveConfig:
  """
  Wrapping all the save configuration parameters into this object.
  This would make it easier to set different save configuration for
  different collections and for the base tensors saved.

  ...

  Attributes
  ----------

  save_interval: int
    save every n steps

  skip_num_steps: int
    start saving after n steps

  save_steps: list of int
    save at all the steps given in this list.
    if this is given, it ignores the save_interval

  when_nan: list of str representing name of tensor
    saves whenever any of the tensors in this list become nan.
  """

  def __init__(self, save_interval=100, skip_num_steps=0, save_steps=None, when_nan=None):
    self.save_interval = int(save_interval)
    self.save_steps = save_steps if save_steps is not None else []
    self.skip_num_steps = skip_num_steps
    self.when_nan = when_nan if when_nan is not None else []

  def export(self):
    separator = '%'
    list_separator = ','

    return separator.join([SAVE_CONFIG_VERSION_NUM, str(self.save_interval),
                           str(self.skip_num_steps),
                           list_separator.join([str(x) for x in self.save_steps]),
                           list_separator.join(self.when_nan)])

  @staticmethod
  def load(s):
    if s is None or s == str(None):
      return None

    separator = '%'
    parts = s.split(separator)
    s_version = parts[0]
    if s_version == 'v0':
      assert len(parts) == 5
      list_separator = ','
      save_interval = int(parts[1])
      skip_num_steps = int(parts[2])
      save_steps = [int(x) for x in parts[3].split(list_separator) if x]
      when_nan = [x for x in parts[4].split(list_separator) if x]
      return SaveConfig(save_interval=save_interval, skip_num_steps=skip_num_steps,
                        save_steps=save_steps, when_nan=when_nan)
    raise RuntimeError('Unable to load SaveConfig from %s' % s)

  def __eq__(self, other):
    if not isinstance(other, SaveConfig):
      return NotImplemented
    return self.save_interval == other.save_interval and \
           self.save_steps == other.save_steps and \
           self.skip_num_steps == other.skip_num_steps and \
           self.when_nan == other.when_nan

  def should_save_step(self, step_num):
    rval = {'step': False, 'when_nan': False}
    if self.save_steps:
      if step_num in self.save_steps:
        rval['step'] = True
    elif step_num >= self.skip_num_steps and step_num % self.save_interval == 0:
      rval['step'] = True
    elif self.when_nan:
      rval['when_nan'] = True
    return rval

