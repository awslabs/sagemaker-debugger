SAVE_CONFIG_VERSION_NUM = 'v0'
from .modes import ModeKeys as modes


class SaveConfigModes:
  def __init__(self, mode_save_configs=None):
    if mode_save_configs is None:
      mode_save_configs = {}
    self.mode_save_configs = mode_save_configs

  def add_for_all_modes(self, save_config):
    for mode in modes:
      self.mode_save_configs[mode] = save_config

  def add(self, mode, save_config):
    self.mode_save_configs[mode] = save_config

  def should_save_step(self, mode, step_num):
    return self.mode_save_configs[mode].should_save_step(step_num)

  def add_when_nan_tensor(self, tensor):
    for mode in modes:
      self.mode_save_configs[mode].when_nan_tensors.append(tensor)

  def get_save_config(self, mode):
    return self.mode_save_configs[mode]

  @staticmethod
  def create_simple_save_mode(save_config):
    sm = SaveConfigModes()
    sm.add_for_all_modes(save_config)
    return sm


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

    # will be populated by hook
    self.when_nan_tensors = []

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

