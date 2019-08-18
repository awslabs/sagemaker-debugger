SAVE_CONFIG_VERSION_NUM = 'v0'
from .modes import ModeKeys as modes
import json

DEFAULT_SAVE_CONFIG_INTERVAL = 100
DEFAULT_SAVE_CONFIG_SKIP_NUM_STEPS = 0
DEFAULT_SAVE_CONFIG_SAVE_STEPS = []
DEFAULT_SAVE_CONFIG_WHEN_NAN = []
ALLOWED_PARAMS = ["save_interval", "save_steps", "skip_num_steps", "when_nan"]

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

  def __init__(self, save_interval=DEFAULT_SAVE_CONFIG_INTERVAL, skip_num_steps=DEFAULT_SAVE_CONFIG_SKIP_NUM_STEPS, save_steps=DEFAULT_SAVE_CONFIG_SAVE_STEPS, when_nan=DEFAULT_SAVE_CONFIG_WHEN_NAN):
    self.save_interval = int(save_interval)
    self.save_steps = save_steps
    self.skip_num_steps = skip_num_steps
    self.when_nan = when_nan
    ## DO NOT RMEOVE, if you add anything here, please make sure that _check & from_json is updated accordingly
    self._check()
    # will be populated by hook
    self.when_nan_tensors = []

  def _check(self):
    if any([x not in ALLOWED_PARAMS for x in self.__dict__]):
      raise ValueError('allowed params for save config can only be one of ' + ','.join(ALLOWED_PARAMS))
    if not isinstance(self.save_interval, int):
      raise ValueError('allowed type in save_interval is int')
    if not isinstance(self.save_steps, list):
      raise ValueError('allowed type in save_steps is list of int')
    if not isinstance(self.skip_num_steps, int):
      raise ValueError('allowed type in skip_num_steps is int')
    if not isinstance(self.when_nan, list):
      raise ValueError('allowed type in when_nan is list of str')
    
  @classmethod
  def from_json(cls, j):
    if isinstance(j, str):
      params = json.loads(j)  
    elif isinstance(j, dict):
      params = j
    else:
      raise ValueError("parameter must be either str or dict")
    if any([x not in ALLOWED_PARAMS for x in params]):
      raise ValueError('allowed params for save config can only be one of ' + ','.join(ALLOWED_PARAMS))   
    save_interval = params.get("save_interval", DEFAULT_SAVE_CONFIG_INTERVAL)
    save_steps = params.get("save_steps", DEFAULT_SAVE_CONFIG_SAVE_STEPS)
    skip_num_steps = params.get("skip_num_steps", DEFAULT_SAVE_CONFIG_SKIP_NUM_STEPS)
    when_nan = params.get("when_nan", DEFAULT_SAVE_CONFIG_WHEN_NAN)
    return cls(save_interval, skip_num_steps, save_steps, when_nan)

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

