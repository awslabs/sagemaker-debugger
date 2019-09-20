SAVE_CONFIG_VERSION_NUM = 'v0'
from tornasole.core.utils import merge_two_dicts, split
from tornasole.core.modes import ModeKeys
import json
from typing import Dict

DEFAULT_SAVE_CONFIG_INTERVAL = 100
DEFAULT_SAVE_CONFIG_SKIP_NUM_STEPS = 0
DEFAULT_SAVE_CONFIG_SAVE_STEPS = []
DEFAULT_SAVE_CONFIG_WHEN_NAN = []
ALLOWED_PARAMS = ["save_interval", "save_steps", "skip_num_steps", "when_nan"]

class SaveConfigModes:
  """Maps modes to SaveConfigs."""

  def __init__(self, mode_save_configs: Dict):
    """Pass in a dictionary mapping modes to SaveConfigs. No parsing here.

    Parameters:
      mode_save_configs (dict, e.g. {
        ModeKeys.TRAIN: SaveConfig,
        ModeKeys.EVAL: SaveConfig,
        ModeKeys.PREDICT: SaveConfig,
        ModeKeys.GLOBAL: SaveConfig
      }).
    """
    if not all([isinstance(mode, ModeKeys) and isinstance(val, SaveConfig) for mode, val in mode_save_configs.items()]):
      raise ValueError(f"Each key,value in mode_save_configs={mode_save_configs} must be of type ModeKey,SaveConfig")
    for mode in ModeKeys:
      if mode not in mode_save_configs:
        mode_save_configs[mode] = SaveConfig()
    self.mode_save_configs = mode_save_configs

  def get_save_config(self, mode):
    return self.mode_save_configs[mode]

  def add(self, mode, save_config):
    self.mode_save_configs[mode] = save_config

  def should_save_step(self, mode, step_num):
    return self.get_save_config(mode).should_save_step(step_num)

  def add_when_nan_tensor(self, tensor):
    for mode in ModeKeys:
      self.get_save_config(mode).when_nan_tensors.append(tensor)

  @staticmethod
  def create_simple_save_mode(save_config):
    return SaveConfigModes(mode_save_configs={
      mode: save_config
      for mode in ModeKeys
    })

  def __repr__(self):
    return f"<class SaveConfigModes: {self.mode_save_configs}>"


class SaveConfig:
  """
  Wrapping all the save configuration parameters into this object.
  This would make it easier to set different save configuration for
  different collections and for the base tensors saved.

  Parameters:
    save_interval (int): Save every n steps.
    skip_num_steps (int): Start saving after n steps.
    save_steps (list of int): Save at all the steps given in this list. Overrides save_interval.
    when_nan (list of str): Saves whenever any of the tensors in this list become nan.
  """
  def __init__(self, save_interval=None, skip_num_steps=None, save_steps=None, when_nan=None):
    self.save_interval = save_interval or DEFAULT_SAVE_CONFIG_INTERVAL
    self.save_steps = save_steps or DEFAULT_SAVE_CONFIG_SAVE_STEPS
    self.skip_num_steps = skip_num_steps or DEFAULT_SAVE_CONFIG_SKIP_NUM_STEPS
    self.when_nan = when_nan or DEFAULT_SAVE_CONFIG_WHEN_NAN
    ## DO NOT REMOVE, if you add anything here, please make sure that _check & from_json is updated accordingly
    self._check()
    # will be populated by hook
    self.when_nan_tensors = []

  def _check(self):
    if any([x not in ALLOWED_PARAMS for x in self.__dict__]):
      raise ValueError('allowed params for save config can only be one of ' + ','.join(ALLOWED_PARAMS))
    if not isinstance(self.save_interval, int):
      raise ValueError('allowed type in save_interval is int')
    if not (isinstance(self.save_steps, list) and all([isinstance(x, int) for x in self.save_steps])):
      raise ValueError('allowed type in save_steps is list of int')
    if not isinstance(self.skip_num_steps, int):
      raise ValueError('allowed type in skip_num_steps is int')
    if not (isinstance(self.when_nan, list) and all([isinstance(x, str) for x in self.when_nan])):
      raise ValueError('allowed type in when_nan is list of str')

  @classmethod
  def from_dict(cls, params):
    if not isinstance(params, dict):
      raise ValueError(f"params={params} is not a dict.")
    return cls(
            save_interval=params.get("save_interval"),
            skip_num_steps=params.get("skip_num_steps"),
            save_steps=params.get("save_steps"),
            when_nan=params.get("when_nan")
          )

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

  def __repr__(self):
    return (
      f"<class SaveConfig: save_interval={self.save_interval}, save_steps={self.save_steps}, "
      f"skip_num_steps={self.skip_num_steps}, when_nan={self.when_nan}>"
    )
