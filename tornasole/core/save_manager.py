import copy
from typing import Any, Dict, List

from .save_config import SaveConfigMode, SaveConfig
from .utils import match_inc
from .modes import ModeKeys

class SaveManager:
  """Main container for all configuration.

  TODO(rahul003): Refactor this into a BaseHook class.
  """

  def __init__(self, collection_manager, include_collections_names,
               default_reduction_config,
               default_save_config):
    self.configs_for_collections = {}
    self.default_save_modes = SaveConfig.parse(default_save_config)
    # Instantiate defaults
    self.default_reduction_config = default_reduction_config
    self.collection_manager = collection_manager
    self.include_collections_names = include_collections_names
    self.save_collections = []
    self.save_states_cache = {}
    # todo clear cache for old steps
    self.tensor_to_collection = {}
    self.prepared = False

  def prepare(self):
    """Ensure every collection has a save_config and reduction_config."""
    # Populate save_collections
    for c_name, c in self.collection_manager.get_collections().items():
      if self._should_collection_be_saved(c_name) and c not in self.save_collections:
        self.save_collections.append(c)

    # Populate configs_for_collections and reduction_config
    for c_name, c in self.collection_manager.get_collections().items():
      # Set to the default if None
      if c.save_config is None:
        self.configs_for_collections[c_name] = self.default_save_modes
      # Otherwise, set missing modes to the defaults
      elif isinstance(c.save_config, SaveConfig):
        # Populate missing modes
        for mode in ModeKeys:
          if c.save_config.mode_save_configs[mode] is None:
            if self.default_save_modes.mode_save_configs[mode] is not None:
              c.save_config.set_save_config(
                mode=mode,
                save_config_mode=copy.deepcopy(self.default_save_modes.get_save_config(mode))
              )
            else:
              c.save_config.set_save_config(mode=mode, save_config_mode=SaveConfigMode())
        # Set the save config
        self.configs_for_collections[c_name] = c.save_config
      else:
        raise ValueError(f"save_config={c.save_config} must be None or SaveConfig")

      if c.reduction_config is None and self.default_reduction_config is not None:
        c.reduction_config = self.default_reduction_config
    self.prepared = True

  def _should_collection_be_saved(self, coll_name):
    return coll_name in self.include_collections_names

  def _raise_error(self):
    raise ValueError('SaveManager is not ready, call prepare() first.')

  def get_all_collections_to_save(self):
    if not self.prepared:
      self._raise_error()
    return self.save_collections

  def collections_to_save(self, mode, step):
    """Mark the proper collections to be saved, return a dictionary of those."""
    if not self.prepared:
      self._raise_error()
    if (mode, step) not in self.save_states_cache:
      collection_save_state = {}
      for coll in self.save_collections:
        sm = self.configs_for_collections[coll.name]
        rv = sm.should_save_step(mode, step)
        if any(rv.values()):
          collection_save_state[coll.name] = rv
      self.save_states_cache[(mode, step)] = collection_save_state
    return self.save_states_cache[(mode, step)]

  def get_save_config(self, collection, mode):
    if not self.prepared:
      self._raise_error()
    return self.configs_for_collections[collection.name].get_save_config(mode)

  def get_reduction_config(self, collection):
    if not self.prepared:
      self._raise_error()
    return collection.get_reduction_config()

  def from_collections(self, tensor_name) -> List['Collection']:
    # for tf this will be prepopulated because of prepare_tensors
    if not tensor_name in self.tensor_to_collection:
      # for mxnet it is computed and then cached
      matched_colls = []
      for coll in self.get_all_collections_to_save():
        if tensor_name in coll.tensor_names:
          # if being matched as reduction,
          # it must be in reduction_tensor_name, not with regex
          matched_colls.append(coll)
        elif match_inc(tensor_name, coll.get_include_regex()):
          coll.add_tensor_name(tensor_name)
          matched_colls.append(coll)
      self.tensor_to_collection[tensor_name] = matched_colls
    return self.tensor_to_collection[tensor_name]

  def should_save_tensor(self, tensorname, mode, step) -> Dict[str, bool]:
    """Return dictionary with two keys: ('step', 'when_nan') mapping to booleans.

    If step is true in the dict, then we are saving this tensor
    because we have hit the step to save this.
    If when_nan is true, we are considering saving this tensor
    because this tensor might be saved if some other tensor is nan.
    """
    colls = self.from_collections(tensorname)
    final_ss = {'step': False, 'when_nan': False}
    ss_colls = self.collections_to_save(mode, step)
    for c in colls:
      if c.name in ss_colls:
        ss = ss_colls[c.name]
        final_ss['step'] = final_ss['step'] or ss['step']
        final_ss['when_nan'] = final_ss['when_nan'] or ss['when_nan']
    return final_ss
