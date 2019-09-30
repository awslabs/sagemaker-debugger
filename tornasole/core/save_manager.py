import copy
from typing import Any, Dict, List, Set
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

  def get_collections_to_save_for_step(self, mode, step) -> Set['Collection']:
    """Mark the proper collections to be saved, return a set of those."""
    if not self.prepared:
      self._raise_error()
    if (mode, step) not in self.save_states_cache:
      coll_to_save_for_step = set()
      for coll in self.save_collections:
        sm = self.configs_for_collections[coll.name]
        if sm.should_save_step(mode, step) is True:
          coll_to_save_for_step.add(coll)
      self.save_states_cache[(mode, step)] = coll_to_save_for_step
    return self.save_states_cache[(mode, step)]

  def get_save_config(self, collection, mode):
    if not self.prepared:
      self._raise_error()
    return self.configs_for_collections[collection.name].get_save_config(mode)

  def get_reduction_config(self, collection):
    if not self.prepared:
      self._raise_error()
    return collection.get_reduction_config()

  def get_collections_with_tensor(self, tensor_name) -> Set['Collection']:
    # for tf this will be prepopulated because of prepare_tensors
    if not tensor_name in self.tensor_to_collection:
      # for mxnet it is computed and then cached
      matched_colls = set()
      for coll in self.get_all_collections_to_save():
        if tensor_name in coll.tensor_names:
          # if being matched as reduction,
          # it must be in reduction_tensor_name, not with regex
          matched_colls.add(coll)
        elif match_inc(tensor_name, coll.get_include_regex()):
          coll.add_tensor_name(tensor_name)
          matched_colls.add(coll)
      self.tensor_to_collection[tensor_name] = matched_colls
    return self.tensor_to_collection[tensor_name]

  def should_save_tensor_for_step(self, tensorname, mode, step) -> bool:
    """Returns whether tensorname should be saved for this mode, mode_step
    as a bool
    """
    colls_to_save = self.get_collections_to_save_for_step(mode, step)
    for coll in self.get_collections_with_tensor(tensorname):
      if coll in colls_to_save:
        return True
    return False
