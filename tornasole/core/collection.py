from .reduction_config import  ReductionConfig
from .save_config import SaveConfig, SaveConfigMode
from .modes import ModeKeys

import json
from typing import Any, Dict, List, Optional, Union

ALLOWED_PARAMS = ['name', 'include_regex', 'reduction_config', 'save_config',
                  'tensor_names', 'save_histogram']

class CollectionKeys:
  DEFAULT = 'default'
  ALL = 'all'

  WEIGHTS = 'weights'
  GRADIENTS = 'gradients'
  LOSSES = 'losses'
  BIASES = 'biases'
  SCALARS = 'scalars'

  TENSORFLOW_SUMMARIES = 'tensorflow_summaries'

  #XGBOOST
  METRIC = "metric"
  PREDICTIONS = "predictions"
  LABELS = "labels"
  FEATURE_IMPORTANCE = "feature_importance"
  AVERAGE_SHAP = "average_shap"

# Collection with summary objects instead of tensors
# so we don't create summaries or reductions of these
SUMMARIES_COLLECTIONS = {
  CollectionKeys.TENSORFLOW_SUMMARIES
}


NON_HISTOGRAM_COLLECTIONS = {
  CollectionKeys.LOSSES, CollectionKeys.SCALARS,
  CollectionKeys.TENSORFLOW_SUMMARIES
}


class Collection:
  """
  Collection object helps group tensors for easier handling during saving as well
  as analysis. A collection has its own list of tensors, reduction config
  and save config. This allows setting of different save and reduction configs
  for different tensors.

  ...
  Attributes
  ----------
  name: str
  name of collection

  include_regex: list of (str representing regex for tensor names or block names)
  list of regex expressions representing names of tensors (tf) or blocks(gluon)
  to include for this collection

  reduction_config: ReductionConfig object
  reduction config to be applied for this collection.
  if this is not passed, uses the default reduction_config

  save_config: SaveConfig object
  save config to be applied for this collection.
  if this is not passed, uses the default save_config
  """
  def __init__(self, name, include_regex=None, tensor_names=None,
               reduction_config=None, save_config=None, save_histogram=True):
    self.name = name
    self.include_regex = include_regex if include_regex is not None else []
    self.set_reduction_config(reduction_config)
    self.set_save_config(save_config)
    self.save_histogram = save_histogram

    # todo: below comment is broken now that we have set. do we need it back?
    # we want to maintain order here so that different collections can be analyzed together
    # for example, weights and gradients collections can have 1:1 mapping if they
    # are entered in the same order
    self.tensor_names = set(tensor_names) if tensor_names is not None else set()

  def get_include_regex(self):
    return self.include_regex

  def get_tensor_names(self):
    return self.tensor_names

  def include(self, t):
    if isinstance(t, list):
      for i in t:
        self.include(i)
    elif isinstance(t, str):
      self.include_regex.append(t)
    else:
      raise TypeError("Can only include str or list")

  def get_reduction_config(self):
    return self.reduction_config

  def get_save_config(self):
    return self.save_config

  def set_reduction_config(self, reduction_config):
    if reduction_config is None:
      self.reduction_config = None
    elif not isinstance(reduction_config, ReductionConfig):
      raise TypeError(f"reduction_config={reduction_config} must be of type ReductionConfig")
    else:
      self.reduction_config = reduction_config

  def set_save_config(
    self,
    save_config: Union[SaveConfig, Dict[ModeKeys, SaveConfigMode]],
  ):
    """Pass in either a fully-formed SaveConfig, or a dictionary with partial keys mapping to SaveConfigMode.

    If partial keys are passed (for example, only ModeKeys.TRAIN), then the other mdoes are populated
    from `base_save_config`.
    """
    if save_config is None:
      self.save_config = None
    elif isinstance(save_config, dict):
      self.save_config = SaveConfig(mode_save_configs=save_config)
    elif isinstance(save_config, SaveConfig):
      self.save_config = save_config
    else:
      raise ValueError(f"save_config={save_config} must be of type SaveConfig of type Dict[ModeKeys, SaveConfigMode]")

  def add_tensor_name(self, tname):
    if tname not in self.tensor_names:
      self.tensor_names.add(tname)

  def remove_tensor_name(self, tname):
    if tname in self.tensor_names:
      self.tensor_names.remove(tname)

  def to_json_dict(self) -> Dict:
    return {
      "name": self.name,
      "include_regex": self.include_regex,
      "tensor_names": sorted(list(self.tensor_names)) if self.tensor_names else [], # Sort for determinism
      "reduction_config": self.reduction_config.to_json_dict() if self.reduction_config else None,
      "save_config": self.save_config.to_json_dict() if self.save_config else None,
      "save_histogram": self.save_histogram
    }

  def to_json(self) -> str:
    return json.dumps(self.to_json_dict())

  @classmethod
  def from_dict(cls, params: Dict) -> 'Collection':
    if not isinstance(params, dict):
      raise ValueError(f"params={params} must be dict")

    res = {
      "name": params.get("name"),
      "include_regex": params.get("include_regex", False),
      "tensor_names": set(params.get("tensor_names", [])),
      "reduction_config": ReductionConfig.from_dict(params["reduction_config"]) if "reduction_config" in params else None,
      "save_config": SaveConfig.from_dict(params["save_config"]) if "save_config" in params else None,
      "save_histogram": params.get("save_histogram", True)
    }
    return cls(**res)

  @classmethod
  def from_json(cls, json_str: str) -> 'Collection':
    return cls.from_dict(json.loads(json_str))

  def __str__(self):
    return str(self.to_json_dict())

  def __hash__(self):
    return hash(self.name)

  def __repr__(self):
    return (
        f"<class Collection: name={self.name}>"
    )

  def __eq__(self, other):
    if not isinstance(other, Collection):
      return NotImplemented

    return self.name == other.name and \
           self.include_regex == other.include_regex and \
           self.tensor_names == other.tensor_names and \
           self.reduction_config == other.reduction_config and \
           self.save_config == other.save_config and \
           self.save_histogram == other.save_histogram
