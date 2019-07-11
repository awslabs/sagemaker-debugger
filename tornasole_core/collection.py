from .reduction_config import  ReductionConfig
from .save_config import SaveConfig

COLLECTION_VERSION_NUM='v0'

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
  def __init__(self, name, include_regex=None,
               reduction_config=None, save_config=None):
    self.name = name

    # we want to maintain order here so that different collections can be analyzed together
    # for example, weights and gradients collections can have 1:1 mapping if they
    # are entered in the same order
    if not include_regex:
      include_regex = []
    self.include_regex = include_regex

    self.reduction_config = reduction_config
    self.save_config = save_config

    self.tensor_names = []
    # when loaded by passing names, tensors method is unsafe
    # todo: handle this
    self.tensors = []

    self.reduction_tensor_names = []
    self.reduction_tensors = []

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

  def set_reduction_config(self, red_cfg):
    if not isinstance(red_cfg, ReductionConfig):
      raise TypeError('Can only take an instance of ReductionConfig')
    self.reduction_config = red_cfg

  def set_save_config(self, save_cfg):
    if not isinstance(save_cfg, SaveConfig):
      raise TypeError('Can only take an instance of SaveConfig')
    self.save_config = save_cfg

  def add_tensor(self, t):
    if t.name not in self.tensor_names:
      self.tensor_names.append(t.name)
      self.tensors.append(t)

  def remove_tensor(self, t):
    # have to compare names because tensors can have variables, \
    # we don't want to end up comparing tensors and variables
    if t.name in self.tensor_names:
      found_index = None
      for i, lt in enumerate(self.tensors):
        if lt.name == t.name:
          found_index = i
      assert found_index is not None
      self.tensors.pop(found_index)
      self.tensor_names.remove(t.name)

  def add_reduction_tensor(self, s):
    self.reduction_tensor_names.append(s.name)
    self.reduction_tensors.append(s)

  def export(self):
    # v0 export
    # defining a format for the exported string so that different versions do not cause issues in the future
    # here, it is a simple line of the following format
    # >> versionNumber <separator>
    # >> CollectionName <separator>
    # >> include_regex <separator>
    # >> names of tensors separated by comma <separator>
    # >> names of abs_reductions separated by comma <separator>
    # >> reduction_config export <separator>
    # >> save_config export
    # The separator in v0 for uniqueness is the string '!@'
    # export only saves names not the actual tensor fields (tensors, reduction_tensors)
    separator = '!@'
    list_separator = ','
    parts = [COLLECTION_VERSION_NUM, self.name,
             list_separator.join(self.include_regex),
             list_separator.join(self.tensor_names),
             list_separator.join(self.reduction_tensor_names),
             self.reduction_config.export() if self.reduction_config else str(None),
             self.save_config.export() if self.save_config else str(None)]
    return separator.join(parts)

  def __str__(self):
    return f'collection_name: {self.name}, include_regex:{self.include_regex},' \
           f'tensors: {self.tensor_names}, reduction_tensors{self.reduction_tensor_names}, ' \
           f'reduction_config:{self.reduction_config}, save_config:{self.save_config}'

  @staticmethod
  def load(s):
    if s is None or s == str(None):
      return None

    separator = '!@'
    parts = s.split(separator)
    if parts[0] == 'v0':
      assert len(parts) == 7
      list_separator = ','
      name = parts[1]
      include = [x for x in parts[2].split(list_separator) if x]
      tensor_names = [x for x in parts[3].split(list_separator) if x]
      reduction_tensor_names = [x for x in parts[4].split(list_separator) if x]
      reduction_config = ReductionConfig.load(parts[5])
      save_config = SaveConfig.load(parts[6])
      c = Collection(name, include_regex=include,
                      reduction_config=reduction_config, save_config=save_config)
      c.reduction_tensor_names = reduction_tensor_names
      c.tensor_names = tensor_names
      return c

  def __eq__(self, other):
    if not isinstance(other, Collection):
      return NotImplemented

    return self.name == other.name and \
           self.include_regex == other.include_regex and \
           self.tensor_names == other.tensor_names and \
           self.reduction_tensor_names == other.reduction_tensor_names and \
           self.reduction_config == other.reduction_config and \
           self.save_config == other.save_config

