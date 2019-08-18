import tensorflow as tf
from tornasole.core.save_config import SaveConfig
from tornasole.core.reduction_config import ReductionConfig
from tornasole.core.modes import ModeKeys
from tornasole.core.collection import Collection as BaseCollection
from tornasole.core.collection_manager import CollectionManager as BaseCollectionManager


class Collection(BaseCollection):

  def __init__(self, name, include_regex=None,
               reduction_config=None, save_config=None):
    super().__init__(name, include_regex, reduction_config, save_config)
    self.tensors = []
    # has the new tensors added to graph
    # reduction_tensor_names has the names of original tensors
    # whose reductions these are
    self.reduction_tensors_added = []

  def add(self, arg):
    if isinstance(arg, list) or isinstance(arg, set):
      for a in arg:
        self.add(a)
    elif isinstance(arg, tf.Operation):
      for t in arg.outputs:
        self.add_tensor(t)
    elif isinstance(arg, tf.Variable) or isinstance(arg, tf.Tensor):
      self.add_tensor(arg)
    else:
      raise TypeError('Unknown type of argument %s.'
                      'Add can only take tf.Operation, tf.Variable, tf.Tensor'
                      'and list or set of any of the above.' % arg)

  def add_tensor(self, t):
    self.add_tensor_name(t.name)
    # tf tries to add variables both by tensor and variable.
    # to avoid duplications, we need to check names
    for x in self.tensors:
      if x.name == t.name:
        return
    self.tensors.append(t)

  def add_reduction_tensor(self, t, original_tensor):
    self.add_reduction_tensor_name(original_tensor.name)
    # tf tries to add variables both by tensor and variable.
    # to avoid duplications, we need to check names
    for x in self.reduction_tensors_added:
      if x.name == t.name:
        return
    self.reduction_tensors_added.append(t)

  def remove_tensor(self, t):
    # have to compare names because tensors can have variables, \
    # we don't want to end up comparing tensors and variables
    if t.name in self.tensor_names:
      found_index = None
      for i, lt in enumerate(self.tensors):
        if lt.name == t.name:
          found_index = i

      self.tensor_names.remove(t.name)

      # this can happen when tensors is cleared but tensor names is not cleared
      # because of emptying tensors and reduction_tensors lists in
      # prepare_collections
      if found_index is None:
        raise IndexError('Could not find tensor to remove')
      self.tensors.pop(found_index)

    @staticmethod
    def load(s):
      if s is None or s == str(None):
        return None
      sc_separator = '$'
      separator = '!@'
      parts = s.split(separator)
      if parts[0] == 'v0':
        assert len(parts) == 7
        list_separator = ','
        name = parts[1]
        include = [x for x in parts[2].split(list_separator) if x]
        tensor_names = set([x for x in parts[3].split(list_separator) if x])
        reduction_tensor_names = set([x for x in parts[4].split(list_separator) if x])
        reduction_config = ReductionConfig.load(parts[5])
        if sc_separator in parts[6]:
          per_modes = parts[6].split(sc_separator)
          save_config = {}
          for per_mode in per_modes:
            per_mode_parts = per_mode.split(':')
            save_config[ModeKeys[per_mode_parts[0]]] = SaveConfig.load(per_mode_parts[1])
        else:
          save_config = SaveConfig.load(parts[6])
        c = Collection(name, include_regex=include,
                       reduction_config=reduction_config,
                       save_config=save_config)
        c.reduction_tensor_names = reduction_tensor_names
        c.tensor_names = tensor_names
        return c

class CollectionManager(BaseCollectionManager):
  def __init__(self, create_default=True):
    super().__init__()
    if create_default:
      self.create_collection('default')

  def create_collection(self, name):
    self.collections[name] = Collection(name)

  @staticmethod
  def load(filename):
    cm = CollectionManager(create_default=False)
    with open(filename, 'r') as f:
      line = f.readline()
      while line:
        c = Collection.load(line.rstrip())
        cm.add(c)
        line = f.readline()
    return cm

  @staticmethod
  def load_from_string(s):
    cm = CollectionManager(create_default=False)
    lines = s.split('\n')
    for line in lines:
      c = Collection.load(line.rstrip())
      cm.add(c)
    return cm

_collection_manager = CollectionManager()

def reset_collections():
  global _collection_manager
  del _collection_manager
  _collection_manager = CollectionManager()

def add_to_collection(collection_name, args):
  get_collection(collection_name).add(args)

def add_to_default_collection(args):
  add_to_collection('default', args)

def get_collection(collection_name):
  try:
    c = _collection_manager.get(collection_name)
  except KeyError:
    _collection_manager.add(collection_name)
    c = _collection_manager.get(collection_name)
  return c

def get_collections():
  return _collection_manager.collections

def export_collections(path):
  if _collection_manager:
    _collection_manager.export(path)

def get_collection_manager():
  return _collection_manager

def load_collections(path):
  global _collection_manager
  _collection_manager = CollectionManager.load(path)
