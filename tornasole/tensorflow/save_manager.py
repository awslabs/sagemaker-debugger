from tornasole.core.save_manager import SaveManager


class TFSaveManager(SaveManager):
  def __init__(self, collection_manager, include_collections_names,
                 default_reduction_config,
                 default_save_config):
    super().__init__(collection_manager=collection_manager,
                     include_collections_names=include_collections_names,
                     default_reduction_config=default_reduction_config,
                     default_save_config=default_save_config)
    self.when_nan_tensors = {}

  def prepare_tensors(self):
    for c_name, c in self.collection_manager.get_collections().items():
      if c_name == 'when_nan':
        continue
      if c not in self.save_collections:
        continue
      for t in c.tensors + c.reduction_tensors_added:
        self._add_tensor_to_collection(t, c)

  def _add_tensor_to_collection(self, t, c):
    if t.name not in self.tensor_to_collection:
      self.tensor_to_collection[t.name] = [c]
    else:
      self.tensor_to_collection[t.name].append(c)

  def add_when_nan_tensor(self, collection, tensor):
    self.configs_for_collections[collection.name].add_when_nan_tensor(tensor)
    if tensor.name not in self.when_nan_tensors:
      self.when_nan_tensors[tensor.name] = []
    self.when_nan_tensors[tensor.name].append(collection)
    self._add_tensor_to_collection(tensor, collection)

    if 'when_nan' not in self.collection_manager.collections:
      self.collection_manager.create_collection('when_nan')
    self.collection_manager.get('when_nan').add_tensor(tensor)

  def is_when_nan_tensor(self, tensor_name):
    return tensor_name in self.when_nan_tensors

  def when_nan_collections(self, tensor_name):
    return self.when_nan_tensors[tensor_name]

