from tornasole.core.save_manager import SaveManager


class TFSaveManager(SaveManager):
  def prepare_tensors(self):
    for c_name, c in self.collection_manager.get_collections().items():
      if c not in self.save_collections:
        continue
      for t in c.tensors + c.reduction_tensors_added:
        self._add_tensor_to_collection(t, c)

  def _add_tensor_to_collection(self, t, c):
    if t.name not in self.tensor_to_collection:
      self.tensor_to_collection[t.name] = [c]
    else:
      self.tensor_to_collection[t.name].append(c)
