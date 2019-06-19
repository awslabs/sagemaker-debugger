from .collection import Collection

class CollectionManager:
  """
  CollectionManager lets you manage group of collections.
  It contains a default collection into which tensors are inserted
  without specifying collection name
  """
  def __init__(self, create_default=True):
    self.collections = {}
    if create_default:
      self.collections['default'] = self.get_new_collection('default')

  def get_new_collection(self, name):
    return Collection(name)

  def add(self, arg):
    if isinstance(arg, str):
      if arg not in self.collections:
        self.collections[arg] = self.get_new_collection(arg)
    elif isinstance(arg, Collection):
      if arg.name not in self.collections:
        self.collections[arg.name] = arg

  def get(self, name):
    return self.collections[name]

  def export(self, filename):
    with open(filename, 'w') as f:
      for n,c in self.collections.items():
        f.write(c.export() + '\n')

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

  def __eq__(self, other):
    if not isinstance(other, CollectionManager):
      return NotImplemented
    return self.collections == other.collections
