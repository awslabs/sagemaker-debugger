from .collection import Collection
from .access_layer import TSAccessFile, TSAccessS3
from .utils import is_s3

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

  def get_collections(self):
    return self.collections

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
    on_s3, bucket, obj = is_s3(filename)
    if on_s3:
      f = TSAccessS3(bucket_name=bucket, key_name=obj, binary=False)
    else:
      f = TSAccessFile(filename, 'w')

    for n,c in self.collections.items():
      f.write(c.export() + '\n')
    f.close()

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
