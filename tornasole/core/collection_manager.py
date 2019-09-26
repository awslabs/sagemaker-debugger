from .collection import Collection
from .access_layer import TSAccessFile, TSAccessS3
from .utils import is_s3, load_json_as_dict
import json

ALLOWED_PARAMS = ['collections']
COLLECTIONS_FILE_NAME = 'collections.json'

class CollectionManager:
  """
  CollectionManager lets you manage group of collections.
  It contains a default collection into which tensors are inserted
  without specifying collection name
  """
  def __init__(self, collections=None, create_default=False):
    if collections is None:
      collections = {}
    self.collections = collections

  def create_collection(self, name):
    self.collections[name] = Collection(name)

  def get_collections(self):
    return self.collections

  def add(self, arg):
    if isinstance(arg, str):
      if arg not in self.collections:
        self.create_collection(arg)
    elif isinstance(arg, Collection):
      if arg.name not in self.collections:
        self.collections[arg.name] = arg

  def get(self, name):
    return self.collections[name]

  def to_json_dict(self):
    d = dict()
    for a, v in self.__dict__.items():
      if a == 'collections':
        coll_dict = dict()
        for n, v in self.collections.items():
          coll_dict[n] = v.to_json_dict()
        d[a] = coll_dict
      elif hasattr(v, "to_json_dict"):
        d[a] = v.to_json_dict()
      else:
        d[a] = v
    return d

  def export(self, filename):
    on_s3, bucket, obj = is_s3(filename)
    if on_s3:
      f = TSAccessS3(bucket_name=bucket, key_name=obj, binary=False)
    else:
      f = TSAccessFile(filename, 'w')
    f.write(json.dumps(self.to_json_dict()))
    f.close()

  @classmethod
  def load(cls, filename, collection_class=Collection):
    with open(filename, 'r') as f:
      s = f.read()
      cm = cls.load_from_string(s, collection_class)
    return cm

  @classmethod
  def load_from_string(cls, s, collection_class=Collection):
    params = load_json_as_dict(s)
    if params is not None:
      if any([x not in ALLOWED_PARAMS for x in params]):
        raise ValueError('allowed params for collection manager can '
                         'only be one of ' + ','.join(ALLOWED_PARAMS))
      cm = cls(create_default=False)
      for c_name, c_dict in params['collections'].items():
        coll = collection_class.from_dict(c_dict)
        cm.add(coll)
      return cm

  def __eq__(self, other):
    if not isinstance(other, CollectionManager):
      return NotImplemented
    return self.collections == other.collections

  def __repr__(self):
    return f"<class CollectionManager: collection_names={list(self.collections.keys())}"