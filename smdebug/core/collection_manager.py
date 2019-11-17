# Standard Library
import json
import os

# Local
from .access_layer import TSAccessFile, TSAccessS3
from .collection import Collection
from .utils import get_path_to_collections, is_s3, load_json_as_dict

ALLOWED_PARAMS = ["collections", "_meta"]


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
        self._meta = {"num_workers": 1}

    def create_collection(self, name, cls=Collection):
        if name not in self.collections:
            self.collections[name] = cls(name)

    def get_collections(self):
        return self.collections

    def add(self, arg):
        if isinstance(arg, str):
            if arg not in self.collections:
                self.create_collection(arg)
        elif isinstance(arg, Collection):
            # overrides any existing collection with that name
            self.collections[arg.name] = arg

    def get(self, name, create=True):
        if name not in self.collections:
            if create:
                self.create_collection(name)
            else:
                raise KeyError(f"Collection {name} has not been created")
        return self.collections[name]

    def update_meta(self, meta):
        assert isinstance(meta, dict)
        self._meta.update(meta)

    def get_num_workers(self):
        return int(self._meta["num_workers"])

    def set_num_workers(self, num_workers):
        self._meta["num_workers"] = int(num_workers)

    def to_json_dict(self):
        d = dict()
        for a, v in self.__dict__.items():
            if a == "_tensors":
                continue
            if a == "collections":
                coll_dict = dict()
                for n, v in self.collections.items():
                    coll_dict[n] = v.to_json_dict()
                d[a] = coll_dict
            elif hasattr(v, "to_json_dict"):
                d[a] = v.to_json_dict()
            else:
                d[a] = v
        return d

    def export(self, out_dir, filename):
        filename = os.path.join(get_path_to_collections(out_dir), filename)
        on_s3, bucket, obj = is_s3(filename)
        if on_s3:
            f = TSAccessS3(bucket_name=bucket, key_name=obj, binary=False)
        else:
            f = TSAccessFile(filename, "w")
        f.write(json.dumps(self.to_json_dict()))
        f.close()

    @classmethod
    def load(cls, filename, collection_class=Collection):
        with open(filename, "r") as f:
            s = f.read()
            cm = cls.load_from_string(s, collection_class)
        return cm

    @classmethod
    def load_from_string(cls, s, collection_class=Collection):
        params = load_json_as_dict(s)
        if params is not None:
            if any([x not in ALLOWED_PARAMS for x in params]):
                raise ValueError(
                    "allowed params for collection manager can "
                    "only be one of " + ",".join(ALLOWED_PARAMS)
                )
            cm = cls(create_default=False)
            for c_name, c_dict in params["collections"].items():
                coll = collection_class.from_dict(c_dict)
                cm.add(coll)
            cm.update_meta(params["_meta"])
            return cm

    def __eq__(self, other):
        if not isinstance(other, CollectionManager):
            return NotImplemented
        return self.collections == other.collections

    def __repr__(self):
        return f"<class CollectionManager: " f"collection_names={list(self.collections.keys())}>"
