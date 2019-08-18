from tornasole.core.collection import Collection as BaseCollection
from tornasole.core.reduction_config import ReductionConfig
from tornasole.core.save_config import SaveConfig
from tornasole.core.modes import ModeKeys
from tornasole.core.collection_manager import CollectionManager as BaseCollectionManager


class Collection(BaseCollection):
    def add_block_tensors(self, block, inputs=False, outputs=False):
        if inputs:
            input_tensor_regex = block.name + "_input_*"
            self.include(input_tensor_regex)
        if outputs:
            output_tensor_regex = block.name + "_output"
            self.include(output_tensor_regex)

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
            self._register_default_collections()

    def create_collection(self, name):
        self.collections[name] = Collection(name)

    def _register_default_collections(self):
        weight_collection = Collection('weights', include_regex=['^(?!gradient).*weight'])
        bias_collection = Collection('bias', include_regex=['^(?!gradient).*bias'])
        gradient_collection = Collection('gradients', include_regex=['^gradient'])
        self.add(gradient_collection)
        self.add(weight_collection)
        self.add(bias_collection)

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

    def export_manager(self, path):
        self.export(path)


_collection_manager = CollectionManager()

def load_collections(path):
    global _collection_manager
    _collection_manager = CollectionManager.load(path)

def reset_collections():
    global _collection_manager
    del _collection_manager
    _collection_manager = CollectionManager()

def add_to_collection(collection_name, args):
    get_collection(collection_name).add(args)

def get_collection_manager():
    return _collection_manager

def add_to_default_collection(args):
    add_to_collection('default', args)

def get_collection(collection_name):
    try:
        c = _collection_manager.get(collection_name)
    except KeyError:
        _collection_manager.create_collection(collection_name)
        c = _collection_manager.get(collection_name)
    return c

def get_collections():
    return _collection_manager.collections
