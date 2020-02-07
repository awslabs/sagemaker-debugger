# First Party
from smdebug.core.collection import DEFAULT_PYTORCH_COLLECTIONS
from smdebug.core.collection import Collection as BaseCollection
from smdebug.core.collection import CollectionKeys
from smdebug.core.collection_manager import CollectionManager as BaseCollectionManager


class Collection(BaseCollection):
    def __init__(
        self,
        name,
        include_regex=None,
        tensor_names=None,
        reduction_config=None,
        save_config=None,
        save_histogram=True,
    ):
        super().__init__(
            name, include_regex, tensor_names, reduction_config, save_config, save_histogram
        )
        # Mapping from module to a tuple of booleans (include_inputs, include_outputs)
        self._modules = {}

    def add_module_tensors(self, module, inputs=False, outputs=True):
        self._modules[module] = (inputs, outputs)

    @property
    def modules(self):
        return self._modules


class CollectionManager(BaseCollectionManager):
    def __init__(self, create_default=True):
        super().__init__(create_default=create_default)
        # self.export_only_once = True
        if create_default:
            self._register_default_collections()

    def _register_default_collections(self):
        for c in DEFAULT_PYTORCH_COLLECTIONS:
            self.create_collection(c)
        self.get(CollectionKeys.WEIGHTS).include("^(?!gradient).*weight")
        self.get(CollectionKeys.BIASES).include("^(?!gradient).*bias")
        self.get(CollectionKeys.GRADIENTS).include("^gradient")
        self.get(CollectionKeys.LOSSES).include("[Ll]oss_(?!input).*output")

    def create_collection(self, name):
        super().create_collection(name, cls=Collection)

    @classmethod
    def load(cls, filename):
        return super().load(cls, filename, Collection)

    @classmethod
    def load_from_string(cls, s):
        return super().load(cls, s, Collection)
