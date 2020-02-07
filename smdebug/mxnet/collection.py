# First Party
from smdebug.core.collection import DEFAULT_MXNET_COLLECTIONS
from smdebug.core.collection import Collection as BaseCollection
from smdebug.core.collection import CollectionKeys
from smdebug.core.collection_manager import CollectionManager as BaseCollectionManager


class Collection(BaseCollection):
    def add_block_tensors(self, block, inputs=False, outputs=True):
        if inputs:
            input_tensor_regex = block.name + "_input_*"
            self.include(input_tensor_regex)
        if outputs:
            output_tensor_regex = block.name + "_output_*"
            self.include(output_tensor_regex)


class CollectionManager(BaseCollectionManager):
    def __init__(self, create_default=True):
        super().__init__(create_default=create_default)
        if create_default:
            self._register_default_collections()

    def _register_default_collections(self):
        for c in DEFAULT_MXNET_COLLECTIONS:
            self.create_collection(c)
        self.get(CollectionKeys.WEIGHTS).include("^(?!gradient).*weight")
        self.get(CollectionKeys.BIASES).include("^(?!gradient).*bias")
        self.get(CollectionKeys.GRADIENTS).include("^gradient")
        self.get(CollectionKeys.LOSSES).include(".*loss._(?!input).*output")

    def create_collection(self, name):
        super().create_collection(name, cls=Collection)

    @classmethod
    def load(cls, filename):
        return super().load(cls, filename, Collection)

    @classmethod
    def load_from_string(cls, s):
        return super().load(cls, s, Collection)
