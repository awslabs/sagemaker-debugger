from enum import Enum
from tornasole.core.logger import get_logger

logger = get_logger()


class TensorType(Enum):
    REGULAR = 1
    REDUCTION = 2
    SUMMARY = 3


class Tensor:
    """
        This method allows us to save additional information for a tf.Tensor.
        Sometimes we want to save tensor with different name than the direct TF tensor.
        This happens when we save Variables for example, especially in Keras.
        The tensor we want to save in that case is variable.value(), whose name is weird and non descriptive.
        So we use the variable's name then.
        This also lets us identify the type of tensor, i.e.reduction or summary.
        While saving, we need to identify what the original tensor for these tensors are.
        This class encapsulates that info.
        The tensor_to_collections object in hook is a mapping from name_in_graph to tf.Tensor
        hook._get_ts_tensor() can be used to fetch the tornasole.Tensor object for that tensor.
    """

    def __init__(
        self, obj_in_graph, tornasole_name=None, type=TensorType.REGULAR, original_tensor=None
    ):
        self.obj_in_graph = obj_in_graph
        self.name_in_graph = obj_in_graph.name
        if tornasole_name is None:
            tornasole_name = self.name_in_graph
        self.tornasole_name = tornasole_name
        self.type = type
        if self.type in [TensorType.REDUCTION, TensorType.SUMMARY]:
            assert original_tensor is not None
            self.original_tensor = original_tensor
        else:
            self.original_tensor = None

    @classmethod
    def from_tensor(cls, tensor, name=None):
        try:
            if name is None:
                name = tensor.name
            return Tensor(tensor, tornasole_name=name)
        except AttributeError:
            logger.debug(
                f"Could not create TornasoleTensor from {tensor}. "
                "Perhaps eager mode is turned on"
            )
            return None

    @classmethod
    def from_variable(cls, variable, name=None):
        try:
            if name is None:
                name = variable.name
            return Tensor(variable.value(), tornasole_name=name)
        except AttributeError:
            logger.debug(
                f"Could not create TornasoleTensor from {variable}. "
                "Perhaps eager mode is turned on"
            )
            return None

    @classmethod
    def create_reduction(cls, tensor, original_tensor, name=None):
        try:
            if name is None:
                name = tensor.name
            return Tensor(
                obj_in_graph=tensor,
                tornasole_name=name,
                type=TensorType.REDUCTION,
                original_tensor=original_tensor,
            )
        except AttributeError:
            logger.debug(
                f"Could not create reduction {tensor} of {original_tensor}."
                "Perhaps eager mode is turned on"
            )
            return None
