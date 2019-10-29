from enum import Enum
from tornasole.core.logger import get_logger

logger = get_logger()


class TensorType(Enum):
    REGULAR = 1
    REDUCTION = 2
    SUMMARY = 3


class TornasoleTFTensor:
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

    @classmethod
    def from_tensor(cls, tensor, name=None):
        try:
            if name is None:
                name = tensor.name
            return TornasoleTFTensor(tensor, tornasole_name=name)
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
            return TornasoleTFTensor(variable.value(), tornasole_name=name)
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
            return TornasoleTFTensor(
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
