# Standard Library
import json
from typing import Any, Dict

# First Party
from smdebug.core.logger import get_logger
from smdebug.core.utils import split

logger = get_logger()


ALLOWED_REDUCTIONS = ["min", "max", "mean", "std", "variance", "sum", "prod"]
ALLOWED_NORMS = ["l1", "l2"]
REDUCTION_CONFIG_VERSION_NUM = "v0"
ALLOWED_PARAMS = [
    "reductions",
    "abs_reductions",
    "norms",
    "abs_norms",
    "save_raw_tensor",
    "save_shape",
]


class ReductionConfig:
    """
  ReductionConfig allows the saving of certain reductions of tensors instead
  of saving the full tensor. The motivation here is to reduce the amount of data
  saved.
  Supports a few reduction operations which are computed in the training process
  and then saved. During analysis, these are available as attributes / properties
  of the original tensor.

  ...

  Attributes
  ----------
  reductions: list of str
      takes list of names of reductions to be computed.
      should be one of 'min', 'max', 'median', 'mean', 'std', 'variance', 'sum', 'prod'

  abs_reductions: list of str
      takes list of names of reductions to be computed after converting the tensor
      to abs(tensor) i.e. reductions are applied on the absolute values of tensor.
      should be one of 'min', 'max', 'median', 'mean', 'std', 'variance', 'sum', 'prod'

  norms: list of str
      takes names of norms to be computed of the tensor.
      should be one of 'l1', 'l2'

  abs_norms: list of str
    takes names of norms to be computed of the tensor after taking absolute value
    should be one of 'l1', 'l2'
  """

    def __init__(
        self,
        reductions=None,
        abs_reductions=None,
        norms=None,
        abs_norms=None,
        save_raw_tensor=False,
        save_shape=False,
    ):
        self.reductions = reductions if reductions is not None else []
        self.abs_reductions = abs_reductions if abs_reductions is not None else []
        self.norms = norms if norms is not None else []
        self.abs_norms = abs_norms if abs_norms is not None else []
        self.save_raw_tensor = save_raw_tensor
        self.save_shape = save_shape
        ## DO NOT REMOVE, if you add anything here, please make sure that _check & from_json is updated accordingly
        self._check()

    def _check(self):
        """Ensure that only valid params are passed in; raises ValueError if not."""
        if any([x not in ALLOWED_PARAMS for x in self.__dict__]):
            raise ValueError(
                "allowed params for reduction config can only be one of " + ",".join(ALLOWED_PARAMS)
            )

        if any([x not in ALLOWED_REDUCTIONS for x in self.reductions]):
            raise ValueError("reductions can only be one of " + ",".join(ALLOWED_REDUCTIONS))
        if any([x not in ALLOWED_REDUCTIONS for x in self.abs_reductions]):
            raise ValueError("abs_reductions can only be one of " + ",".join(ALLOWED_REDUCTIONS))
        if any([x not in ALLOWED_NORMS for x in self.norms]):
            raise ValueError("norms can only be one of " + ",".join(ALLOWED_NORMS))
        if any([x not in ALLOWED_NORMS for x in self.abs_norms]):
            raise ValueError("abs_norms can only be one of " + ",".join(ALLOWED_NORMS))
        if not isinstance(self.save_raw_tensor, bool):
            raise ValueError(f"save_raw_tensor={self.save_raw_tensor} must be a boolean")
        if not isinstance(self.save_shape, bool):
            raise ValueError(f"save_shape={self.save_shape} must be a boolean")

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ReductionConfig":
        """Parses a flattened dict with two keys: `save_raw_tensor` and `reductions`."""
        if params is None:
            return None
        if not isinstance(params, dict):
            raise ValueError(f"params={params} must be dict")
        save_shape = params.get("save_shape", False)
        save_raw_tensor = params.get("save_raw_tensor", False)
        # Parse comma-separated string into array
        all_reductions = split(params.get("reductions", ""))
        # Parse list of reductions into various types, e.g. convert "abs_l1_norm" into "l1"
        reductions, norms, abs_reductions, abs_norms = [], [], [], []
        for red in all_reductions:
            if red != "":  # possible artifact of using split()
                if red.startswith("abs_"):
                    if red.endswith("_norm"):
                        abs_norms.append(red.split("_")[1])  # abs_l1_norm -> l1
                    else:
                        abs_reductions.append(red.split("_")[1])  # abs_mean -> mean
                else:
                    if red.endswith("_norm"):
                        norms.append(red.split("_")[0])  # l1_norm -> l1
                    else:
                        reductions.append(red)  # mean -> mean

        return cls(
            reductions=reductions,
            abs_reductions=abs_reductions,
            norms=norms,
            abs_norms=abs_norms,
            save_raw_tensor=save_raw_tensor,
            save_shape=save_shape,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ReductionConfig":
        d = json.loads(json_str)
        return cls.from_dict(d)

    def to_json_dict(self) -> Dict[str, Any]:
        # Convert reductions from various arrays into single comma-separated string
        all_reductions = []
        for red in self.reductions:
            all_reductions.append(red)
        for red in self.norms:
            all_reductions.append(f"{red}_norm")
        for red in self.abs_reductions:
            all_reductions.append(f"abs_{red}")
        for red in self.abs_norms:
            all_reductions.append(f"abs_{red}_norm")
        all_reductions_str = ",".join(all_reductions)
        # Return the dict
        return {
            "save_raw_tensor": self.save_raw_tensor,
            "reductions": all_reductions_str,
            "save_shape": self.save_shape,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict())

    def __eq__(self, other):
        if not isinstance(other, ReductionConfig):
            return NotImplemented

        return (
            self.reductions == other.reductions
            and self.abs_reductions == other.abs_reductions
            and self.norms == other.norms
            and self.abs_norms == other.abs_norms
            and self.save_raw_tensor == other.save_raw_tensor
            and self.save_shape == other.save_shape
        )

    def __repr__(self):
        return (
            f"<class ReductionConfig: reductions={self.reductions}, "
            f"abs_reductions={self.abs_reductions}, norms={self.norms}, abs_norms={self.abs_norms}>, save_shape={self.save_shape}, save_raw_tensor={self.save_raw_tensor}"
        )
