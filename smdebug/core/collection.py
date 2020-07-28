# Standard Library
import json
from typing import Dict, Union

# Local
from .modes import ModeKeys
from .reduction_config import ReductionConfig
from .save_config import SaveConfig, SaveConfigMode

ALLOWED_PARAMS = [
    "name",
    "include_regex",
    "reduction_config",
    "save_config",
    "tensor_names",
    "save_histogram",
]


class CollectionKeys:
    DEFAULT = "default"
    ALL = "all"
    INPUTS = "inputs"
    OUTPUTS = "outputs"
    WEIGHTS = "weights"
    GRADIENTS = "gradients"
    LOSSES = "losses"
    BIASES = "biases"
    LAYERS = "layers"

    # Use this collection to log scalars other than losses/metrics to SageMaker.
    # Mainly for Tensorflow. For all other frameworks, call save_scalar() API
    # with details of the scalar to be saved.
    SM_METRICS = "sm_metrics"

    OPTIMIZER_VARIABLES = "optimizer_variables"
    TENSORFLOW_SUMMARIES = "tensorflow_summaries"
    METRICS = "metrics"

    # XGBOOST
    HYPERPARAMETERS = "hyperparameters"
    PREDICTIONS = "predictions"
    LABELS = "labels"
    FEATURE_IMPORTANCE = "feature_importance"
    AVERAGE_SHAP = "average_shap"
    FULL_SHAP = "full_shap"
    TREES = "trees"


# Collection with summary objects instead of tensors
# so we don't create summaries or reductions of these
SUMMARIES_COLLECTIONS = {CollectionKeys.TENSORFLOW_SUMMARIES}

SCALAR_COLLECTIONS = {
    CollectionKeys.LOSSES,
    CollectionKeys.METRICS,
    CollectionKeys.FEATURE_IMPORTANCE,
    CollectionKeys.AVERAGE_SHAP,
    CollectionKeys.SM_METRICS,
}

SM_METRIC_COLLECTIONS = {CollectionKeys.LOSSES, CollectionKeys.METRICS, CollectionKeys.SM_METRICS}

# used by pt, mx, keras
NON_REDUCTION_COLLECTIONS = SCALAR_COLLECTIONS.union(SUMMARIES_COLLECTIONS)

NON_HISTOGRAM_COLLECTIONS = SCALAR_COLLECTIONS.union(SUMMARIES_COLLECTIONS)

DEFAULT_TF_COLLECTIONS = {
    CollectionKeys.ALL,
    CollectionKeys.DEFAULT,
    CollectionKeys.WEIGHTS,
    CollectionKeys.BIASES,
    CollectionKeys.GRADIENTS,
    CollectionKeys.LOSSES,
    CollectionKeys.METRICS,
    CollectionKeys.INPUTS,
    CollectionKeys.OUTPUTS,
    CollectionKeys.LAYERS,
    CollectionKeys.SM_METRICS,
    CollectionKeys.OPTIMIZER_VARIABLES,
}

DEFAULT_PYTORCH_COLLECTIONS = {
    CollectionKeys.ALL,
    CollectionKeys.DEFAULT,
    CollectionKeys.WEIGHTS,
    CollectionKeys.BIASES,
    CollectionKeys.GRADIENTS,
    CollectionKeys.LOSSES,
}

DEFAULT_MXNET_COLLECTIONS = {
    CollectionKeys.ALL,
    CollectionKeys.DEFAULT,
    CollectionKeys.WEIGHTS,
    CollectionKeys.BIASES,
    CollectionKeys.GRADIENTS,
    CollectionKeys.LOSSES,
}

DEFAULT_XGBOOST_COLLECTIONS = {
    CollectionKeys.ALL,
    CollectionKeys.DEFAULT,
    CollectionKeys.HYPERPARAMETERS,
    CollectionKeys.PREDICTIONS,
    CollectionKeys.LABELS,
    CollectionKeys.FEATURE_IMPORTANCE,
    CollectionKeys.AVERAGE_SHAP,
    CollectionKeys.FULL_SHAP,
    CollectionKeys.TREES,
}


class Collection:
    """
    Collection object helps group tensors for easier handling during saving as well
    as analysis. A collection has its own list of tensors, reduction config
    and save config. This allows setting of different save and reduction configs
    for different tensors.

    ...
    Attributes
    ----------
    name: str
    name of collection

    include_regex: list of (str representing regex for tensor names or block names)
    list of regex expressions representing names of tensors (tf) or blocks(gluon)
    to include for this collection

    reduction_config: ReductionConfig object
    reduction config to be applied for this collection.
    if this is not passed, uses the default reduction_config

    save_config: SaveConfig object
    save config to be applied for this collection.
    if this is not passed, uses the default save_config
    """

    def __init__(
        self,
        name,
        include_regex=None,
        tensor_names=None,
        reduction_config=None,
        save_config=None,
        save_histogram=True,
    ):
        self.name = name
        self.include_regex = include_regex if include_regex is not None else []
        self.reduction_config = reduction_config
        self.save_config = save_config
        self.save_histogram = save_histogram

        # todo: below comment is broken now that we have set. do we need it back?
        # we want to maintain order here so that different collections can be analyzed together
        # for example, weights and gradients collections can have 1:1 mapping if they
        # are entered in the same order
        self.tensor_names = tensor_names

    @property
    def tensor_names(self):
        return self._tensor_names

    @tensor_names.setter
    def tensor_names(self, tensor_names):
        if tensor_names is None:
            tensor_names = set()
        elif isinstance(tensor_names, list):
            tensor_names = set(tensor_names)
        elif not isinstance(tensor_names, set):
            raise TypeError("tensor_names can only be list or set")
        self._tensor_names = tensor_names

    def include(self, t):
        if isinstance(t, list):
            for i in t:
                self.include(i)
        elif isinstance(t, str):
            self.include_regex.append(t)
        else:
            raise TypeError("Can only include str or list")

    @property
    def reduction_config(self):
        return self._reduction_config

    @property
    def save_config(self):
        return self._save_config

    @reduction_config.setter
    def reduction_config(self, reduction_config):
        if reduction_config is None:
            self._reduction_config = None
        elif not isinstance(reduction_config, ReductionConfig):
            raise TypeError(f"reduction_config={reduction_config} must be of type ReductionConfig")
        else:
            self._reduction_config = reduction_config

    @save_config.setter
    def save_config(self, save_config: Union[SaveConfig, Dict[ModeKeys, SaveConfigMode]]):
        """Pass in either a fully-formed SaveConfig, or a dictionary with partial keys mapping to SaveConfigMode.

        If partial keys are passed (for example, only ModeKeys.TRAIN), then the other mdoes are populated
        from `base_save_config`.
        """
        if save_config is None:
            self._save_config = None
        elif isinstance(save_config, dict):
            self._save_config = SaveConfig(mode_save_configs=save_config)
        elif isinstance(save_config, SaveConfig):
            self._save_config = save_config
        else:
            raise ValueError(
                f"save_config={save_config} must be of type SaveConfig of type Dict[ModeKeys, SaveConfigMode]"
            )

    def has_tensor_name(self, tname):
        return tname in self._tensor_names

    def add_tensor_name(self, tname):
        if tname not in self._tensor_names:
            self._tensor_names.add(tname)

    def remove_tensor_name(self, tname):
        if tname in self._tensor_names:
            self._tensor_names.remove(tname)

    def to_json_dict(self) -> Dict:
        return {
            "name": self.name,
            "include_regex": self.include_regex,
            "tensor_names": sorted(list(self.tensor_names))
            if self.tensor_names
            else [],  # Sort for determinism
            "reduction_config": self.reduction_config.to_json_dict()
            if self.reduction_config
            else None,
            "save_config": self.save_config.to_json_dict() if self.save_config else None,
            "save_histogram": self.save_histogram,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict())

    @classmethod
    def from_dict(cls, params: Dict) -> "Collection":
        if not isinstance(params, dict):
            raise ValueError(f"params={params} must be dict")

        res = {
            "name": params.get("name"),
            "include_regex": params.get("include_regex", False),
            "tensor_names": set(params.get("tensor_names", [])),
            "reduction_config": ReductionConfig.from_dict(params["reduction_config"])
            if "reduction_config" in params
            else None,
            "save_config": SaveConfig.from_dict(params["save_config"])
            if "save_config" in params
            else None,
            "save_histogram": params.get("save_histogram", True),
        }
        return cls(**res)

    @classmethod
    def from_json(cls, json_str: str) -> "Collection":
        return cls.from_dict(json.loads(json_str))

    def __str__(self):
        return str(self.to_json_dict())

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"<class Collection: name={self.name}>"

    def __eq__(self, other):
        if not isinstance(other, Collection):
            return NotImplemented

        return (
            self.name == other.name
            and self.include_regex == other.include_regex
            and self.tensor_names == other.tensor_names
            and self.reduction_config == other.reduction_config
            and self.save_config == other.save_config
            and self.save_histogram == other.save_histogram
        )
