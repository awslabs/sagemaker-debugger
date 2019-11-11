import tensorflow as tf
from packaging import version

if version.parse(tf.__version__) >= version.parse("2.0.0") or version.parse(
    tf.__version__
) < version.parse("1.13.0"):
    raise ImportError("Tornasole only supports TensorFlow 1.13.0 <= version <= 1.15.x")

# If using keras standalone, it has to be 2.3.x

from .session import TornasoleHook, TornasoleSessionHook, TornasoleEstimatorHook
from .keras import TornasoleKerasHook

from .collection import Collection, CollectionManager

from .collection import (
    get_collections,
    get_collection,
    load_collections,
    export_collections,
    add_to_collection,
    add_to_default_collection,
    reset_collections,
)

from .singleton_utils import get_hook, set_hook, del_hook
from tornasole.trials import create_trial
from tornasole import modes
from tornasole.core.collection import CollectionKeys
from tornasole import SaveConfig, SaveConfigMode, ReductionConfig
