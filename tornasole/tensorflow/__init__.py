# Third Party
import tensorflow as tf
from packaging import version

# First Party
from tornasole import ReductionConfig, SaveConfig, SaveConfigMode, modes
from tornasole.core.collection import CollectionKeys
from tornasole.trials import create_trial

# Local
from .collection import (
    Collection,
    CollectionManager,
    add_to_collection,
    add_to_default_collection,
    export_collections,
    get_collection,
    get_collections,
    load_collections,
    reset_collections,
)
from .keras import TornasoleKerasHook
from .session import TornasoleEstimatorHook, TornasoleHook, TornasoleSessionHook
from .singleton_utils import del_hook, get_hook, set_hook

if version.parse(tf.__version__) >= version.parse("2.0.0") or version.parse(
    tf.__version__
) < version.parse("1.13.0"):
    raise ImportError("Tornasole only supports TensorFlow 1.13.0 <= version <= 1.15.x")

# If using keras standalone, it has to be 2.3.x
