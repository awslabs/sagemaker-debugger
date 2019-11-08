from .hook import TornasoleHook
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

from .optimizer import TornasoleOptimizer
from .singleton_utils import get_hook, set_hook, del_hook
from tornasole.trials import create_trial
from tornasole import modes
from tornasole.core.collection import CollectionKeys
from tornasole import SaveConfig, SaveConfigMode, ReductionConfig
