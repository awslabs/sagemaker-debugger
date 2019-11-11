# First Party
from tornasole import SaveConfig, SaveConfigMode, modes
from tornasole.trials import create_trial

# Local
from .collection import (
    add_to_collection,
    add_to_default_collection,
    get_collection,
    get_collection_manager,
    get_collections,
    load_collections,
    reset_collections,
)
from .hook import TornasoleHook
from .singleton_utils import del_hook, get_hook, set_hook
