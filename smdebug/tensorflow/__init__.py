# Third Party
import tensorflow as tf
from packaging import version

# First Party
from smdebug import ReductionConfig, SaveConfig, SaveConfigMode, modes
from smdebug.core.collection import CollectionKeys
from smdebug.trials import create_trial

# Local
from .keras import KerasHook
from .session import EstimatorHook, SessionHook
from .singleton_utils import del_hook, get_hook, set_hook

if version.parse(tf.__version__) >= version.parse("2.0.0") or version.parse(
    tf.__version__
) < version.parse("1.13.0"):
    raise ImportError("SMDebug only supports TensorFlow 1.13.x, 1.14.x and 1.15.x")

# If using keras standalone, it has to be 2.3.x
