# Third Party
import tensorflow.compat.v1 as tf
from packaging import version

# Separating this out to a different file because black doesn't like any code before imports
# So this can't go at the top of init.py
# If this is not at the top, then we see unhelpful messages as we fail to import some modules
if version.parse(tf.__version__) < version.parse("1.14.0"):
    raise ImportError(
        "SMDebug only supports TensorFlow 1.14, 1.15, and experimental functionality in 2.x."
    )
