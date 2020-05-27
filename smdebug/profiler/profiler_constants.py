# The traceevents will be stored in following format
# $ENV_BASE_FOLDER/framework/pevents/$START_TIME_YYYYMMDDHR/$FILEEVENTSTARTTIMEUTCINEPOCH_{
# $ENV_NODE_ID_4digits0padded}_model_timeline.json
DEFAULT_PREFIX = "framework/pevents"
TRACE_DIRECTORY_FORMAT = "%Y%m%d%H"
PYTHONTIMELINE_PREFIX = "pythontimeline.json"
MODELTIMELINE_PREFIX = "model_timeline.json"
TENSORBOARDTIMELINE_PREFIX = "tensorboard_timeline.json"
HOROVODTIMELINE_PREFIX = "horovod_timeline.json"
# In order to look for events occurred within a window, we will add a buffer of 3 minutes on each side (start and
# time) to look for trace files.
TIME_BUFFER = 3 * 60 * 1000 * 1000  # 3 minutes
