# The traceevents will be stored in following format
# $ENV_BASE_FOLDER/framework/pevents/$START_TIME_YYYYMMDDHR/$FILEEVENTSTARTTIMEUTCINEPOCH_{
# $ENV_NODE_ID_4digits0padded}_model_timeline.json
DEFAULT_PREFIX = "framework/pevents"
TRACE_DIRECTORY_FORMAT = "%Y%m%d%H"
PYTHONTIMELINE_SUFFIX = "pythontimeline.json"
MODELTIMELINE_SUFFIX = "model_timeline.json"
TENSORBOARDTIMELINE_SUFFIX = "tensorboard_timeline.json"
HOROVODTIMELINE_PREFIX = "horovod_timeline.json"
# In order to look for events occurred within a window, we will add a buffer of 3 minutes on each side (start and
# time) to look for trace files.
TIME_BUFFER = 3 * 60 * 1000 * 1000  # 3 minutes

CONFIG_PATH_DEFAULT = "/opt/ml/input/config/profilerconfig.json"
CONVERT_TO_MICROSECS = 1000000
MAX_FILE_SIZE_DEFAULT = 10485760  # default 10MB
CLOSE_FILE_INTERVAL_DEFAULT = 60  # default 60 seconds
FILE_OPEN_FAIL_THRESHOLD_DEFAULT = 50
BASE_FOLDER_DEFAULT = "/var/opt/im/metrics/"

PROFILER_NUM_STEPS_DEFAULT = 1
PROFILER_DURATION_DEFAULT = None  # profile until next batch
