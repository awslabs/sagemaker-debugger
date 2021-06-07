# The traceevents will be stored in following format
# $ENV_BASE_FOLDER/framework/pevents/$START_TIME_YYYYMMDDHR/$FILEEVENTSTARTTIMEUTCINEPOCH_{
# $ENV_NODE_ID_4digits0padded}_model_timeline.json
DEFAULT_PREFIX = "framework/pevents"
DEFAULT_SYSTEM_PROFILER_PREFIX = "system"
TRACE_DIRECTORY_FORMAT = "%Y%m%d%H"
PYTHONTIMELINE_SUFFIX = "pythontimeline.json"
MODELTIMELINE_SUFFIX = "model_timeline.json"
TENSORBOARDTIMELINE_SUFFIX = "trace.json.gz"
HOROVODTIMELINE_SUFFIX = "horovod_timeline.json"
SMDATAPARALLELTIMELINE_SUFFIX = "smdataparallel_timeline.json"
MERGEDTIMELINE_SUFFIX = "merged_timeline.json"
PT_DATALOADER_WORKER = "DataLoaderWorker"
PT_DATALOADER_ITER = "DataLoaderIter"
PT_DATALOADER_INITIALIZE = "DataLoaderIterInitialize"
TF_DATALOADER_ITER = "DataIterator"

"""
When users query the events within certain time range, the value TIME BUFFER_SECONDS is used to extend the time range.
This is done so that we can find the candidate tracefiles that can potentially contain the events that might have
started within the given range but not completed by the given 'end' timestamp. Such events will be reported in the
tracefile corresponding to timestamp later than the given 'end' timestamp.
"""
# Environment variable to set the time buffer in seconds.
ENV_TIME_BUFFER = "TIME_BUFFER_MICROSECONDS"
# In order to look for events occurred within a window, we will add a buffer of 3 minutes on each side (start and
# time) to look for trace files.
TIME_BUFFER_DEFAULT = 3 * 60 * 1000 * 1000  # 3 minutes

"""
The S3MetricReader obtains the list of prefixes (i.e. the list of tracefiles available in S3 bucket) using
‘list_objects_v2’ API and providing the ‘start_prefix’. The ‘start_prefix’  indicates ‘list_object_v2’
API to return the prefixes that are after the ‘start_prefix’.
If we set the 'start_prefix' equivalent to the latest available tracefile obtained in the previous ‘list_object_v2’,
we may miss the files that have timestamps less than latest available timestamp but arrived later (after the previous
invocation of 'list_object_v2').
For example, assume that the latest available file contains the end timestamp to be ‘T_n’.  It is possible that in
one of the nodes in distributed training, there exists a file ‘T_m’ (where T_m < Tn) that is closed but not yet
uploaded to S3. If we set ‘start_prefix’ to be ‘T_n’ for subsequent  ‘list_objects_v2’, we will never enumerate ‘T_m’
file and hence we will never report events from that file.
In order to handle such cases, we set the ‘start_prefix’  to be trailing behind the last available
timestamp. The environment variable "TRAILING_DURATION_SECONDS" controls how far the start_prefix is trailing behind
the latests tracefile available. The default value for this duration is 5 minutes.
This means that we expect that if the file is closed on the node, we will wait for 5 minutes for it to be uploaded to S3.
"""
# Environment variable to set the trailing duration in seconds.
ENV_TRAIILING_DURATION = "TRAILING_DURATION_MICROSECONDS"
# This is a duration used for computing the start after prefix.
TRAILING_DURATION_DEFAULT = 5 * 60 * 1000 * 1000  # 5 minutes

CONFIG_PATH_DEFAULT = "/opt/ml/input/config/profilerconfig.json"
CONVERT_TO_MICROSECS = 1000000
CONVERT_MICRO_TO_NS = 1000
MAX_FILE_SIZE_DEFAULT = 10485760  # default 10MB
CLOSE_FILE_INTERVAL_DEFAULT = 60  # default 60 seconds
FILE_OPEN_FAIL_THRESHOLD_DEFAULT = 50
BASE_FOLDER_DEFAULT = "/opt/ml/output/profiler"

PYTHON_PROFILING_START_STEP_DEFAULT = 9
PROFILING_NUM_STEPS_DEFAULT = 1
PROFILER_DURATION_DEFAULT = float("inf")

TF_METRICS_PREFIX = "aws_marker-"

TF_DATALOADER_START_FLAG_FILENAME = "tf_dataloader_start_flag.tmp"
TF_DATALOADER_END_FLAG_FILENAME = "tf_dataloader_end_flag.tmp"

CPROFILE_STATS_FILENAME = "python_stats"
PYINSTRUMENT_JSON_FILENAME = "python_stats.json"
PYINSTRUMENT_HTML_FILENAME = "python_stats.html"
CPROFILE_NAME = "cprofile"
PYINSTRUMENT_NAME = "pyinstrument"
