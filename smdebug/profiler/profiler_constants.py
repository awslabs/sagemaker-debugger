# The traceevents will be stored in following format
# $ENV_BASE_FOLDER/framework/pevents/$START_TIME_YYMMDDHR/$FILEEVENTSTARTTIMEUTCINEPOCH_{$ENV_NODE_ID_4digits0padded}_model_timeline.json
DEFAULT_PREFIX = "framework/pevents"
PYTHONTIMELINE_PREFIX = "pythontimeline.json"
MODELTIMELINE_PREFIX = "model_timeline.json"
TENSORBOARDTIMELINE_PREFIX = "tensorboard_timeline.json"
HOROVODTIMELINE_PREFIX = "horovod_timeline.json"
