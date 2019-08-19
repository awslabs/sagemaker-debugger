from .tensor_pb2 import TensorProto
from .tensor_shape_pb2 import TensorShapeProto
import numpy as np
import os, re
from tornasole.core.utils import get_immediate_subdirectories
from tornasole.core.utils import get_logger

logger = get_logger()

# hash value of ndarray.dtype is not the same as np.float class
# so we need to convert the type classes below to np.dtype object
_NP_DATATYPE_TO_PROTO_DATATYPE = {
    np.dtype(np.float16): "DT_INT32",
    np.dtype(np.float32):"DT_FLOAT",
    np.dtype(np.float64):"DT_DOUBLE",
    np.dtype(np.int32):"DT_INT32",
    np.dtype(np.int64):"DT_INT64",
    np.dtype(np.uint8):"DT_UINT8",
    np.dtype(np.uint16):"DT_UINT16",
    np.dtype(np.uint32):"DT_UINT32",
    np.dtype(np.uint64):"DT_UINT64",
    np.dtype(np.int8):"DT_INT8",
    np.dtype(np.int16):"DT_INT16",
    np.dtype(np.complex64):"DT_COMPLEX64",
    np.dtype(np.complex128):"DT_COMPLEX128",
    np.dtype(np.bool):"DT_BOOL"
}

def _get_proto_dtype(npdtype):
    if npdtype.kind == 'U':
        return (False, "DT_STRING")
    return (True, _NP_DATATYPE_TO_PROTO_DATATYPE[npdtype])

def make_tensor_proto(nparray_data, tag):
    (isnum, dtype) = _get_proto_dtype(nparray_data.dtype)
    dimensions = [TensorShapeProto.Dim(size=d, name="{0}_{1}".format(tag, d)) for d in nparray_data.shape]
    tps = TensorShapeProto(dim=dimensions)
    if isnum:
        tensor_proto = TensorProto(dtype=dtype,
                                   tensor_content=nparray_data.tostring(),
                                   tensor_shape=tps)
    else:
        tensor_proto = TensorProto(tensor_shape=tps)
        for s in nparray_data:
            sb = bytes(s,encoding='utf-8')
            tensor_proto.string_val.append(sb)
    return tensor_proto

STEP_NUMBER_FORMATTING_LENGTH = '012'

class EventFileLocation:
    def __init__(self, step_num, worker_name):
        self.step_num = int(step_num)
        self.worker_name = worker_name

    def get_location(self, run_dir=''):
        step_num_str = str(format(self.step_num, STEP_NUMBER_FORMATTING_LENGTH))
        event_filename = step_num_str + "_" + self.worker_name + ".tfevents"
        if run_dir:
            event_key_prefix = os.path.join(run_dir, "events")
        else:
            event_key_prefix = "events"
        return os.path.join(event_key_prefix, step_num_str, event_filename)

    @staticmethod
    def match_regex(s):
        return EventFileLocation.load_filename(s, print_error=False)

    @staticmethod
    def load_filename(s, print_error=True):
        last_delimiter_index = s.rfind('/')
        event_file_name = s[last_delimiter_index+1 : ]
        m = re.search('(.*)_(.*).tfevents', event_file_name)
        if m:
            step_num = int(m.group(1))
            worker_name = m.group(2)
            return EventFileLocation(step_num=step_num, worker_name=worker_name)
        else:
            if print_error:
                logger.error('Failed to load efl: ', s)
            return None

    @staticmethod
    def get_step_dirs(trial_dir):
        return get_immediate_subdirectories(os.path.join(trial_dir, 'events'))

    @staticmethod
    def get_step_dir_path(trial_dir, step_num):
        step_num = int(step_num)
        return os.path.join(trial_dir, 'events', format(step_num, STEP_NUMBER_FORMATTING_LENGTH))
