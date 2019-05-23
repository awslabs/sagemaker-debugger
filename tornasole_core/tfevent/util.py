from .tensor_pb2 import TensorProto
from .tensor_shape_pb2 import TensorShapeProto
import numpy as np


_NP_DATATYPE_TO_PROTO_DATATYPE = {
    np.float16: "DT_FLOAT16",
    np.float32:"DT_FLOAT",
    np.float64:"DT_DOUBLE",
    #float64:"DT_DOUBLE",
    np.int32:"DT_INT32",
    np.int64:"DT_INT64",
    np.uint8:"DT_UINT8",
    np.uint16:"DT_UINT16",
    np.uint32:"DT_UINT32",
    np.uint64:"DT_UINT64",
    np.int8:"DT_INT8",
    np.int16:"DT_INT16",
    np.complex64:"DT_COMPLEX64",
    np.complex128:"DT_COMPLEX128",
    np.bool:"DT_BOOL"
}

def _get_proto_dtype(npdtype):
    if npdtype == np.float64:
        return "DT_DOUBLE"
    if npdtype == np.float32:
        return "DT_FLOAT"
    return _NP_DATATYPE_TO_PROTO_DATATYPE[npdtype]

def make_tensor_proto(nparray_data, tag):
    dimensions = [TensorShapeProto.Dim(size=d, name="{0}_{1}".format(tag, d)) for d in nparray_data.shape]
    tensor_proto = TensorProto(dtype=_get_proto_dtype(nparray_data.dtype),
                               tensor_content=nparray_data.tostring(),
                               tensor_shape=TensorShapeProto(dim=dimensions))
    return tensor_proto