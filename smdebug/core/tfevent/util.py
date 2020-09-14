# Third Party
import numpy as np
from packaging import version

# First Party
from smdebug.core.logger import get_logger

# Local
from .proto.tensor_pb2 import TensorProto
from .proto.tensor_shape_pb2 import TensorShapeProto

logger = get_logger()


def _get_proto_dtype(npdtype):
    # hash value of ndarray.dtype is not the same as np.float class
    # so we need to convert the type classes below to np.dtype object
    _NP_DATATYPE_TO_PROTO_DATATYPE = {
        np.dtype(np.float16): "DT_INT32",
        np.dtype(np.float32): "DT_FLOAT",
        np.dtype(np.float64): "DT_DOUBLE",
        np.dtype(np.int32): "DT_INT32",
        np.dtype(np.int64): "DT_INT64",
        np.dtype(np.uint8): "DT_UINT8",
        np.dtype(np.uint16): "DT_UINT16",
        np.dtype(np.uint32): "DT_UINT32",
        np.dtype(np.uint64): "DT_UINT64",
        np.dtype(np.int8): "DT_INT8",
        np.dtype(np.int16): "DT_INT16",
        np.dtype(np.complex64): "DT_COMPLEX64",
        np.dtype(np.complex128): "DT_COMPLEX128",
        np.dtype(np.bool): "DT_BOOL",
    }

    try:
        from tensorflow import __version__ as tf_version

        if version.parse(tf_version) >= version.parse("2.0.0"):
            _NP_DATATYPE_TO_PROTO_DATATYPE.update(
                {
                    np.dtype([("qint8", "i1")]): "DT_QINT8",
                    np.dtype([("quint8", "u1")]): "DT_QUINT8",
                    np.dtype([("qint16", "<i2")]): "DT_QINT16",
                    np.dtype([("quint16", "<u2")]): "DT_UINT16",
                    np.dtype([("qint32", "<i4")]): "DT_INT32",
                }
            )
            from tensorflow.python import _pywrap_bfloat16

            # TF 2.x.x Implements a Custom Numpy Datatype for Brain Floating Type
            _np_bfloat16 = _pywrap_bfloat16.TF_bfloat16_type()
            _NP_DATATYPE_TO_PROTO_DATATYPE.update({np.dtype(_np_bfloat16): "DT_BFLOAT16"})
    except (ModuleNotFoundError, ValueError, ImportError):
        pass
    if hasattr(npdtype, "kind"):
        if npdtype.kind == "U" or npdtype.kind == "O" or npdtype.kind == "S":
            return (False, "DT_STRING")
    try:
        return (True, _NP_DATATYPE_TO_PROTO_DATATYPE[npdtype])
    except KeyError:
        raise TypeError(f"Numpy Datatype: {np.dtype(npdtype)} is currently not supported")


def make_tensor_proto(nparray_data, tag):
    (isnum, dtype) = _get_proto_dtype(nparray_data.dtype)
    dimensions = [
        TensorShapeProto.Dim(size=d, name="{0}_{1}".format(tag, d)) for d in nparray_data.shape
    ]
    tps = TensorShapeProto(dim=dimensions)
    if isnum:
        tensor_proto = TensorProto(
            dtype=dtype, tensor_content=nparray_data.tostring(), tensor_shape=tps
        )
    else:
        tensor_proto = TensorProto(tensor_shape=tps)
        for s in nparray_data:
            sb = bytes(s, encoding="utf-8")
            tensor_proto.string_val.append(sb)
    return tensor_proto
