# Standard Library
import json
import os

# First Party
from smdebug.tensorflow.utils import (
    TFDistributionStrategy,
    get_num_workers_from_tf_config,
    get_worker_id_from_tf_config,
    is_parameter_server_strategy,
    tensor_can_be_saved,
)


def test_read_tf_config():
    # Case 1: No TF_CONFIG

    assert is_parameter_server_strategy(os.getenv("TF_CONFIG")) is False

    # Case 2: TF_CONFIG present but empty
    os.environ["TF_CONFIG"] = json.dumps({})

    assert is_parameter_server_strategy(os.getenv("TF_CONFIG")) is False

    # Case 3: TF_CONFIG present but invalid because of missing ps field
    os.environ["TF_CONFIG"] = json.dumps(
        {
            "cluster": {"worker": ["host1:port", "host2:port", "host3:port"]},
            "task": {"type": "worker", "index": 1},
        }
    )

    assert is_parameter_server_strategy(os.getenv("TF_CONFIG")) is False

    # Case 4: TF_CONFIG present and valid
    os.environ["TF_CONFIG"] = json.dumps(
        {
            "cluster": {
                "worker": ["host1:port", "host2:port", "host3:port"],
                "ps": ["host4:port", "host5:port"],
            },
            "task": {"type": "worker", "index": 1},
        }
    )

    assert is_parameter_server_strategy(os.getenv("TF_CONFIG")) is True

    del os.environ["TF_CONFIG"]


def test_get_worker_id_from_tf_config():
    os.environ["TF_CONFIG"] = json.dumps(
        {
            "cluster": {
                "worker": ["host1:port", "host2:port", "host3:port"],
                "ps": ["host4:port", "host5:port"],
            },
            "task": {"type": "worker", "index": 1},
        }
    )

    worker_id = get_worker_id_from_tf_config(os.getenv("TF_CONFIG"))
    assert worker_id == "worker_1"

    del os.environ["TF_CONFIG"]

    os.environ["TF_CONFIG"] = json.dumps(
        {
            "cluster": {
                "worker": ["host1:port", "host2:port", "host3:port"],
                "ps": ["host4:port", "host5:port"],
            },
            "task": {"type": "ps", "index": 0},
        }
    )

    worker_id = get_worker_id_from_tf_config(os.getenv("TF_CONFIG"))
    assert worker_id == "ps_0"

    del os.environ["TF_CONFIG"]

    os.environ["TF_CONFIG"] = json.dumps(
        {
            "cluster": {
                "chief": ["host0:port"],
                "worker": ["host1:port", "host2:port", "host3:port"],
                "ps": ["host4:port", "host5:port"],
            },
            "task": {"type": "chief", "index": 0},
        }
    )

    worker_id = get_worker_id_from_tf_config(os.getenv("TF_CONFIG"))
    assert worker_id == "chief_0"

    del os.environ["TF_CONFIG"]

    os.environ["TF_CONFIG"] = json.dumps(
        {
            "cluster": {
                "chief": ["host0:port"],
                "worker": ["host1:port", "host2:port", "host3:port"],
                "ps": ["host4:port", "host5:port"],
            },
            "task": {"type": "evaluator", "index": 0},
        }
    )

    worker_id = get_worker_id_from_tf_config(os.getenv("TF_CONFIG"))
    assert worker_id == "evaluator_0"
    del os.environ["TF_CONFIG"]


def test_get_num_workers_from_tf_config():
    os.environ["TF_CONFIG"] = json.dumps(
        {
            "cluster": {
                "worker": ["host1:port", "host2:port", "host3:port"],
                "ps": ["host4:port", "host5:port"],
            },
            "task": {"type": "worker", "index": 1},
        }
    )

    num_workers = get_num_workers_from_tf_config(os.getenv("TF_CONFIG"))
    assert num_workers == 3

    del os.environ["TF_CONFIG"]

    os.environ["TF_CONFIG"] = json.dumps(
        {
            "cluster": {
                "chief": ["host0:port"],
                "worker": ["host1:port", "host2:port", "host3:port"],
                "ps": ["host4:port", "host5:port"],
            },
            "task": {"type": "worker", "index": 1},
        }
    )

    num_workers = get_num_workers_from_tf_config(os.getenv("TF_CONFIG"))
    assert num_workers == 4
    del os.environ["TF_CONFIG"]


class TFOpMock:
    def __init__(self, name, inputs):
        self.inputs = inputs
        self.name = name
        self.op = self


class TFTensorMock:
    def __init__(self, op):
        self.op = op


def test_path_to_subgraph_1():
    """
    	  out1
    in1		    out2
            in2	    in3
    """
    in1 = TFOpMock("in1", [])
    in2 = TFOpMock("in2", [])
    in3 = TFOpMock("in3", [])
    out2 = TFOpMock("out2", [in2, in3])
    out1 = TFOpMock("out1", [in1, out2])
    subgraph_nodes = {in1.name, in2.name, in3.name}
    assert tensor_can_be_saved(out1, subgraph_nodes, set())


def test_path_to_subgraph_2():
    """
    	  out1
    in1		    out2
            in2	    out3
    """
    in1 = TFOpMock("in1", [])
    in2 = TFOpMock("in2", [])
    out3 = TFOpMock("out3", [])
    out2 = TFOpMock("out2", [in2, out3])
    out1 = TFOpMock("out1", [in1, out2])
    subgraph_nodes = {in1.name, in2.name}
    assert tensor_can_be_saved(out1, subgraph_nodes, {out3}) is False
    assert tensor_can_be_saved(out1, subgraph_nodes, {})


def test_path_to_subgraph_3():
    """
    	  out1
       out2	   out3
    out4         out5
    """
    out4 = TFOpMock("out4", [])
    out5 = TFOpMock("out5", [])
    out3 = TFOpMock("out3", [out5])
    out2 = TFOpMock("out2", [out4])
    out1 = TFOpMock("out1", [out2, out3])
    tf_tensor = TFTensorMock(out1)
    subgraph_nodes = {}
    assert tensor_can_be_saved(out1, subgraph_nodes, set())
    assert tensor_can_be_saved(out1, subgraph_nodes, {out5}) is False


def test_path_to_subgraph_4():
    """
    out1
    """
    out1 = TFOpMock("out1", [])
    subgraph_nodes = {}
    assert tensor_can_be_saved(out1, subgraph_nodes, {out1}) is False
    assert tensor_can_be_saved(out1, subgraph_nodes, {})


def test_path_to_subgraph_5():
    """
    in1
    """
    in1 = TFOpMock("in1", [])
    subgraph_nodes = {in1.name}
    assert tensor_can_be_saved(in1, subgraph_nodes, {in1})
    assert tensor_can_be_saved(in1, subgraph_nodes, {})


def test_path_to_subgraph_6():
    """
              in1
        in2        in3
    in5   in4    in6
    """
    in5 = TFOpMock("in5", [])
    in4 = TFOpMock("in4", [])
    in6 = TFOpMock("in6", [])
    in2 = TFOpMock("in2", [in5, in4])
    in3 = TFOpMock("in3", [in6])
    in1 = TFOpMock("in1", [in2, in3])
    subgraph_nodes = {in1.name, in2.name, in3.name, in4.name, in5.name, in6.name}
    assert tensor_can_be_saved(in1, subgraph_nodes, {in2, in3})
    assert tensor_can_be_saved(in1, subgraph_nodes, {in1})
