import os
import json

from tornasole.tensorflow.hook import TornasoleHook
from tornasole.tensorflow.utils import (
    TFDistributionStrategy,
    get_worker_id_from_tf_config,
    get_num_workers_from_tf_config,
)


def test_read_tf_config():
    # Case 1: No TF_CONFIG
    distibution_strategy = TornasoleHook.get_distribution_strategy()
    assert distibution_strategy == TFDistributionStrategy.NONE

    # Case 2: TF_CONFIG present but empty
    os.environ["TF_CONFIG"] = json.dumps({})

    distibution_strategy = TornasoleHook.get_distribution_strategy()
    assert distibution_strategy == TFDistributionStrategy.NONE

    # Case 2: TF_CONFIG present but invalid because of missing ps field
    os.environ["TF_CONFIG"] = json.dumps(
        {
            "cluster": {"worker": ["host1:port", "host2:port", "host3:port"]},
            "task": {"type": "worker", "index": 1},
        }
    )

    distibution_strategy = TornasoleHook.get_distribution_strategy()
    assert distibution_strategy == TFDistributionStrategy.NONE

    # Case 2: TF_CONFIG present and valid
    os.environ["TF_CONFIG"] = json.dumps(
        {
            "cluster": {
                "worker": ["host1:port", "host2:port", "host3:port"],
                "ps": ["host4:port", "host5:port"],
            },
            "task": {"type": "worker", "index": 1},
        }
    )

    distibution_strategy = TornasoleHook.get_distribution_strategy()
    assert distibution_strategy == TFDistributionStrategy.PARAMETER_SERVER_STRATEGY

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
