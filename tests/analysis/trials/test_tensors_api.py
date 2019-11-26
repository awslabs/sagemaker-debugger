# Standard Library
import socket
from datetime import datetime

# Third Party
import numpy as np
import pytest
from tests.analysis.utils import generate_data

# First Party
from smdebug import modes
from smdebug.core.collection_manager import CollectionManager
from smdebug.core.config_constants import DEFAULT_COLLECTIONS_FILE_NAME
from smdebug.core.writer import FileWriter
from smdebug.trials import create_trial


def test_tensors(out_dir):
    num_steps = 20
    num_tensors = 10
    for i in range(num_steps):
        generate_data(
            path=out_dir,
            trial="test",
            num_tensors=num_tensors,
            step=i,
            tname_prefix="foo",
            worker="algo-1",
            shape=(3, 3, 3),
        )
    for i in range(num_steps, num_steps * 2):
        generate_data(
            path=out_dir,
            trial="test",
            num_tensors=num_tensors,
            step=i,
            tname_prefix="boo",
            worker="algo-1",
            shape=(3, 3, 3),
        )
    tr = create_trial(out_dir + "/test")
    tr.collection("test").include("foo")
    tr.collection("test").add_tensor_name("boo_5")
    tr.collection("test").add_tensor_name("boo_6")
    tr.collection("test").add_tensor_name("boo_17")  # missing tensor
    print(tr.tensor_names())
    assert len(tr.tensor_names()) == num_tensors * 2
    assert len(tr.tensor_names(regex="foo")) == num_tensors
    assert len(tr.tensor_names(collection="test")) == num_tensors + 2
    assert len(tr.tensor_names(collection=tr.collection("test"))) == num_tensors + 2

    with pytest.raises(ValueError):
        tr.tensors(collection=tr.collection("test"), regex="a")


def test_mode_data():
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    trial_dir = "ts_outputs/" + run_id

    c = CollectionManager()
    c.add("default")
    c.get("default").tensor_names = ["arr_1"]
    c.get("default").tensor_names = ["arr_2"]
    c.export(trial_dir, DEFAULT_COLLECTIONS_FILE_NAME)
    trial = create_trial(trial_dir)
    worker = socket.gethostname()
    for s in range(0, 10):
        fw = FileWriter(trial_dir=trial_dir, step=s, worker=worker)
        if s % 2 == 0:
            fw.write_tensor(
                tdata=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                tname="arr_1",
                mode=modes.TRAIN,
                mode_step=s // 2,
            )
        else:
            fw.write_tensor(
                tdata=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                tname="arr_2",
                mode=modes.EVAL,
                mode_step=s // 2,
            )
        fw.close()

    assert trial.tensor_names() == ["arr_1", "arr_2"]
    assert trial.tensor_names(step=0) == ["arr_1"]
    assert trial.tensor_names(step=1) == ["arr_2"]
    assert trial.tensor_names(step=0, mode=modes.TRAIN) == ["arr_1"]
    assert trial.tensor_names(step=0, mode=modes.EVAL) == ["arr_2"]

    assert trial.tensor_names(mode=modes.TRAIN) == ["arr_1"]
    assert trial.tensor_names(mode=modes.EVAL) == ["arr_2"]
