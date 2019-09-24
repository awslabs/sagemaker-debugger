from tornasole.core.writer import FileWriter
import numpy as np
from tornasole.core.collection_manager import CollectionManager, \
    COLLECTIONS_FILE_NAME
import os


def generate_data(path, trial, step, tname_prefix,
                  num_tensors, worker, shape, dtype=np.float32,
                  rank=None, mode=None, mode_step=None, export_colls=True,
                  data=None):
    with FileWriter(trial_dir=os.path.join(path, trial),
                    step=step, worker=worker) as fw:
        for i in range(num_tensors):
            if data is None:
                data = np.ones(shape=shape, dtype=dtype) * step
            fw.write_tensor(tdata=data, tname=f'{tname_prefix}_{i}',
                            mode=mode, mode_step=mode_step)
    if export_colls:
        c = CollectionManager()
        c.add("default")
        c.get("default").tensor_names = [f'{tname_prefix}_{i}' for i in range(num_tensors)]
        c.add('gradients')
        c.get("gradients").tensor_names = [f'{tname_prefix}_{i}' for i in range(num_tensors)]
        c.export(os.path.join(path, trial, COLLECTIONS_FILE_NAME))


def check_trial(trial_obj, num_steps, num_tensors):
    assert len(trial_obj.tensors()) == num_tensors
    for t in trial_obj.tensors():
        assert len(trial_obj.tensor(t).steps()) == num_steps
        for s in trial_obj.tensor(t).steps():
            v = trial_obj.tensor(t).value(s)
            assert v is not None
