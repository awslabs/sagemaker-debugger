import numpy as np
from tornasole_numpy.writer import FileWriter
from tornasole_numpy.reader import FileReader

def test_basic():
    with FileWriter(logdir='./ts_output/') as fw:
        fname = fw.name()
        print( f'Saving data in {fname}')
        for i in range(10):
            data = np.ones(shape=(4,4), dtype=np.float32)*i
            fw.add_tensor(data, trial='t', step=0, tensor=f'foo_{i}', worker='worker_1')

    fr = FileReader(fname=fname)
    while True:
        ts = fr.read_tensors()
        print(ts)
        if ts is None:
            break


    assert False
    pass