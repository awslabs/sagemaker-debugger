import numpy as np
from tornasole_numpy.writer import FileWriter
from tornasole_numpy.reader import FileReader

def test_basic():
    """
    Checks that we can save data and read it back the way it was
    """
    with FileWriter(logdir='./ts_output/') as fw:
        fname = fw.name()
        print( f'Saving data in {fname}')
        for i in range(10):
            data = np.ones(shape=(4,4), dtype=np.float32)*i
            fw.write_tensor(data, trial='t', step=0, tensor=f'foo_{i}', worker='worker_1')

    fr = FileReader(fname=fname)
    for i,ts in enumerate(fr.read_tensors()):
        print(i,ts)
        assert np.all(ts[1]==i)
    pass