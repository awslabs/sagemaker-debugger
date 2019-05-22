import numpy as np
from tornasole_core.writer import FileWriter
from tornasole_core.reader import FileReader

def test_basic():
    """
    Checks that we can save data and read it back the way it was
    """
    with FileWriter(logdir='./ts_output/', trial='my_trial', step=20, worker='algo-1') as fw:
        fname = fw.name()
        print( f'Saving data in {fname}')
        for i in range(10):
            data = np.ones(shape=(4,4), dtype=np.float32)*i
            fw.write_tensor(tdata=data, tname=f'foo_{i}')

    fr = FileReader(fname=fname)
    for i,ts in enumerate(fr.read_tensors()):
        print(i,ts)
        assert np.all(ts[1]==i)
    pass
