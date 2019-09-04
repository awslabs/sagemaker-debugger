import csv
from tornasole.core.writer import FileWriter
from tornasole.core.tfevent.event_file_writer import *
from tornasole.core.reader import FileReader
from tornasole.core.tfevent.util import EventFileLocation
from tornasole.core.indexutils import *
import shutil
import os

def test_index():
    numpy_tensor = [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                    np.array([[1.0, 2.0, 4.0], [3.0, 4.0, 5.0]], dtype=np.float32)]
    runid = "default"
    logdir = "."
    step = 0
    worker = "worker_0"
    run_dir = os.path.join(logdir,runid)
    writer = FileWriter(trial_dir=run_dir,
                        step=step, worker=worker, verbose=True)
    for i in (0, len(numpy_tensor) - 1):
        n = "tensor" + str(i)
        writer.write_tensor(tdata=numpy_tensor[i], tname=n)
    writer.flush()
    writer.close()
    efl = EventFileLocation(step_num=step, worker_name=worker)
    eventfile = efl.get_location(run_dir=run_dir)
    indexfile = IndexUtil.get_index_key_for_step(run_dir, step,worker)

    fo = open(eventfile, "rb")

    with open(indexfile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        i = 0
        for row in csv_reader:
            count = int(row[-2])
            fo.seek(count, 0)
            end = int(row[-1])
            line = fo.read(end)
            zoo = open("test.txt", "wb")
            zoo.write(line)
            zoo.close()
            testfile_reader = FileReader("./test.txt")
            tensor_values = list(testfile_reader.read_tensors())
            assert np.allclose(tensor_values[0][2].all(), numpy_tensor[i].all()), "indexwriter not working"
            i = i + 1

    fo.close()
    shutil.rmtree(run_dir)
    os.remove("test.txt")
