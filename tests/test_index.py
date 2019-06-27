import tensorflow as tf
from tornasole_core.writer import FileWriter
import numpy as np
import csv
from tornasole_core.reader import FileReader
from tornasole_core.reader import FileReader
from tornasole_core.tfevent.event_file_reader import get_tensor_data


def test_index():
    numpy_tensor = [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                    np.array([[1.0, 2.0, 4.0], [3.0, 4.0, 5.0]], dtype=np.float32)]
    runid = "default"
    step = 0
    writer = FileWriter(logdir=".", trial=runid, step=step, worker='worker_0', verbose=True)
    for i in (0, len(numpy_tensor) - 1):
        n = "tensor" + str(i)
        writer.write_tensor(tdata=numpy_tensor[i], tname=n, step=0)
    writer.flush()
    writer.close()
    import os
    eventfile = [filename for filename in os.listdir('.default/events') if filename.endswith("tfevents")]
    indexfile = [filename for filename in os.listdir('.default/index') if filename.endswith("csv")]

    fo = open(eventfile[0], "rb")

    with open(indexfile[0]) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        i = 0
        for row in csv_reader:
            count = int(row[1])
            fo.seek(count, 0)
            end = int(row[2])
            line = fo.read(end)
            zoo = open("test.txt", "wb")
            zoo.write(line)
            zoo.close()
            testfile_reader = FileReader("test.txt")
            tensor_values = testfile_reader.read_tensors()
            tensor_values = list(tensor_values)  ##values converted from tuples to list
            assert np.allclose(tensor_values[0][2].all(), numpy_tensor[i].all()), "indexwriter not working"
            i = i + 1

    fo.close()
    os.remove(indexfile[0])
    os.remove("test.txt")
    for i in eventfile:
        os.remove(i)
