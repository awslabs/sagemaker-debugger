import pytest
import numpy as np
from tornasole_core.access_layer.s3handler import *
from tornasole_core.tfrecord.tensor_reader import *
######## HELPER CLASSES AND FUNCTIONS #######
class TensorLocation:
    def __init__(self, event_file_name, start=0, length=None):
        self.event_file_name = event_file_name
        self.start = start
        self.length = None

class Index():
    def __init__(self):
        self.dummy = dict()
        self.dummy["s3://ljain-tests/tfevents"] = dict()
        for i in range(5000):
            self.dummy["s3://ljain-tests/tfevents"]["demo_" + str(i)] = [(0, TensorLocation("s3://ljain-tests/tfevents/demo_"+str(i)+".out.tfevents"))]

    # input to get_index_for_tensors is a dict {path:{tensornames:[step_nums]}}
    # output of that fn is dict {path:{tname:[(step_num, TensorLocation)]}}
    def get_index_for_tensors(self, t_dict):
        dict_to_return = dict()
        for key in t_dict.keys():
            dict_to_return[key] = dict()
            for tname in t_dict[key]:
                dict_to_return[key][tname] = self.dummy[key][tname]
        return dict_to_return

def load_index():
    return Index()

######### HELPER FUNCTIONS ######

def read_tensor_from_record(data):
    event_str = read_record(data)
    event = Event()
    event.ParseFromString(event_str)
    assert event.HasField('summary')
    summ = event.summary
    tensors = []
    for v in summ.value:
        tensor_name = v.tag
        tensor_data = get_tensor_data(v.tensor)
        tensors += [tensor_data]
    return tensors

def read_record(data, check=True):
    payload = None
    strlen_bytes = data[:8]
    data = data[8:]
    # will give you payload for the record, which is essentially the event.
    strlen = struct.unpack('Q', strlen_bytes)[0]
    saved_len_crc = struct.unpack('I', data[:4])[0]
    data = data[4:]
    payload = data[:strlen]
    data = data[strlen:]
    saved_payload_crc = struct.unpack('I', data[:4])[0]
    return payload

##########################################

# tlist should be a list of [(tname, [steps])]. This method will return a 
# dictionary with key = (tname, step) and value being the corresponding tensor.
# If the corresponding tensor is not fetchable, then None is stored for its dictionary entry.
def get_tensors(index, s3_handler, tlist, num_async_calls=500, timer=False):
    object_requests = []
    bucket = "ljain-tests"
    prefix = "tfevents"
    index_dict = dict()
    parent_path = "s3://" + bucket + "/" + prefix
    for name, steps in tlist:
        index_dict[name] = steps
    key_lst = []
    t_index = index.get_index_for_tensors({parent_path:index_dict})
    for tname in t_index[parent_path].keys():
        for step, tloc in t_index[parent_path][tname]:
            path, start, length = tloc.event_file_name, tloc.start, tloc.length
            req = ReadObjectRequest(path, start, length)
            object_requests += [req]
            key_lst += [(tname, step)]
    raw_data = s3_handler.get_objects(object_requests, num_async_calls, timer=timer)
    tensors = dict()
    for i in range(len(raw_data)):
        data = raw_data[i]
        assert data is not None
        tensors += [data]
        # tensors[key_lst[i]] = None
        # continue
        # as read_tensor_from_record returns a list containing tensors and as this file only has one tensor
        # take the first/only element of the list
        # tensors[key_lst[i]] = list(TensorReader(data).read_tensors())[0]
    return tensors

##########################################################
## Tests that downloads of objects from S3 handler are working correctly
## Downloads and checks values of 100 numpy tensors asynchronously from the S3 bucket ljain-tests
@pytest.mark.skip("No bucket access")
def test_download_objects(compare_speeds = False):
    # s3trial = S3Trial('test', 'ljain-tests', 'demo')
    index = load_index()
    s3_handler = S3Handler()
    tlist = [("demo_" + str(i), 0) for i in range(100)]
    print("Async...")
    tensors = get_tensors(index, s3_handler, tlist, timer=True)
    assert len(tensors.keys()) == 100
    for tup in tensors.keys(): 
        tensor = tensors[tup]
        assert tensor.shape == (300, 300, 2)
        assert not np.any(np.ones((300,300,2)) - tensor)
    if compare_speeds:
        print("Synchronous...")
        tensors = get_tensors(index, s3_handler, tlist, num_async_calls = 1, timer=True)
    s3_handler.close_client()
##########################################################
## Tests that listing of objects from S3 handler are working correctly
## Lists files from 4 different directories
## Also tests the StartAfter functionality and the delimiter and prefix functionality
@pytest.mark.skip("No bucket access")
def test_list_objects():
    # s3trial = S3Trial('test', 'ljain-tests', 'demo')
    s3_handler = S3Handler()
    req1 = ListRequest('ljain-tests', 'tfevents', '', '')
    req2 = ListRequest('ljain-tests', 'rand_4mb_1000', '', '')
    req3 = ListRequest('ljain-tests', 'rand_8mb_1000', '', '')
    req4 = ListRequest('ljain-tests', 'demo_dir_structure/attempts/', '/')
    req5 = ListRequest('ljain-tests', 'demo_dir_structure/attempts/', '/', 'demo_dir_structure/attempts/help')
    files = s3_handler.list_prefixes([req1, req2, req3, req4, req5])
    # test StartAfter and delimiters
    assert len(files[3]) == 5 and len(files[4]) == 3
    assert len(files[0]) == 5000
    assert len(files[1]) == 1001 
    assert len(files[2]) == 1001
    s3_handler.close_client()
