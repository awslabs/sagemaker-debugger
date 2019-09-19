import os
import json
from tornasole.core.locations import TensorLocation, IndexFileLocationUtils
from tornasole.core.s3_utils import list_s3_objects
from tornasole.core.access_layer.s3handler import ReadObjectRequest, S3Handler
from tornasole.core.utils import is_s3, list_files_in_directory, step_in_range
from tornasole.core.logger import get_logger
from tornasole.core.tfrecord.tensor_reader import TensorReader
from tornasole.core.modes import ModeKeys


"""

The IndexReader class is responsible for reading index data
from both S3 and Local sources. 

It currently exposes two functions:

- fetch_tensor_value : a static function that returns the tensor value given a Tensorlocation object

- load_tensor_data_from_index_files : a function that returns a dictionary populated with data from index files


"""

logger = get_logger()


class S3IndexReader:
     
    @staticmethod
    def get_s3_responses(bucket_name, prefix_name, start_after_key, range_steps=None):
        object_requests = []
        steps = []
        index_files, last_index_token = S3IndexReader.list_all_index_files_from_s3(bucket_name, prefix_name,
                                                                                  start_after_key)
        for index_file in index_files:
            step = IndexFileLocationUtils.parse_step_from_index_file_name(index_file)
            if (range_steps is not None and step_in_range(range_steps, step)) or \
                    range_steps is None:
                steps.append(step)
                object_requests.append(ReadObjectRequest(format(f"s3://{bucket_name}/") + index_file))

        responses = S3Handler().get_objects(object_requests)
        return responses, steps, last_index_token

    @staticmethod
    def list_all_index_files_from_s3(bucket_name, prefix_name, start_after_key=None):
        index_files, last_index_token = list_s3_objects(bucket_name,
                                                        IndexFileLocationUtils.get_index_path(prefix_name),
                                                        start_after_key)

        return index_files, last_index_token
    

class LocalIndexReader:
    
    @staticmethod
    def list_index_files_in_dir(dirname):
        index_dirname = IndexFileLocationUtils.get_index_path(dirname)
        index_files = list_files_in_directory(index_dirname)
        return sorted(index_files)

    @staticmethod
    def get_disk_responses(path, start_after_key=0, range_steps=None):
        index_files = LocalIndexReader.list_index_files_in_dir(path)
        steps = []
        responses = []
        index_files = index_files[start_after_key:]  # ignore files we have already read
        for index_file in index_files:
            step = IndexFileLocationUtils.parse_step_from_index_file_name(index_file)
            if (range_steps is not None and step_in_range(range_steps, step)) or \
                    range_steps is None:
                steps.append(IndexFileLocationUtils.parse_step_from_index_file_name(index_file))
                with open(index_file) as f:
                    responses.append(f.read().encode())
        start_after_key += len(index_files)  # Last file that we have read
        return responses, steps, start_after_key


class IndexReader:

    @staticmethod
    def fetch_tensor_value(tensor_location):
        event_file_name = tensor_location.event_file_name
        start = tensor_location.start_idx
        length = tensor_location.length
        s3, bucket_name, prefix_name = is_s3(event_file_name)
        if s3:
            request = [ReadObjectRequest(event_file_name, int(start), int(length))]
            s3_handler = S3Handler()
            res = s3_handler.get_objects(request)
        else:
            with open(event_file_name, 'rb') as event_file:
                event_file.seek(start)
                tensor_object = event_file.read(length)
            res = [tensor_object]

        tr = TensorReader(res[0])  # Access the only element in res
        tensor_tuple = list(tr.read_tensors(read_data=True))[0]  # Access the only element in the list
        tensor_name, step, tensor_data, mode, mode_step = tensor_tuple
        return tensor_data

    @staticmethod
    def load_tensor_data_from_index_files(path, start_after_key=None, range_steps=None):
        s3, bucket_name, prefix_name = is_s3(path)
        if s3:
            if start_after_key == 0:
                start_after_key = None
            responses, steps, last_index_token = \
                S3IndexReader.get_s3_responses(bucket_name, prefix_name, start_after_key, range_steps)
        else:
            responses, steps, last_index_token = LocalIndexReader.get_disk_responses(path, start_after_key, range_steps)
        tensor_data = {}
        for step, response in zip(steps, responses):
            tensor_data = IndexReader._update_tensors_from_json(tensor_data, step, response, path)
        return tensor_data, last_index_token

    @staticmethod
    def _update_tensors_from_json(index_tensors_dict, step, response, path):
        index_dict = json.loads(response)
        index_meta = index_dict['meta']
        mode = index_meta['mode']
        mode = ModeKeys[mode.strip()]
        mode_step = index_meta['mode_step']
        event_file_name = os.path.join(path, index_meta['event_file_name'])
        tensors = index_dict['tensor_payload']
        for tensor in tensors:
            tensor_name = tensor['tensorname']
            start_idx = tensor['start_idx']
            length = tensor['length']
            tensor_location = TensorLocation(tensor_name, mode, mode_step, event_file_name, start_idx, length)
            if tensor_name in index_tensors_dict:
                index_tensors_dict[tensor_name].update({
                    step: {
                        "tensor_location": tensor_location
                    }})
            else:
                index_tensors_dict[tensor_name] = {step: {
                    "tensor_location": tensor_location
                }}

        return index_tensors_dict

