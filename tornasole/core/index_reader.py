import numpy as np
import os
import json
from bisect import bisect_left
from typing import Any, Dict, List, Tuple
from tornasole.core.locations import TensorLocation, IndexFileLocationUtils
from tornasole.core.s3_utils import list_s3_objects
from tornasole.core.access_layer.s3handler import ReadObjectRequest, S3Handler
from tornasole.core.config_constants import TORNASOLE_CONFIG_MAX_WAIT_STEPS
from tornasole.core.utils import (
    is_s3,
    list_files_in_directory,
    step_in_range,
    parse_worker_name_from_file,
)
from tornasole.core.logger import get_logger
from tornasole.core.tfrecord.tensor_reader import TensorReader
from tornasole.core.modes import ModeKeys
from tornasole.exceptions import IndexReaderException, TensorUnavailable, TensorUnavailableForStep


"""

The IndexReader class is responsible for reading index data
from both S3 and Local sources.

It currently exposes two functions:

- fetch_tensor_value : a static function that returns the tensor value given a Tensorlocation object

- load_tensor_data_from_index_files : a function that returns a dictionary populated with data from index files


"""

logger = get_logger()


class IndexFilesCache:
    """
    Simple lookup cache to prevent repeated reads of index files.
    The file cache currently empties itself when it has saved the
    max number of filenames it can save
    """

    def __init__(self):
        self.lookup_set = set()
        self.cache_limit = TORNASOLE_CONFIG_MAX_WAIT_STEPS

    def has_not_read(self, index_file: str) -> bool:
        return index_file not in self.lookup_set

    def add(self, index_file: str) -> None:
        if len(self.lookup_set) == self.cache_limit:
            self.lookup_set.pop()
        self.lookup_set.add(index_file)


index_file_cache = IndexFilesCache()


class S3IndexReader:
    @staticmethod
    def get_s3_responses(bucket_name, prefix_name, start_after_key, range_steps=None):
        object_requests = []
        steps = []
        workers = []
        index_files, start_after_key = S3IndexReader.list_all_index_files_from_s3(
            bucket_name, prefix_name, start_after_key
        )
        logger.debug(f'Loaded Index Files: {",".join(index_files)}')
        for index_file in index_files:
            if index_file_cache.has_not_read(index_file):
                step = IndexFileLocationUtils.parse_step_from_index_file_name(index_file)
                if (
                    range_steps is not None and step_in_range(range_steps, step)
                ) or range_steps is None:
                    steps.append(step)
                    workers.append(parse_worker_name_from_file(index_file))
                    object_requests.append(
                        ReadObjectRequest(format(f"s3://{bucket_name}/") + index_file)
                    )
                index_file_cache.add(index_file)

        responses = S3Handler().get_objects(object_requests)
        return responses, steps, start_after_key, workers

    @staticmethod
    def list_all_index_files_from_s3(bucket_name, prefix_name, start_after_key=None):
        index_files, last_index_token = list_s3_objects(
            bucket_name, IndexFileLocationUtils.get_index_path(prefix_name), start_after_key
        )

        return index_files, last_index_token


class LocalIndexReader:
    @staticmethod
    def list_index_files_in_dir(dirname):
        index_dirname = IndexFileLocationUtils.get_index_path(dirname)
        index_files = list_files_in_directory(index_dirname)
        return sorted(index_files)

    @staticmethod
    def get_disk_responses(
        path, start_after_key, range_steps=None
    ) -> Tuple[List[bytes], List[int], str, List[int]]:
        """Read files like `trial_{datetime}/index/000/{step}_{worker}.json.

        Returns:
            responses: List of the contents of each file, encoded as bytes.
            steps: List of steps read.
            start_after_key: An int referring where to start reading next time.
        """
        index_files = LocalIndexReader.list_index_files_in_dir(path)
        steps = []
        workers = []
        responses = []
        if start_after_key is not None:
            start_after_index = bisect_left(index_files, start_after_key)
        else:
            start_after_index = 0
        index_files = index_files[start_after_index:]  # ignore files we have already read
        for index_file in index_files:
            if index_file_cache.has_not_read(index_file):
                step = IndexFileLocationUtils.parse_step_from_index_file_name(index_file)
                if (
                    range_steps is not None and step_in_range(range_steps, step)
                ) or range_steps is None:
                    steps.append(step)
                    workers.append(parse_worker_name_from_file(index_file))
                    with open(index_file) as f:
                        responses.append(f.read().encode())
                index_file_cache.add(index_file)
        if len(index_files) > 0:
            start_after_key = index_files[-1]  # Last file that we have read
        return responses, steps, start_after_key, workers


class IndexReader:
    @staticmethod
    def fetch_tensor_value(tensor_location: TensorLocation) -> np.ndarray:
        event_file_name = tensor_location.event_file_name
        start = tensor_location.start_idx
        length = tensor_location.length
        s3, bucket_name, prefix_name = is_s3(event_file_name)
        res = []
        num_retries = 5
        if s3:
            while not bool(res) and num_retries > 0:
                request = [ReadObjectRequest(event_file_name, int(start), int(length))]
                s3_handler = S3Handler()
                res = s3_handler.get_objects(request)
                num_retries -= 1
        else:
            tensor_object = None
            while not bool(tensor_object) and num_retries > 0:
                try:
                    with open(event_file_name, "rb") as event_file:
                        event_file.seek(start)
                        tensor_object = event_file.read(length)
                except EnvironmentError:  # parent of IOError, OSError
                    num_retries -= 1
            res = [tensor_object]
        if res[0] is None:
            raise TensorUnavailableForStep(
                tname=tensor_location.tensorname,
                mode=tensor_location.mode,
                step=tensor_location.mode_step,
            )
        tr = TensorReader(res[0])  # Access the only element in res
        tensor_tuple = list(tr.read_tensors())[0]  # Access the only element in the list
        tensor_name, step, tensor_data, mode, mode_step = tensor_tuple
        return tensor_data

    @staticmethod
    def load_tensor_data_from_index_files(
        path, start_after_key=None, range_steps=None
    ) -> Tuple[Dict[str, Dict[int, Dict[str, TensorLocation]]], int]:
        """Return a triply nested dict referring to tensor data."""
        s3, bucket_name, prefix_name = is_s3(path)
        if s3:
            if start_after_key == 0:
                start_after_key = None
            responses, steps, last_index_token, workers = S3IndexReader.get_s3_responses(
                bucket_name, prefix_name, start_after_key, range_steps
            )
        else:
            responses, steps, last_index_token, workers = LocalIndexReader.get_disk_responses(
                path, start_after_key, range_steps
            )
        tensor_data = {}
        for step, response, worker in zip(steps, responses, workers):
            tensor_data = IndexReader._update_tensors_from_json(
                tensor_data, step, response, path, worker
            )
        return tensor_data, last_index_token

    @staticmethod
    def _validate(index_dict):
        if "meta" not in index_dict or len(index_dict["meta"]) == 0:
            raise IndexReaderException("meta section is not present")
        if "tensor_payload" not in index_dict:
            raise IndexReaderException("tensor_payload section is not present")

    @staticmethod
    def _update_tensors_from_json(
        index_tensors_dict, step, response: bytes, path, worker
    ) -> Dict[str, Dict[int, Dict[str, TensorLocation]]]:
        """Return a triply nested dict referring to tensor data.

        Example:
        {
            'dense/bias:0': {
                0: {
                    'tensor_location': <TensorLocation object>
                },
                2: { ... },
                ...
            },
            'conv2d/kernel:0': { ... },
            ...
        }
        """
        index_dict = json.loads(response)
        IndexReader._validate(index_dict)
        index_meta = index_dict["meta"]
        mode = index_meta["mode"]
        mode = ModeKeys[mode.strip()]
        mode_step = index_meta["mode_step"]
        event_file_name = os.path.join(path, index_meta["event_file_name"])
        tensors = index_dict["tensor_payload"]
        for tensor in tensors:
            tensor_name = tensor["tensorname"]
            start_idx = tensor["start_idx"]
            length = tensor["length"]
            tensor_location = TensorLocation(
                tensor_name, mode, mode_step, event_file_name, start_idx, length, worker
            )
            if tensor_name in index_tensors_dict:
                if step in index_tensors_dict[tensor_name]:
                    index_tensors_dict[tensor_name][step].update(
                        {worker: {"tensor_location": tensor_location}}
                    )
                else:
                    index_tensors_dict[tensor_name].update(
                        {step: {worker: {"tensor_location": tensor_location}}}
                    )
            else:
                index_tensors_dict[tensor_name] = {
                    step: {worker: {"tensor_location": tensor_location}}
                }
        return index_tensors_dict
