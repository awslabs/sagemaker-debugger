# Standard Library
import json
import os
import time
from abc import ABC, abstractmethod
from bisect import bisect_left
from typing import Dict, List, Tuple

# Third Party
import numpy as np

# First Party
from smdebug.core.access_layer.s3handler import ReadObjectRequest, S3Handler
from smdebug.core.access_layer.utils import has_training_ended
from smdebug.core.config_constants import (
    MISSING_EVENT_FILE_RETRY_LIMIT,
    MISSING_EVENT_FILE_RETRY_LIMIT_KEY,
)
from smdebug.core.locations import IndexFileLocationUtils, TensorLocation, TensorShape
from smdebug.core.logger import get_logger
from smdebug.core.modes import ModeKeys
from smdebug.core.s3_utils import list_s3_objects
from smdebug.core.tfrecord.tensor_reader import TensorReader
from smdebug.core.utils import (
    get_path_to_events_directory,
    is_s3,
    list_files_in_directory,
    parse_worker_name_from_file,
    step_in_range,
)
from smdebug.exceptions import IndexReaderException, TensorUnavailableForStep


class ReadIndexFilesCache:
    """
    Simple lookup cache to prevent repeated reads of index files.

    The cache stores the names of the index files we have already read
    to prevent repeated reads.

    We query for files to be read from either disk or s3.

    The query window is lexically ordered, beginning with a start_after_key.

    The penalty of a cache miss, is a repeated read of of an index file.
    This would mean an additional network call or disk access.

    The cache limit has currently been arbitrarily set to 1000 to prevent the cache from
    growing too large.

    Eviction Strategy: If the cache limit has been hit, we remove all the elements
    that are lexical predecessors of the start_after_prefix, since, we will not be attempting
    to read these files again.

    Note: cache_limit is a soft limit.

    The size of the cache can exceed this limit if eviction_point is 0.
    This can happen if start_after_key is None (unset) or is equal to the first element in the cache.

    If start_after_key happens to the be first element in sorted(self.lookup_set), then we do not
    evict any element from the cache, but add more elements to cache.

    The start_after_key will eventually move forward, if we see a complete step that is greater than the
    step marked by the start_after_key, or we breach the INCOMPLETE_STEP_WAIT_WINDOW, that
    changes the start_after_key if we are waiting for too many steps.

    Elements with a lower lexical rank than the start_after_key are guaranteed to be never read again.

    """

    def __init__(self):
        self.lookup_set = set()
        self.cache_limit = 1000

    def has_not_read(self, index_file: str) -> bool:
        return index_file not in self.lookup_set

    def _evict_cache(self, start_after_key: str) -> None:
        read_files = sorted(self.lookup_set)
        start_after_key = "" if start_after_key is None else start_after_key
        # eviction_point = 0 if start_after_key is None (unset).
        # This happens if more than self.cache_limit number of files are read on the first read attempt.
        eviction_point = bisect_left(read_files, start_after_key)
        for i in range(eviction_point):
            self.lookup_set.remove(read_files[i])

    def add(self, index_file: str, start_after_key: str) -> None:
        if len(self.lookup_set) > self.cache_limit:
            self._evict_cache(start_after_key)
        self.lookup_set.add(index_file)


class IndexReader(ABC):
    """

    The IndexReader class is responsible for reading index data
    from both S3 and Local sources.

    It currently exposes two functions:

    - fetch_tensor_value : a static function that returns the tensor value given a Tensorlocation object

    - load_tensor_data_from_index_files : a function that returns a dictionary populated with data from index files


    """

    def __init__(self, path):
        self.event_file_retry_limit = int(
            os.getenv(MISSING_EVENT_FILE_RETRY_LIMIT_KEY, MISSING_EVENT_FILE_RETRY_LIMIT)
        )
        self.path = path
        self.logger = get_logger()

    @abstractmethod
    def fetch_tensor_value(self, tensor_location: TensorLocation):
        pass

    @abstractmethod
    def list_event_files(self, start_after_prefix):
        pass

    def load_tensor_data_from_index_files(
        self, start_after_key=None, range_steps=None
    ) -> Tuple[Dict[str, Dict[int, Dict[str, TensorLocation]]], str]:
        """Return a triply nested dict referring to tensor data."""

        responses, steps, last_index_token, workers = self.read_index_files(
            start_after_key, range_steps
        )

        tensor_data = {}
        for step, response, worker in zip(steps, responses, workers):
            tensor_data = self._update_tensors_from_json(
                tensor_data, step, response, self.path, worker
            )
        return tensor_data, last_index_token

    @abstractmethod
    def _is_event_file_present(self, file_name) -> bool:
        pass

    def event_file_present_loop(self, tensor_location: TensorLocation):
        event_file_name = tensor_location.event_file_name
        event_file_present = self._is_event_file_present(event_file_name)
        num_retry = 0
        while not event_file_present and num_retry < self.event_file_retry_limit:
            if self._has_event_file_been_skipped(event_file_name):
                raise TensorUnavailableForStep(
                    tname=tensor_location.tensorname,
                    mode=tensor_location.mode,
                    step=tensor_location.mode_step,
                )
            elif has_training_ended(self.path) is True:
                self.logger.warn(
                    f"IndexReader: Training Has Ended"
                    f"\nIndexReader: {event_file_name} was written but not found."
                )
                raise TensorUnavailableForStep(
                    tname=tensor_location.tensorname,
                    mode=tensor_location.mode,
                    step=tensor_location.mode_step,
                )
            event_file_present = self._is_event_file_present(event_file_name)
            num_retry += 1
            time.sleep(2)
        if num_retry >= self.event_file_retry_limit:
            self.logger.warn(
                f"IndexReader: {event_file_name} was written but not found. After {num_retry} retries."
            )
            raise TensorUnavailableForStep(
                tname=tensor_location.tensorname,
                mode=tensor_location.mode,
                step=tensor_location.mode_step,
            )
        return

    def _has_event_file_been_skipped(self, missing_event_file_name: str) -> bool:
        """
        Checks if an event file will ever be downloaded.
        if event_file present --> return False
        if the worker has written the next event file --> return True
        if none of the above --> return False
        :param missing_event_file_name:
        :return:
        """
        self.logger.info(f" Index Reader: Event File {missing_event_file_name} not found.")
        missing_worker = parse_worker_name_from_file(missing_event_file_name)
        missing_step = IndexFileLocationUtils.parse_step_from_index_file_name(
            missing_event_file_name
        )
        event_files = self.list_event_files(missing_event_file_name)
        for event_file in event_files:
            if missing_worker == parse_worker_name_from_file(event_file):
                step = IndexFileLocationUtils.parse_step_from_index_file_name(event_file)
                if missing_step == step:
                    """
                        The missing step file may have been written to disk before
                        we perform the list operation.
                    """
                    return False
                self.logger.warn(
                    f" Index Reader: Event File {missing_event_file_name} was written but not found "
                    f"\nHowever Event File {event_file} found."
                )
                self.logger.warn(f"IndexReader: Skipping {missing_event_file_name} ")
                return True
        return False

    @staticmethod
    def _validate(index_dict):
        if "meta" not in index_dict:
            raise IndexReaderException("meta section is not present")
        if len(index_dict["meta"]) == 0:
            raise IndexReaderException("meta section is empty")
        if "tensor_payload" not in index_dict and "shape_payload" not in index_dict:
            raise IndexReaderException(
                "neither tensor_payload nor shape_payload sections are present"
            )

    def _update_tensors_from_json(
        self, index_tensors_dict, step, response: bytes, path, worker
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
        try:
            index_dict = json.loads(response)
        except ValueError:
            raise IndexReaderException("Empty/Corrupt Index File")
        IndexReader._validate(index_dict)
        index_meta = index_dict["meta"]
        mode = index_meta["mode"]
        mode = ModeKeys[mode.strip()]
        mode_step = index_meta["mode_step"]

        to_update_index_dict = []

        if "tensor_payload" in index_dict and len(index_dict["tensor_payload"]):
            event_file_name = os.path.join(path, index_meta["event_file_name"])
            for tensor in index_dict["tensor_payload"]:
                tensor_name = tensor["tensorname"]
                start_idx = tensor["start_idx"]
                length = tensor["length"]
                tensor_location = TensorLocation(
                    tensor_name, mode, mode_step, event_file_name, start_idx, length, worker
                )
                to_update_index_dict.append((tensor_name, step, tensor_location))

        if "shape_payload" in index_dict and len(index_dict["shape_payload"]):
            for tensor in index_dict["shape_payload"]:
                tensor_name = tensor["tensorname"]
                original_name = tensor["originalname"]
                shape = tensor["shape"]
                ts = TensorShape(tensor_name, mode, mode_step, shape, original_name)
                to_update_index_dict.append((tensor_name, step, ts))

        for tu in to_update_index_dict:
            tensor_name, step, obj = tu
            if isinstance(obj, TensorLocation):
                obj_dict = {"tensor_location": obj}
            elif isinstance(obj, TensorShape):
                obj_dict = {"tensor_shape": obj}
            if tensor_name in index_tensors_dict:
                if step in index_tensors_dict[tensor_name]:
                    index_tensors_dict[tensor_name][step].update({worker: obj_dict})
                else:
                    index_tensors_dict[tensor_name].update({step: {worker: obj_dict}})
            else:
                index_tensors_dict[tensor_name] = {step: {worker: obj_dict}}
        return index_tensors_dict


class S3IndexReader(IndexReader):
    def __init__(self, path):
        super().__init__(path)
        self.path = path
        _, self.bucket_name, self.prefix_name = is_s3(path)
        self.index_file_cache = ReadIndexFilesCache()

    def _is_event_file_present(self, file):
        event_files = self.list_event_files()
        _, _, prefix = is_s3(file)
        return prefix in set(event_files)

    def fetch_tensor_value(self, tensor_location: TensorLocation) -> np.ndarray:
        event_file_name = tensor_location.event_file_name

        if not self._is_event_file_present(event_file_name):
            self.event_file_present_loop(tensor_location)

        start = tensor_location.start_idx
        length = tensor_location.length
        request = [ReadObjectRequest(event_file_name, int(start), int(length))]
        res = S3Handler.get_objects(request)
        tr = TensorReader(res[0])  # Access the only element in res
        tensor_tuple = list(tr.read_tensors())[0]  # Access the only element in the list
        tensor_name, step, tensor_data, mode, mode_step = tensor_tuple
        return tensor_data

    def read_index_files(
        self, start_after_key: str, range_steps=None
    ) -> Tuple[List[bytes], list, str, List[str]]:
        """
            Read files like `trial_{datetime}/index/000/{step}_{worker}.json.
        :param start_after_key: str
        :param range_steps:
        :return: Tuple( responses, steps, start_after_key, workers)
        """
        object_requests = []
        steps = []
        workers = []
        index_files, start_after_key = self.list_index_files(start_after_key)
        self.logger.debug(f'Loaded Index Files: {",".join(index_files)}')
        for index_file in index_files:
            if self.index_file_cache.has_not_read(index_file):

                step = IndexFileLocationUtils.parse_step_from_index_file_name(index_file)
                if (
                    range_steps is not None and step_in_range(range_steps, step)
                ) or range_steps is None:
                    steps.append(step)
                    workers.append(parse_worker_name_from_file(index_file))
                    object_requests.append(
                        ReadObjectRequest(format(f"s3://{self.bucket_name}/") + index_file)
                    )
                    self.logger.debug(f"Will read index_file: {index_file}")
                    self.index_file_cache.add(index_file, start_after_key)
            else:
                self.logger.debug(
                    f"index_file:{index_file} Indexcache contents:{self.index_file_cache.lookup_set}"
                )

        responses = S3Handler.get_objects(object_requests)
        assert len(responses) == len(object_requests)
        return responses, steps, start_after_key, workers

    def list_index_files(self, start_after_key=None):
        index_files, last_index_token = list_s3_objects(
            self.bucket_name,
            IndexFileLocationUtils.get_index_path(self.prefix_name),
            start_after_key,
        )

        return index_files, last_index_token

    def list_event_files(self, start_after_key=None):
        event_files, last_index_token = list_s3_objects(
            self.bucket_name, get_path_to_events_directory(self.prefix_name), start_after_key
        )
        return event_files


class LocalIndexReader(IndexReader):
    def __init__(self, path):
        super().__init__(path)
        self.index_file_cache = ReadIndexFilesCache()
        self.path = path

    def _is_event_file_present(self, file):
        return os.path.exists(file)

    def list_index_files(self):
        index_dirname = IndexFileLocationUtils.get_index_path(self.path)
        # index files are json files or csv files ending with string ".csv" or ".json"
        index_files_regex = r"(.+)\.(json|csv)$"
        index_files = list_files_in_directory(index_dirname, file_regex=index_files_regex)
        return sorted(index_files)

    def list_event_files(self, start_after_key=None):
        # event files are ending with string ".tfevents"
        event_file_regex = r"(.+)\.(tfevents)$"
        event_files = list_files_in_directory(
            get_path_to_events_directory(self.path), file_regex=event_file_regex
        )
        event_files.sort()
        start_after_index = bisect_left(event_files, start_after_key)
        return event_files[start_after_index:]

    def fetch_tensor_value(self, tensor_location: TensorLocation) -> np.ndarray:
        event_file_name = tensor_location.event_file_name

        if not self._is_event_file_present(event_file_name):
            self.event_file_present_loop(tensor_location)

        start = tensor_location.start_idx
        length = tensor_location.length

        with open(event_file_name, "rb") as event_file:
            event_file.seek(start)
            tensor_object = event_file.read(length)

        tr = TensorReader(tensor_object)
        tensor_tuple = list(tr.read_tensors())[0]  # Access the only element in the list
        tensor_name, step, tensor_data, mode, mode_step = tensor_tuple
        return tensor_data

    def read_index_files(
        self, start_after_key: str, range_steps=None
    ) -> Tuple[List[bytes], list, str, List[str]]:
        """
            Read files like `trial_{datetime}/index/000/{step}_{worker}.json.
        :param start_after_key: str
        :param range_steps: str
        :return: Tuple( responses, steps, start_after_key, workers)
        """
        index_files = self.list_index_files()
        steps = []
        workers = []
        responses = []
        if start_after_key is not None:
            start_after_index = bisect_left(index_files, start_after_key)
        else:
            start_after_index = 0
        self.logger.debug(f"Found index_files:{index_files}")
        index_files = index_files[start_after_index:]  # ignore files we have already read
        self.logger.debug(
            f"Curtailed Found index_files to :{index_files} start_after_index:{start_after_index} start_after_key:{start_after_key}"
        )
        for index_file in index_files:
            if self.index_file_cache.has_not_read(index_file):
                step = IndexFileLocationUtils.parse_step_from_index_file_name(index_file)
                if (
                    range_steps is not None and step_in_range(range_steps, step)
                ) or range_steps is None:
                    steps.append(step)
                    workers.append(parse_worker_name_from_file(index_file))
                    self.logger.debug(
                        f"Sagemaker-Debugger: Read {os.path.getsize(index_file)} bytes from file {index_file}"
                    )
                    self.logger.debug(f"Will read index file:{index_file}")
                    with open(index_file) as f:
                        responses.append(f.read().encode())
                    self.index_file_cache.add(index_file, start_after_key)
            else:
                self.logger.debug(
                    f"IndexFile:{index_file} Indexcache contents:{self.index_file_cache.lookup_set}"
                )

        if len(index_files) > 0:
            start_after_key = index_files[-1]  # Last file that we have read
        return responses, steps, start_after_key, workers
