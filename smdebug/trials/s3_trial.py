# Standard Library
import os

# First Party
from smdebug.core.access_layer.s3handler import ReadObjectRequest, S3Handler
from smdebug.core.collection_manager import CollectionManager
from smdebug.core.index_reader import S3IndexReader
from smdebug.core.s3_utils import list_s3_objects
from smdebug.core.utils import get_path_to_collections

# Local
from .trial import Trial


class S3Trial(Trial):
    def __init__(
        self,
        name,
        bucket_name,
        prefix_name,
        range_steps=None,
        check=False,
        index_mode=True,
        cache=False,
    ):
        """
        :param name: for sagemaker job, this should be sagemaker training job name
        :param bucket_name: name of bucket where data is saved
        :param prefix_name: name of prefix such that s3://bucket/prefix is where data is saved
        :param range_steps: range_steps is a tuple representing (start_step, end_step).
                            Only the data from steps in between this range will be loaded
        :param check: whether to check checksum of data saved
        """
        super().__init__(
            name,
            range_steps=range_steps,
            parallel=False,
            check=check,
            index_mode=index_mode,
            cache=cache,
        )
        self.logger.info(f"Loading trial {name} at path s3://{bucket_name}/{prefix_name}")
        self.bucket_name = bucket_name
        self.prefix_name = os.path.join(prefix_name, "")
        self.path = "s3://" + os.path.join(self.bucket_name, self.prefix_name)
        self.index_reader = S3IndexReader(self.path)
        self._load_collections()
        self._load_tensors()

    def _get_collection_files(self) -> list:
        collection_files, _ = list_s3_objects(
            self.bucket_name,
            get_path_to_collections(self.prefix_name),
            start_after_key=None,
            delimiter="",
        )
        return collection_files

    def _load_tensors_from_index_tensors(self, index_tensors_dict):
        for tname in index_tensors_dict:
            for step, itds in index_tensors_dict[tname].items():
                for worker in itds:
                    self._add_tensor(int(step), worker, itds[worker]["tensor_location"])

    def _read_collections(self, collection_files):
        first_collection_file = collection_files[0]  # First Collection File
        key = os.path.join(first_collection_file)
        collections_req = ReadObjectRequest(self._get_s3_location(key))
        obj_data = S3Handler.get_objects([collections_req])[0]
        obj_data = obj_data.decode("utf-8")
        self.collection_manager = CollectionManager.load_from_string(obj_data)
        self.num_workers = self.collection_manager.get_num_workers()

    def _get_s3_location(self, obj):
        return "s3://" + self.bucket_name + "/" + obj
