# Standard Library
import os

# First Party
from smdebug.core.collection_manager import CollectionManager
from smdebug.core.index_reader import LocalIndexReader
from smdebug.core.utils import get_path_to_collections, list_collection_files_in_directory

# Local
from .trial import Trial


class LocalTrial(Trial):
    def __init__(
        self,
        name,
        dirname,
        range_steps=None,
        parallel=True,
        check=False,
        index_mode=True,
        cache=False,
    ):
        super().__init__(
            name,
            range_steps=range_steps,
            parallel=parallel,
            check=check,
            index_mode=index_mode,
            cache=cache,
        )
        self.path = os.path.expanduser(dirname)
        self.trial_dir = self.path
        self.index_reader = LocalIndexReader(self.path)
        self.logger.info(f"Loading trial {name} at path {self.trial_dir}")
        self._load_collections()
        self._load_tensors()

    def _get_collection_files(self) -> list:
        return list_collection_files_in_directory(get_path_to_collections(self.path))

    def _load_tensors_from_index_tensors(self, index_tensors_dict):
        for tname in index_tensors_dict:
            for step, itds in index_tensors_dict[tname].items():
                for worker in itds:
                    self._add_tensor(int(step), worker, itds[worker]["tensor_location"])

    def _read_collections(self, collection_files):
        first_collection_file = collection_files[0]  # First Collection File
        self.collection_manager = CollectionManager.load(first_collection_file)
        self.num_workers = self.collection_manager.get_num_workers()
