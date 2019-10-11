import os

from tornasole.core.access_layer.s3handler import ReadObjectRequest, S3Handler
from tornasole.core.s3_utils import list_s3_objects
from tornasole.core.locations import TensorFileLocation
from tornasole.core.collection_manager import CollectionManager
from tornasole.core.tfrecord.tensor_reader import TensorReader
from tornasole.core.utils import step_in_range

from .trial import EventFileTensor, Trial


class S3Trial(Trial):
    def __init__(self, name, bucket_name, prefix_name,
                 range_steps=None,
                 check=False,
                 index_mode=True,
                 cache=False):
        """
        :param name: for sagemaker job, this should be sagemaker training job name
        :param bucket_name: name of bucket where data is saved
        :param prefix_name: name of prefix such that s3://bucket/prefix is where data is saved
        :param range_steps: range_steps is a tuple representing (start_step, end_step).
                            Only the data from steps in between this range will be loaded
        :param check: whether to check checksum of data saved
        """
        super().__init__(name, range_steps=range_steps,
                         parallel=False, check=check, index_mode=index_mode, cache=cache)
        self.logger.info(f'Loading trial {name} at path s3://{bucket_name}/{prefix_name}')
        self.bucket_name = bucket_name
        self.prefix_name = os.path.join(prefix_name, '')
        self.path = "s3://" + os.path.join(self.bucket_name, self.prefix_name)
        self.s3_handler = S3Handler()
        self._load_collections()
        self.load_tensors()

    def _load_tensors_from_index_tensors(self, index_tensors_dict):
        for tname in index_tensors_dict:
            for step, itds in index_tensors_dict[tname].items():
                for worker in itds:
                    self.add_tensor(int(step), worker, itds[worker]['tensor_location'])

    def read_collections(self, collection_files):
        first_collection_file = collection_files[0]  # First Collection File
        key = os.path.join(self.prefix_name, first_collection_file)
        collections_req = ReadObjectRequest(self._get_s3_location(key))
        obj_data = self.s3_handler.get_objects([collections_req])[0]
        obj_data = obj_data.decode('utf-8')
        self.collection_manager = CollectionManager.load_from_string(obj_data)
        self.num_workers = self.collection_manager.get_num_workers()

    def get_tensors(self, tname_steps_dict, should_regex_match=False):
        # to be used when getting selective tensors from S3
        # now we do not need to do anything since we read the full event file from S3
        pass

    def _load_tensors_from_event_files(self, start_after_key=None):
        # TODO
        # if job ended is saved then there is no more listing required for this bucket prefix
        # if job has ended, save that job ended
        self.keys = []
        # todo get path for events from tornasole.core
        objects, self.last_event_token = list_s3_objects(self.bucket_name,
                                                         os.path.join(self.prefix_name, 'events'),
                                                         start_after_key)
        self.logger.debug("Got objects:{}".format(objects))
        for objname in objects:
            efl = TensorFileLocation.match_regex(objname)
            if efl:
                if (self.range_steps is not None and step_in_range(self.range_steps, efl.step_num)) or \
                        self.range_steps is None:
                    self.keys.append(objname)
                else:
                    self.logger.debug("Skipping step:{} as it is not in range{} {}"
                                      .format(efl.step_num, self.range_steps[0], self.range_steps[1]))
            else:
                self.logger.debug(f'Skipping object {objname}')
        self.logger.debug(f'Loading {len(self.keys)} new steps')
        self._read_keys()

    def _read_keys(self):
        reqs = []
        filenames = []
        for key in self.keys:
            reqs += self._read_key(key)
            filenames += [self._get_s3_location(key)]
        raw_data = self.s3_handler.get_objects(reqs)
        tensors_in_eventfiles = []
        for i in range(len(raw_data)):
            data = raw_data[i]
            sf = self._read_tensors_from_data(data)
            for tup in sf:
                n, s, d, mode, mode_step = tup
                eft = EventFileTensor(filenames[i], tensor_name=n, step_num=s, tensor_value=d,
                                      mode=mode, mode_step=mode_step)
                tensors_in_eventfiles.append(eft)
        self._add_tensors_at_steps(tensors_in_eventfiles)

    def _read_key(self, key):
        reqs = []
        full_name = self._get_s3_location(key)
        self.logger.debug(f'Reading from {full_name}')
        req = ReadObjectRequest(full_name)
        reqs += [req]
        return reqs

    def _get_s3_location(self, obj):
        return 's3://' + self.bucket_name + "/" + obj

    def _read_tensors_from_data(self, data):
        tr = TensorReader(data)
        res = tr.read_tensors(check=self.check)
        return list(res)
