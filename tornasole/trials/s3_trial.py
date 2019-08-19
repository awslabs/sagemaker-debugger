import time
import os

from tornasole.core.access_layer.s3handler import ReadObjectRequest, ListRequest, S3Handler
from tornasole.core.access_layer.utils import has_training_ended
from tornasole.core.tfevent.util import EventFileLocation
from tornasole.core.collection_manager import CollectionManager
from tornasole.core.tfrecord.tensor_reader import TensorReader

from .trial import EventFileTensor, Trial


class S3Trial(Trial):
    def __init__(self, name, bucket_name, prefix_name,
                 range_steps=None,
                 check=False):
        """
        :param name: for sagemaker job, this should be sagemaker training job name
        :param bucket_name: name of bucket where data is saved
        :param prefix_name: name of prefix such that s3://bucket/prefix is where data is saved
        :param range_steps: range_steps is a tuple representing (start_step, end_step).
                            Only the data from steps in between this range will be loaded
        :param check: whether to check checksum of data saved
        """
        super().__init__(name, range_steps=range_steps,
                         parallel=False, check=check)
        self.logger.info(f'Loading trial {name} at path s3://{bucket_name}/{prefix_name}')
        self.bucket_name = bucket_name
        self.prefix_name = prefix_name
        self.last_event_token = None

        self.s3_handler = S3Handler()

        self._load_collections()
        self._load_tensors()

    def _load_tensors(self):
        self._read_all_events_file_from_s3()

    def training_ended(self):
        return has_training_ended("s3://{}/{}".format(self.bucket_name, self.prefix_name))

    def _load_collections(self):
        num_times_before_warning = 10
        while True:
            # todo get this path from tornasole.core
            key = os.path.join(self.prefix_name, 'collections.ts')
            collections_req = ReadObjectRequest(self._get_s3_location(key))
            obj_data = self.s3_handler.get_objects([collections_req])[0]
            if obj_data is None:
                num_times_before_warning -= 1
                if num_times_before_warning < 0:
                    self.logger.warning('Waiting to read collections')
                else:
                    self.logger.debug('Waiting to read collections')
                time.sleep(2)
                continue

            obj_data = obj_data.decode('utf-8')
            self.collection_manager = CollectionManager.load_from_string(obj_data)
            self.logger.debug('Loaded collections for trial {}'.format(self.name))
            return

    def __hash__(self):
        return hash((self.name, self.bucket_name, self.prefix_name))

    def __eq__(self, other):
        return (self.name, self.bucket_name, self.prefix_name) \
               == (other.name, other.bucket_name, other.prefix_name)

    def get_tensors(self, tname_steps_dict, should_regex_match=False):
        # to be used when getting selective tensors from S3
        # now we do not need to do anything since we read the full event file from S3
        pass

    def refresh_tensors(self):
        #TODO if job finished
        self._read_all_events_file_from_s3(start_after_key=self.last_event_token)

    def _list_s3_objects(self, bucket, prefix, start_after_key=None):
        if start_after_key is None:
            start_after_key = prefix
        self.logger.debug(f'Trying to load events after {start_after_key}')
        list_params = {'Bucket': bucket,'Prefix': prefix, 'StartAfter': start_after_key}
        req = ListRequest(**list_params)
        objects = self._list_prefixes([req])
        if len(objects) > 0:
            self.last_event_token = objects[-1]
        return objects

    def _read_all_events_file_from_s3(self, start_after_key=None):
        # TODO
        # if job ended is saved then there is no more listing required for this bucket prefix
        # if job has ended, save that job ended
        self.keys = []
        # todo get path for events from tornasole.core
        objects = self._list_s3_objects(self.bucket_name,
                                        os.path.join(self.prefix_name, 'events'),
                                        start_after_key)
        self.logger.debug("Got objects:{}".format(objects))
        for objname in objects:
            efl = EventFileLocation.match_regex(objname)
            if efl:
                if (self.range_steps is not None and self._step_in_range(efl.step_num)) or \
                  self.range_steps is None:
                    self.keys.append(objname)
                else:
                    self.logger.debug("Skipping step:{} as it is not in range{} {}".format(efl.step_num, self.range_steps[0], self.range_steps[1]))
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
        res = tr.read_tensors(read_data=self.read_data, check=self.check)
        return list(res)

    # list_info will be a list of ListRequest objects. Returns list of lists of files for each request
    def _list_prefixes(self, list_info):
        files = self.s3_handler.list_prefixes(list_info)
        if len(files) == 1:
            files = files[0]
        return files
