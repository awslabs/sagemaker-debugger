# Standard Library

import bisect
import json
import simplejson as sjson
import orjson as ojson
import os
import pandas as pd
import numpy as np
import time
import copy

from multiprocessing import Process, Queue, shared_memory

# First Party
from smdebug.core.access_layer.s3handler import ListRequest, ReadObjectRequest, S3Handler, is_s3
from smdebug.profiler.metrics_reader_base import MetricsReaderBase
from smdebug.profiler.profiler_constants import (
    DEFAULT_SYSTEM_PROFILER_PREFIX,
    ENV_TIME_BUFFER,
    TIME_BUFFER_DEFAULT,
)
from smdebug.profiler.system_profiler_file_parser import ProfilerSystemEvents
from smdebug.profiler.utils import get_utctimestamp_us_since_epoch_from_system_profiler_file
from smdebug.profiler.utils import (
    TimeUnits,
    convert_utc_timestamp_to_microseconds,
    is_valid_tfprof_tracefilename,
    is_valid_tracefilename,
    validate_system_profiler_file,
)

class SystemMetricsReader(MetricsReaderBase):
    def __init__(self, use_in_memory_cache=False):
        super().__init__(use_in_memory_cache)
        self.prefix = DEFAULT_SYSTEM_PROFILER_PREFIX
        self._SystemProfilerEventParser = ProfilerSystemEvents()
        self._event_parsers = [self._SystemProfilerEventParser]

    """
    Return the profiler system event files that were written during the given range. If use_buffer is True, we will consider adding a
    buffer of TIME_BUFFER_DEFAULT microseconds to increase the time range. This is done because the events are written to the
    file after they end. It is possible that an event would have started within the window of start and end, however it
    did not complete at or before 'end' time. Hence the event will not appear in the event file that corresponds to
    'end' timestamp. It will appear in the future event file.
    We will also add a buffer for the 'start' i.e. we will look for event files that were written prior to 'start'.
    Those files might contain 'B' type events that had started prior to 'start'
    """

    def _get_event_files_in_the_range(
        self, start_time_microseconds, end_time_microseconds, use_buffer=True
    ):
        # increase the time range using TIME_BUFFER_DEFAULT
        if use_buffer:
            time_buffer = os.getenv(ENV_TIME_BUFFER, TIME_BUFFER_DEFAULT)
            start_time_microseconds = start_time_microseconds - time_buffer
            end_time_microseconds = end_time_microseconds + time_buffer

        """
        We need to intelligently detect whether we need to refresh the list of available event files.
        Approach 1: Keep the start prefix for S3, 'x' minutes (say 5) lagging behind the last available timestamp.
        This will cover for the case where a node or writer is not efficient enough to upload the files to S3
        immediately. For local mode we may have to walk the directory every time.
        This is currently implemented by computing the start prefix and TRAILING DURATION.
        TODO:
        Approach 2: If we can know the expected number of files per node and per writer, we can intelligently wait
        for that type of file for certain amount of time.
        """

        """
        In case of S3, we will refresh the event file list if the requested end timestamp is less than the timestamp
        of _startAfterPrefix.
        In case of local mode, the event file list will be refreshed if the end timestamp is not less than the last
        available timestamp
        """

        if self._startAfter_prefix is not "":
            if end_time_microseconds >= get_utctimestamp_us_since_epoch_from_system_profiler_file(
                self._startAfter_prefix
            ):
                self.refresh_event_file_list()
        else:
            if end_time_microseconds >= self.get_timestamp_of_latest_available_file():
                self.refresh_event_file_list()

        timestamps = sorted(self._timestamp_to_filename.keys())

        # Find the timestamp that is smaller than or equal start_time_microseconds. The event file corresponding to
        # that timestamp will contain events that are active during start_time_microseconds
        lower_bound_timestamp_index = bisect.bisect_right(timestamps, start_time_microseconds)
        if lower_bound_timestamp_index > 0:
            lower_bound_timestamp_index -= 1

        # Find the timestamp that is immediate right to the end_time_microseconds. The event file corresponding to
        # that timestamp will contain events that are active during end_time_microseconds.
        upper_bound_timestamp_index = bisect.bisect_left(timestamps, end_time_microseconds)

        event_files = list()
        for index in timestamps[lower_bound_timestamp_index : upper_bound_timestamp_index + 1]:
            event_files.extend(self._timestamp_to_filename[index])
        return event_files

    def _get_event_parser(self, filename):
        return self._SystemProfilerEventParser

    def _get_timestamp_from_filename(self, event_file):
        return get_utctimestamp_us_since_epoch_from_system_profiler_file(event_file)

    def _get_event_file_regex(self):
        return r"(.+)\.json"


class LocalSystemMetricsReader(SystemMetricsReader):
    """
    The metrics reader is created with root folder in which the system event files are stored.
    """

    def __init__(self, trace_root_folder, use_in_memory_cache=False):
        self.trace_root_folder = trace_root_folder
        super().__init__(use_in_memory_cache)
        # Pre-build the file list so that user can query get_timestamp_of_latest_available_file()
        # and get_current_time_range_for_event_query
        self.refresh_event_file_list()

    """
    Create a map of timestamp to filename
    """

    def refresh_event_file_list(self):
        self._refresh_event_file_list_local_mode(self.trace_root_folder)

    def parse_event_files(self, event_files):
        self._parse_event_files_local_mode(event_files)


class S3SystemMetricsReader(SystemMetricsReader):
    """
    The s3_trial_path points to a s3 folder in which the system metric event files are stored. e.g.
    s3://my_bucket/experiment_base_folder
    """

    def __init__(self, s3_trial_path, use_in_memory_cache=False):
        super().__init__(use_in_memory_cache)
        s3, bucket_name, base_folder = is_s3(s3_trial_path)
        if not s3:
            self.logger.error(
                "The trial path is expected to be S3 path e.g. s3://bucket_name/trial_folder"
            )
        else:
            self.bucket_name = bucket_name
            self.base_folder = base_folder
            self.prefix = os.path.join(self.base_folder, self.prefix, "")
        # Pre-build the file list so that user can query get_timestamp_of_latest_available_file()
        # and get_current_time_range_for_event_query
        self.refresh_event_file_list()

    def parse_event_files(self, event_files):
        file_read_requests = []
        event_files_to_read = []

        for event_file in event_files:
            if event_file not in self._parsed_files:
                event_files_to_read.append(event_file)
                file_read_requests.append(ReadObjectRequest(path=event_file))

        st_loc1 = time.perf_counter_ns()
        event_data_list = S3Handler.get_objects(file_read_requests)
        en_loc1 = time.perf_counter_ns()
        print("Plain S3 reader retrieved data iN SECONDS {}".format((en_loc1-st_loc1)/1000000000))

        for event_data, event_file in zip(event_data_list, event_files_to_read):
            event_string = event_data.decode("utf-8")
            event_items = event_string.split("\n")
            event_items.remove("")
            for item in event_items:
                event = json.loads(item)
                self._SystemProfilerEventParser.read_event_from_dict(event)
            self._parsed_files.add(event_file)

    """
    Create a map of timestamp to filename
    """

    def refresh_event_file_list(self):
        list_dir = ListRequest(
            Bucket=self.bucket_name,
            Prefix=self.prefix,
            StartAfter=self._startAfter_prefix if self._startAfter_prefix else self.prefix,
        )
        self._refresh_event_file_list_s3_mode(list_dir)

class S3NumpySystemMetricsReader(S3SystemMetricsReader):
    #TODO: Settle the blurb below once the data structures clean up
    """
    While the output of S3SystemMetricsReader is a list of events,
    the output of S3NumpySystemMetricsReader is a dictionary of numpys (pandas? still evolving...)
    We'll start with pandas owing to fillna, 
    """
    separator = '#'

    def __init__(self, s3_trial_path, use_in_memory_cache, 
                 col_names, col_dict,
                 np_store, store_chunk_size, store_time_file_prefix,
                 n_nodes, frequency, n_process=None):
        super().__init__(s3_trial_path, use_in_memory_cache)
        self.col_names = col_names #list of names of indicators
        self.col_dict = col_dict #mapping from name to position. Possibly will decide not needed
        self.np_store = np_store
        self.np_chunk_size = store_chunk_size
        self.tprefix = store_time_file_prefix
        self.n_nodes = n_nodes
        self.frequency = frequency #one out of  100, 200, 500, 1000, 5000, 60000. Can be inferred, not worth it
        self.n_process = n_process #have a say on the S3 reading processes, may want to enforce

    """
    This function is the raison d'etre of a metric reader. It is defined in MetricsReaderBase
    The reason to modify it here for S3NumpySystemMetricsReader is twofold:
    1) Speed. We cut the get_events_within_time_range part; instead, we filter directly
    2) Speed. Separation along algo-1, algo-2, etc requires a different code structure
    """
    def _json_to_numpy(start_time, end_time, freq_delta, 
                       num_rows, num_cols, event_data_list,
                       shared_mem_id, col_dict, queue):

        min_row = num_rows
        max_row = 0
        min_time = end_time
        max_time = 0

        separator = S3NumpySystemMetricsReader.separator

        shm = shared_memory.SharedMemory(name=shared_mem_id)
        arr_from_sh = np.ndarray(num_rows*num_cols,
                                      dtype=np.float64, buffer=shm.buf)
        np_arr = arr_from_sh.reshape((num_rows, num_cols))
        np_arr[:] = np.nan
        
        n_zeros = 0
        n_nonzeros = 0
        for event_data in event_data_list:
            event_string = event_data.decode("utf-8")
            event_items = event_string.split("\n")
            event_items.remove("")
            for item in event_items:
                event = json.loads(item) #ojson better
                if event['Name'].startswith("cpu") == False and\
                   event['Name'].startswith("gpu") == False: #not used
                    continue
                col_ind = col_dict[event['Name']+separator+event['Dimension']]
                cur_time = int(event['Timestamp']*1000000) #fast

                cur_rounded_time = (cur_time//freq_delta)*freq_delta

                #the files contain "later" items, of no interest to us
                if cur_time >= end_time or cur_rounded_time < start_time:
                    continue

                row_ind = (cur_time - start_time)//freq_delta

                #pd_df.iloc[row_ind,col_ind] = event['Value']
                np_arr[row_ind,col_ind] = event['Value']
                if event['Value'] == 0:
                    n_zeros += 1
                else:
                    n_nonzeros += 1

                min_row = min(min_row, row_ind)
                max_row = max(max_row, row_ind)
                min_time = min(min_time, cur_rounded_time)
                max_time = max(max_time, cur_rounded_time)
        
        shm.close()
        queue.put((min_row, max_row, min_time, max_time))

    def get_events(
        self,
        start_time,
        end_time,
        unit=TimeUnits.MICROSECONDS
    ):
        """
        n_nodes argument: a helper, could be figured out but it is more
        economical this way. Will be used to preallocate either:
        1. A list of numpys. Each may have its own st_time/end_time 
        if the nodes turn out to emit data out of sync
        2. One 3 dimensional numpy, if the nodes turn out to be reasonably
        in sync

        For now, we test on one node, no issues
        """

        
        start_time = convert_utc_timestamp_to_microseconds(start_time, unit)
        end_time = convert_utc_timestamp_to_microseconds(end_time, unit)

        self.logger.info("S3NumpyReader requesting {}, {}".format(start_time, end_time))
        n_nodes = self.n_nodes
        freq_delta = self.frequency*1000
        print ("Frequency {}".format(self.frequency))

        # We may have to pass in microsecond timestamps if they are not 
        assert start_time == (start_time//freq_delta)*freq_delta
        assert end_time == (end_time//freq_delta)*freq_delta

        event_files = self._get_event_files_in_the_range(start_time, end_time)

        self.logger.info(f"Getting {len(event_files)} event files")
        self.logger.debug(f"Getting event files : {event_files} ")

        # Download files and parse the events
        file_read_requests = []
        event_files_to_read = []

        """
        TODO : restore the smarts in this one, for now brute force
        for event_file in event_files:
            if event_file not in self._parsed_files:
                event_files_to_read.append(event_file)
                file_read_requests.append(ReadObjectRequest(path=event_file))
        """
        for event_file in event_files:
            event_files_to_read.append(event_file)
            file_read_requests.append(ReadObjectRequest(path=event_file))



        #n_process overrides the "min num file" threshold in S3Handler. We try to multiprocess more, aka a threshold of 100 files may be too much
        nan_per_sec = 1000000000
        st_loc1 = time.perf_counter_ns()
        event_data_list = S3Handler.get_objects(file_read_requests, use_multiprocessing=True, n_process=self.n_process)
        en_loc1 = time.perf_counter_ns()
        self.logger.info("S3NumpyReader: retrieved jsons in {} seconds".format((en_loc1-st_loc1)/nan_per_sec))

        #TODO 1. remove prints 2. get fancy later with n_rows
        num_rows = (end_time-start_time)//(self.frequency*1000)
        num_cols = len(self.col_names)
        self.logger.info("NumpyS3Reader: untrimmed DF shape ({},{})".format(num_rows,len(self.col_names))) 
        pd_df = pd.DataFrame(index=np.arange(num_rows), columns=self.col_names)
        np_arr = np.full((num_rows, num_cols), np.nan)

        min_row = num_rows
        max_row = 0
        min_time = end_time
        max_time = 0

        st_loc1 = time.perf_counter_ns()
        # Data that is multiprocess friendly
        queue = Queue()
        shared_mem_ids = [None]*n_nodes
        for i in range (0, n_nodes):
            shm = shared_memory.SharedMemory(create=True,
                              size = num_rows*num_cols*np.dtype(np.double).itemsize)
            shared_mem_ids[i] = shm.name


        # separate files by nodes
        n_files = len(event_files_to_read)
        event_data_lists = [[]]*n_nodes
        for i in range (0, n_files):
            # event_file looks like ".../../al/2020072222/1595455620.algo-1.json
            comps = event_files_to_read[i].split('.')
            comps = comps[1].split('-')
            node_ind = int(comps[1]) #TODO error handling
            event_data_lists[node_ind-1].append(event_data_list[i])

        # Multiprocess json to numpy
        tasks = [None]*n_nodes
        for i in range (0, n_nodes):
            tasks[i] = Process(target=S3NumpySystemMetricsReader._json_to_numpy,
                            args=(start_time, end_time, freq_delta,
                            num_rows, num_cols, event_data_lists[i],
                            shared_mem_ids[i], copy.deepcopy(self.col_dict), queue))
            tasks[i].start()
        for i in range (0, n_nodes):
            tasks[i].join()

        en_loc1 = time.perf_counter_ns()
        self.logger.info("S3NumpyReader: created numpys in {} seconds".format((en_loc1-st_loc1)/nan_per_sec))

        #These files were parsed inside the tasks. Could come useful for caching
        for event_file in event_files_to_read:
            self._parsed_files.add(event_file)

        min_row = num_rows
        max_row = 0
        min_time = end_time
        max_time = 0

        while queue.empty() is False:
            loc_min_row, loc_max_row, loc_min_time, loc_max_time = queue.get()
            min_row = min(min_row, loc_min_row)
            max_row = max(max_row, loc_max_row)
            min_time = min(min_time, loc_min_time)
            max_time = max(max_time, loc_max_time)
    
        np_arrs = []

        # Could multiprocess the backfill as well. Keep for now
        post_process = True
        for i in range (0, n_nodes):
            shm = shared_memory.SharedMemory(name=shared_mem_ids[i])
            arr_from_sh = np.ndarray(num_rows*num_cols,
                                      dtype=np.float64, buffer=shm.buf)
            np_arr = arr_from_sh.reshape((num_rows, num_cols))

            np_arr = np_arr[min_row:max_row+1,:]
            assert (max_time+freq_delta-min_time)//freq_delta == np_arr.shape[0]

            #post process (fillna/etc) Move out to smdebug_kernel perhaps
            if post_process:
                mask = np.isnan(np_arr)
                self.logger.info("S3NumpyReader: {} nans out of {}".format(mask.sum(), np_arr.size))
                st_loc = time.perf_counter_ns()
                temp_df = pd.DataFrame(np_arr)
                temp_df.fillna(method='ffill', axis=0, inplace=True)
                temp_df.fillna(method='bfill', axis=0, inplace=True)
                fill_val = 0
                temp_df.fillna(fill_val, axis=0, inplace=True)
                np_arr = temp_df.values.astype(np.uint8)
                en_loc = time.perf_counter_ns()
                self.logger.info("S3NumpyReader: DF Non naning took {} seconds".format((en_loc-st_loc)/nan_per_sec))

                np_arrs.append(np.copy(np_arr)) #np.copy? relationship to "unlink"?
                shm.close()
                shm.unlink()

        self.logger.info("S3NumpyReader: returned min/max times: {}/{}".format(min_time, max_time))

        #add freq_delta for slicing consistency
        n = np_arr.shape[0]
        return np_arrs, min_time, max_time+freq_delta

    #mono: collects 1 numpy
    def get_events_mono(
        self,
        start_time,
        end_time,
        unit=TimeUnits.MICROSECONDS
    ):   
        """
        n_nodes argument: a helper, could be figured out but it is more
        economical this way. Will be used to preallocate either:
        1. A list of numpys. Each may have its own st_time/end_time 
        if the nodes turn out to emit data out of sync
        2. One 3 dimensional numpy, if the nodes turn out to be reasonably
        in sync

        For now, we test on one node, no issues
        """
    
        start_time = convert_utc_timestamp_to_microseconds(start_time, unit)
        end_time = convert_utc_timestamp_to_microseconds(end_time, unit)
    
        self.logger.info("S3NumpyReader requesting {}, {}".format(start_time, end_time))
        
        n_nodes = self.n_nodes
        freq_delta = self.frequency*1000
        # We may have to pass in microsecond timestamps if they are not 
        assert start_time == (start_time//freq_delta)*freq_delta
        assert end_time == (end_time//freq_delta)*freq_delta

        event_files = self._get_event_files_in_the_range(start_time, end_time)

        self.logger.info(f"Getting {len(event_files)} event files")
        self.logger.debug(f"Getting event files : {event_files} ")

        # Download files and parse the events
        file_read_requests = []
        event_files_to_read = []

        """
        TODO : restore the smarts in this one, for now brute force
        for event_file in event_files:
            if event_file not in self._parsed_files:
                event_files_to_read.append(event_file)
                file_read_requests.append(ReadObjectRequest(path=event_file))
        """
        for event_file in event_files:
            event_files_to_read.append(event_file)
            file_read_requests.append(ReadObjectRequest(path=event_file))

        #n_process overrides the "min num file" threshold in S3Handler. We try to multiprocess more often
        nan_per_sec = 1000000000
        st_loc1 = time.perf_counter_ns()
        event_data_list = S3Handler.get_objects(file_read_requests, use_multiprocessing=True, n_process=self.n_process)
        en_loc1 = time.perf_counter_ns()
        self.logger.info("S3NumpyReader: retrieved jsons in {} seconds".format((en_loc1-st_loc1)/nan_per_sec))

        #TODO 1. remove prints 2. get fancy later with n_rows
        num_rows = (end_time-start_time)//(self.frequency*1000)
        num_cols = len(self.col_names)
        self.logger.info("NumpyS3Reader: untrimmed DF shape ({},{})".format(num_rows,len(self.col_names)))
        pd_df = pd.DataFrame(index=np.arange(num_rows), columns=self.col_names)
        np_arr = np.full((num_rows, len(self.col_names)), np.nan)

        np_chunk_size = self.np_chunk_size
        print ("np_chunk_size {}".format(np_chunk_size))

        num_chunks = (num_rows+np_chunk_size-1)//np_chunk_size
        np_val_chunks = [None]*num_chunks
        np_time_chunks = [None]*num_chunks
        np_ragged_sizes = [None]*num_chunks
        jagged_metadata = [None]*num_chunks

        n_extra = np_chunk_size//100 
        for i in range (0, num_chunks):
            np_val_chunks[i] =\
                np.full((np_chunk_size+n_extra, num_cols), 
                          -1, dtype = np.int32)
            np_time_chunks[i] =\
                np.full((np_chunk_size+n_extra, num_cols), 
                          -1, dtype = np.int64)
            np_ragged_sizes[i] = np.zeros((num_cols,), dtype = np.int32)

        #multiprocess below as well, by 'algo-XYZ' Keep as is for now
        min_row = num_rows
        max_row = 0
        min_time = end_time
        max_time = 0

        separator = S3NumpySystemMetricsReader.separator

        st_loc1 = time.perf_counter_ns()
        #n_files = len(event_files_to_read)
        for event_data, event_file in zip(event_data_list, event_files_to_read):
            event_string = event_data.decode("utf-8")
            event_items = event_string.split("\n")
            event_items.remove("")
            for item in event_items:
                event = json.loads(item) #ojson better
                if event['Name'].startswith("cpu") == False and\
                   event['Name'].startswith("gpu") == False: #not used
                    continue
                col_ind = self.col_dict[event['Name']+separator+event['Dimension']]
                #cur_time = convert_utc_timestamp_to_microseconds(event['Timestamp'], unit)
                cur_time = int(event['Timestamp']*1000000) #fast

                cur_rounded_time = (cur_time//freq_delta)*freq_delta

                #the files contain earlier or later items, of no interest to us
                if cur_time >= end_time or cur_rounded_time < start_time:
                    continue

                row_ind = (cur_time - start_time)//freq_delta
                chunk_ind = row_ind//np_chunk_size
                chunk_row_ind = row_ind - chunk_ind*np_chunk_size

                # Store the data that will go in the in memory DB
                np_arr[row_ind,col_ind] = event['Value']

                # Store the data that will be written to disk
                # TODO: protect against overflow
                cur_row_ind_in_chunk = np_ragged_sizes[chunk_ind][col_ind]
                try:
                    np_time_chunks[chunk_ind][cur_row_ind_in_chunk][col_ind] =\
                        cur_time
                    np_val_chunks[chunk_ind][cur_row_ind_in_chunk][col_ind] =\
                        int(event['Value'])
                except:
                    print ("chunk_id {} cur_row_ind_in_chunk {} col_ind {}".format(chunk_ind, cur_row_ind_in_chunk, col_ind))
                np_ragged_sizes[chunk_ind][col_ind] += 1

                min_row = min(min_row, row_ind)
                max_row = max(max_row, row_ind)
                min_time = min(min_time, cur_rounded_time)
                max_time = max(max_time, cur_rounded_time)

            self._parsed_files.add(event_file)

        en_loc1 = time.perf_counter_ns()
        self.logger.info("S3NumpyReader: created numpys in {} seconds".format((en_loc1-st_loc1)/nan_per_sec))

        # Sanity checks and write to disk
        st_loc1 = time.perf_counter_ns()
        node_ind = 0
        node_name = "algo-"+str(node_ind+1)
        os.makedirs(os.path.join(self.np_store, node_name), exist_ok=True)
        jagged_metadata[node_ind] = [[], []]

        print ("Number of chunks {}".format(num_chunks))
        for chunk_ind in range (0, num_chunks):
            max_entries_in_chunk = 0
            min_time_in_chunk = max_time # This will yield a filename

            for col_ind in range (0, num_cols):
                n_entries = np_ragged_sizes[chunk_ind][col_ind]
                max_entries_in_chunk = max(max_entries_in_chunk, n_entries)

                if n_entries < 2:
                    continue

                min_t = np_time_chunks[chunk_ind][0][col_ind]
                min_time_in_chunk = min(min_time_in_chunk, min_t)
                max_t = np_time_chunks[chunk_ind][n_entries-1][col_ind]

                # assering that collection is more or less at freq_delta
                # TODO: remove the assertion once enough testing was done
                assert (max_t-min_t)/freq_delta < 1.01*(n_entries-1)

            if max_entries_in_chunk < 2:
                continue
    
            if max_entries_in_chunk > 0:
                val_filename = str(min_time_in_chunk) + ".npy"
                time_filename = self.tprefix + separator + val_filename
                val_filepath =\
                        os.path.join(self.np_store, node_name, val_filename)
                time_filepath =\
                        os.path.join(self.np_store, node_name, time_filename)

                fp_val = np.memmap(val_filepath, 
                                   dtype=np.int32, offset=0, mode='w+',
                                   shape = (np_chunk_size+n_extra, num_cols))
                fp_time = np.memmap(time_filepath, 
                                   dtype=np.int64, offset=0, mode='w+', 
                                   shape = (np_chunk_size+n_extra, num_cols))
                fp_val[:] = np_val_chunks[chunk_ind]
                fp_time[:] = np_time_chunks[chunk_ind]
                fp_val.flush()
                fp_time.flush()

                # store the relevant part of the filename
                jagged_metadata[node_ind][0].append(min_time_in_chunk)
                jagged_metadata[node_ind][1].append(np_ragged_sizes[chunk_ind])
                print (np_ragged_sizes[chunk_ind]) # TODO: remove; visually checking

        en_loc1 = time.perf_counter_ns()
        self.logger.info("S3NumpyReader: write numpys to disk in {} seconds".format((en_loc1-st_loc1)/nan_per_sec))

        np_arr = np_arr[min_row:max_row+1,:]

        # Post process (fillna/etc) Move out to smdebug_kernel
        post_process = True
        if post_process:
            mask = np.isnan(np_arr)
            self.logger.info("S3NumpyReader: {} nans out of {}".format(mask.sum(), np_arr.size))
            st_loc = time.perf_counter_ns()
            #print (np_arr[0:100,:])
            temp_df = pd.DataFrame(np_arr)
            temp_df.fillna(method='ffill', axis=0, inplace=True)
            temp_df.fillna(method='bfill', axis=0, inplace=True)
            fill_val = 101 #TODO constant out
            temp_df.fillna(fill_val, axis=0, inplace=True)
            mask = np.isnan(np_arr)
            #self.logger.info("S3NumpyReader: {} nans out of {}".format(mask.sum(), np_arr.size))
            np_arr = temp_df.values.astype(np.uint8)
            #print (np_arr[0:100,:])
            en_loc = time.perf_counter_ns()
            self.logger.info("S3NumpyReader: DF Non naning took {} seconds".format((en_loc-st_loc)/nan_per_sec))

        self.logger.info("S3NumpyReader: returned min/max times: {}/{}".format(min_time, max_time))
        #add freq_delta for consistency
        assert (max_time+freq_delta-min_time)//freq_delta == np_arr.shape[0]
        return [np_arr], min_time, max_time+freq_delta, jagged_metadata
        #return pd_df.loc[min_row:max_row]

        """
        event_parsers = self.get_all_event_parsers()
        for eventParser in event_parsers:
            result.extend(
                    eventParser.get_events_within_time_range(
                        start_time, end_time, TimeUnits.MICROSECONDS, None 
                    )
                )
            if not self._cache_events_in_memory:
                # clear eventParser events
                eventParser.clear_events()
                # cleanup parsed files set to force the reading of files again
                self._parsed_files = set()
        """

        return result


    def refresh_event_file_list(self):
        list_dir = ListRequest(
            Bucket=self.bucket_name,
            Prefix=self.prefix,
            StartAfter=self._startAfter_prefix if self._startAfter_prefix else self.prefix,
        )
        self._refresh_event_file_list_s3_mode(list_dir)

