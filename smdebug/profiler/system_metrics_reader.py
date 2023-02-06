# Standard Library

import bisect
import json
import os
import io
import sys
import math
import pickle
import pandas as pd
import numpy as np
import time
import copy
import psutil
import boto3

if sys.version_info.minor > 7:
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

        if self._startAfter_prefix != "":
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

        event_data_list = S3Handler.get_objects(file_read_requests)

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
    """
    The fuctionality below is inspired by MetricsReaderBase
    The reason to modify it here for S3NumpySystemMetricsReader is twofold:
    1) Speed. We cut the get_events_within_time_range part; instead, we filter directly
    2) Speed. Separation along algo-1, algo-2, etc requires a different code structure

    While the output of S3SystemMetricsReader is a list of events,
    the output of S3NumpySystemMetricsReader is a list of numpys
    """
    separator = '#'
    accu_prefix = 'accu_'
    fill_val = 101

    def __init__(self, s3_trial_path, use_in_memory_cache, 
                 col_names, extra_col_names, col_dict,
                 np_store, store_chunk_size, store_time_file_prefix,
                 group_size, n_nodes, frequency, 
                 accumulate_forward = False, accumulate_minutes = 20,
                 n_process=None, logger = None):
        super().__init__(s3_trial_path, use_in_memory_cache)
        self.col_names = col_names # list of names of indicators
        self.extra_col_names = extra_col_names # list of extra indicators, not stored in memory, as opposed to col_names
        self.col_dict = col_dict # mapping from name to position. Possibly will decide not needed
        self.np_store = np_store
        self.np_chunk_size = store_chunk_size
        self.tprefix = store_time_file_prefix
        self.group_size = group_size
        self.n_nodes = n_nodes
        self.frequency = frequency # one out of  100, 200, 500, 1000, 5000, 60000. Can be inferred, not worth it
        self.n_process = n_process # have a say on the S3 reading processes, may want to enforce

        if logger is not None:
            self.logger = logger

        """
        The fields below serve the following purpose (long story):
        The reader is used for ingesting profiling data, the users of the reader
        prioritizing fresh data over old.
        Old data will be ingested in larger chunks; when written to disk this 
        will not cause file fragmentation.
        Fresh data will be ingested more often (to keep current) which leads to
        file fragmentation. To address this, when ingesting fresh data (smaller
        data batches) we will also accumulate the data from these batches until it
        reaches a threshold, and then we send it to the user. The user has the
        option of doing file manipulation: small files written so far can be deleted
        and only the accumulated version can be kept.

        The reader works under the assumption that reading requests for old and
        new data may alternate: we name read requests for old data "backwards" reads
        and read requests for new data "forward" reads.
        For the purpose of accumulating small reads, we are concerned only with
        forward reads.
        """
        self.accu = accumulate_forward
        self.accu_mins = accumulate_minutes
        self.accu_delta = accumulate_minutes * 60 * 1000000 # microseconds
        self.last_accu_start_time = 0
        self.accu_n_rows = 0
        self.accu_val_mem_ids = [None]*n_nodes 
        self.accu_time_mem_ids = [None]*n_nodes
        self.accu_cnt_mem_ids = [None]*n_nodes
        self.init_accumulators()

    def __del__(self):
        for i in range (0, self.n_nodes):
            if self.accu_val_mem_ids[i] is not None:
               shm = shared_memory.SharedMemory(name=self.accu_val_mem_ids[i])
               shm.unlink()
               shm.close()

            if self.accu_time_mem_ids[i] is not None:
               shm = shared_memory.SharedMemory(name=self.accu_time_mem_ids[i])
               shm.unlink()
               shm.close()

            if self.accu_cnt_mem_ids[i] is not None:
               shm = shared_memory.SharedMemory(name=self.accu_cnt_mem_ids[i])
               shm.unlink()
               shm.close()

    def init_accumulators(self):
        if self.accu is False:
            return

        num_rows = self.accu_mins * 60 * 10 # Max, at highest frequency
        num_rows = num_rows + num_rows//10 # Buffer
        # extra_col_names treated separately because, unlike col_names indices,
        # the indices corresponding to col_names, they are stored on disk only
        num_cols = len(self.col_names) + len(self.extra_col_names)
        self.accu_n_rows = num_rows

        for i in range (0, self.n_nodes):
            shm = shared_memory.SharedMemory(create=True,
                              size = num_rows*num_cols*np.dtype(np.int32).itemsize)
            self.accu_val_mem_ids[i] = shm.name
            shm.close()

            shm = shared_memory.SharedMemory(create=True,
                              size = num_rows*num_cols*np.dtype(np.int64).itemsize)
            self.accu_time_mem_ids[i] = shm.name
            times = np.ndarray((num_rows,num_cols), dtype=np.int64, buffer=shm.buf)
            times[:] = 0 # Useful when determining min time
            shm.close()

            shm = shared_memory.SharedMemory(create=True,
                              size = num_cols*np.dtype(np.int32).itemsize)
            self.accu_cnt_mem_ids[i] = shm.name
            counts = np.ndarray((num_cols,), dtype=np.int32, buffer=shm.buf)
            counts[:] = 0
            shm.close()

            self.logger.info("NumpyS3Reader: ALLOCATED ACCU MEM for node {}".format(i))

    def _json_to_numpy(node_ind, start_time, end_time, freq_delta, 
                       num_rows, num_cols, num_extra_cols,
                       np_chunk_size, event_data_list,
                       shared_mem_id, col_dict,
                       accu_val_mem_id, accu_time_mem_id,
                       accu_cnt_mem_id, accu_num_rows,
                       np_store, tprefix, queue, logger):

        min_row = num_rows
        max_row = 0
        min_time = end_time
        max_time = 0

        num_chunks = (num_rows+np_chunk_size-1)//np_chunk_size
        np_val_chunks = [None]*num_chunks
        np_time_chunks = [None]*num_chunks
        np_ragged_sizes = [None]*num_chunks
        jagged_metadata = [[], []]

        accu_shm_count = None
        accu_shm_val = None
        accu_shm_time = None
        accu_counts = None
        accu_vals = None
        accu_times = None

        if accu_val_mem_id is not None:
            assert accu_time_mem_id is not None
            assert accu_cnt_mem_id is not None

            accu_shm_count = shared_memory.SharedMemory(name=accu_cnt_mem_id)
            accu_counts =\
                np.ndarray((num_cols+num_extra_cols,),
                           dtype=np.int32, buffer=accu_shm_count.buf)

            accu_shm_val = shared_memory.SharedMemory(name=accu_val_mem_id)
            accu_vals = np.ndarray((accu_num_rows, num_cols+num_extra_cols),
                                   dtype=np.int32, buffer=accu_shm_val.buf)

            accu_shm_time = shared_memory.SharedMemory(name=accu_time_mem_id)
            accu_times = np.ndarray((accu_num_rows, num_cols+num_extra_cols),
                               dtype=np.int64, buffer=accu_shm_time.buf)

        n_extra = np_chunk_size//100
        for i in range (0, num_chunks):
            np_val_chunks[i] =\
                np.full((np_chunk_size+n_extra, num_cols+num_extra_cols),
                          -1, dtype = np.int32)
            np_time_chunks[i] =\
                np.full((np_chunk_size+n_extra, num_cols+num_extra_cols),
                          -1, dtype = np.int64)
            np_ragged_sizes[i] = np.zeros((num_cols+num_extra_cols,), dtype = np.int32)

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
                if event['Dimension'] != "Algorithm" and\
                   event['Name'] != "MemoryUsedPercent" and\
                   event['Name'].startswith("cpu") == False and\
                   event['Name'].startswith("gpu") == False:
                    continue

                if event['Name'] == "MemoryUsedPercent":
                    # More informative, when very little RAM work
                    event['Value'] = math.ceil(event['Value'])

                is_extra = (event['Dimension'] == "Algorithm" or\
                            event['Name'] == "MemoryUsedPercent")
                col_ind = col_dict[event['Name']+separator+event['Dimension']]
                cur_time = int(event['Timestamp']*1000000) #fast

                cur_rounded_time = (cur_time//freq_delta)*freq_delta

                #the files contain "later" items, of no interest to us
                if cur_time >= end_time or cur_rounded_time < start_time:
                    continue

                row_ind = (cur_time - start_time)//freq_delta
                chunk_ind = row_ind//np_chunk_size
                chunk_row_ind = row_ind - chunk_ind*np_chunk_size

                if is_extra is False:
                    np_arr[row_ind,col_ind] = event['Value']
                    if event['Value'] == 0:
                        n_zeros += 1
                    else:
                        n_nonzeros += 1

                try: 
                    if accu_val_mem_id is not None and accu_counts[col_ind] < accu_num_rows:
                        cur_count = accu_counts[col_ind]
                        accu_vals[cur_count][col_ind] = int(event['Value'])
                        accu_times[cur_count][col_ind] = cur_time
                        accu_counts[col_ind] += 1
                except:
                    pass

                cur_row_ind_in_chunk = np_ragged_sizes[chunk_ind][col_ind]
                try:
                    np_time_chunks[chunk_ind][cur_row_ind_in_chunk][col_ind] =\
                        cur_time
                    np_val_chunks[chunk_ind][cur_row_ind_in_chunk][col_ind] =\
                        int(event['Value'])
                except:
                    pass
                    #print ("chunk_id {} cur_row_ind_in_chunk {} col_ind {}".format(chunk_ind, cur_row_ind_in_chunk, col_ind))
                np_ragged_sizes[chunk_ind][col_ind] += 1

                min_row = min(min_row, row_ind)
                max_row = max(max_row, row_ind)
                min_time = min(min_time, cur_rounded_time)
                max_time = max(max_time, cur_rounded_time)
        
        shm.close()

        if accu_val_mem_id is not None:
            accu_shm_count.close()
            accu_shm_val.close()
            accu_shm_time.close()

        node_name = S3NumpySystemMetricsReader.node_name_from_index(node_ind)
        if not np_store.startswith("s3://"):
            os.makedirs(os.path.join(np_store, node_name), exist_ok=True)

        for chunk_ind in range (0, num_chunks):
            max_entries_in_chunk = 0
            min_time_in_chunk = max_time # This will yield a filename

            for col_ind in range (0, num_cols+num_extra_cols):
                n_entries = np_ragged_sizes[chunk_ind][col_ind]
                max_entries_in_chunk = max(max_entries_in_chunk, n_entries)

                if n_entries < 2:
                    continue

                min_t = np_time_chunks[chunk_ind][0][col_ind]
                min_time_in_chunk = min(min_time_in_chunk, min_t)
                max_t = np_time_chunks[chunk_ind][n_entries-1][col_ind]

                # Is the collection is more or less at freq_delta?
                # assert (max_t-min_t)/freq_delta < 1.01*n_entries
                if (max_t-min_t)/freq_delta > 1.01*n_entries:
                    entries_ratio = (max_t-min_t)/(freq_delta*n_entries)

                    logger.info("S3NumpyReader _json_to_np imbalance on node {}, ratio {}".format(
                        node_ind, entries_ratio))

            if max_entries_in_chunk < 2:
                continue

            if max_entries_in_chunk > 0:
                shp = (np_chunk_size+n_extra, num_cols+num_extra_cols)
                S3NumpySystemMetricsReader.store_vals_times(
                         node_ind, min_time_in_chunk, np_store, tprefix,
                         shp, np_val_chunks[chunk_ind], np_time_chunks[chunk_ind])
                # store the relevant part of the filename
                jagged_metadata[0].append(min_time_in_chunk.item())
                jagged_metadata[1].append(np_ragged_sizes[chunk_ind])
                #print("RAGGED min_time_in_chunk type: {}".format(type(min_time_in_chunk.item())))
                #print("RAGGED np_ragged_sized type: {}".format(type(np_ragged_sizes[chunk_ind])))
        
        logger.info("S3NumpyReader _json_to_numpy FINISHED for node {} min_row {}, max_row {}, min_time {}, max_time {}".format(node_ind, min_row, max_row, min_time, max_time))
        queue.put((node_ind, min_row, max_row, min_time, max_time, jagged_metadata))

    def get_events(
        self,
        start_time,
        end_time,
        forward,
        unit=TimeUnits.MICROSECONDS
    ):
        start_time = convert_utc_timestamp_to_microseconds(start_time, unit)
        end_time = convert_utc_timestamp_to_microseconds(end_time, unit)
        self.logger.info("S3NumpyReader requesting {}, {}".format(start_time, end_time))

        """
        If forward accumulation (see __init__ ...) do some book keeping
        Also, flush the "accumulated" files to disk and prepare 
        "accumulated" jagged metadata
        to return
        """
        jagged_accu_metadata =\
                self.collect_accu_metadata(start_time, end_time, forward)

        n_nodes = self.n_nodes
        group_size = self.group_size
        freq_delta = self.frequency*1000
        freq_delta_group = freq_delta*group_size

        # The time interval must arrive in proper multiplicity
        assert start_time == (start_time//freq_delta_group)*freq_delta_group
        assert end_time == (end_time//freq_delta_group)*freq_delta_group

        event_files = self._get_event_files_in_the_range(start_time, end_time)

        self.logger.info(f"Getting {len(event_files)} event files")
        self.logger.debug(f"Getting event files : {event_files} ")

        # Download files and parse the events
        file_read_requests = [[] for i in range(n_nodes)]
        event_files_to_read = [[] for i in range(n_nodes)]
        event_data_lists = [None]*n_nodes

        """
        TODO : restore the smarts in this one, for now brute force
        for event_file in event_files:
            if event_file not in self._parsed_files:
                event_files_to_read.append(event_file)
                file_read_requests.append(ReadObjectRequest(path=event_file))
        """
        for event_file in event_files:
            comps = event_file.split('.')
            comps = comps[1].split('-')
            node_ind = int(comps[1]) #TODO error handling
            if node_ind > n_nodes:
                continue
            event_files_to_read[node_ind-1].append(event_file)
            file_read_requests[node_ind-1].append(ReadObjectRequest(path=event_file))

        nan_per_sec = 1000000000

        st_loc1 = time.perf_counter_ns()
        for i in range (0, n_nodes):
            event_data_lists[i] = S3Handler.get_objects(file_read_requests[i], use_multiprocessing=True, n_process=self.n_process)
        en_loc1 = time.perf_counter_ns()
        self.logger.info("S3NumpyReader: retrieved jsons in {} seconds".format((en_loc1-st_loc1)/nan_per_sec))

        num_rows = (end_time-start_time)//(self.frequency*1000)
        num_cols = len(self.col_names)
        # Not stored in memory, as opposed to col_names based indicators:
        num_extra_cols = len(self.extra_col_names)
        np_chunk_size = self.np_chunk_size
        self.logger.info("NumpyS3Reader: untrimmed DF shape ({},{})".format(num_rows,len(self.col_names))) 
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
                              size = num_rows*num_cols*np.dtype(np.float64).itemsize)
            shared_mem_ids[i] = shm.name
            np_arr = np.ndarray(num_rows*num_cols,
                        dtype=np.float64, buffer=shm.buf).reshape((num_rows, num_cols))
            np_arr[:,:] = np.nan
            self.logger.info("NumpyS3Reader: ALLOCATED AND NANED for node {}".format(i))

        """
        # separate files by nodes
        n_files = len(event_files_to_read)
        event_data_lists = [[] for i in range(n_nodes)]
        for i in range (0, n_files):
            # event_file looks like ".../../al/2020072222/1595455620.algo-1.json
            comps = event_files_to_read[i].split('.')
            comps = comps[1].split('-')
            node_ind = int(comps[1]) #TODO error handling
            event_data_lists[node_ind-1].append(event_data_list[i])
        """

        # Multiprocess json to numpy
        tasks = [None]*n_nodes
        for i in range (0, n_nodes):
            tasks[i] = Process(target=S3NumpySystemMetricsReader._json_to_numpy,
                          args=(i, start_time, end_time, freq_delta,
                          num_rows, num_cols, num_extra_cols, np_chunk_size,
                          event_data_lists[i],
                          shared_mem_ids[i], copy.deepcopy(self.col_dict),
                          self.accu_val_mem_ids[i] if forward else None,
                          self.accu_time_mem_ids[i] if forward else None,
                          self.accu_cnt_mem_ids[i] if forward else None,
                          self.accu_n_rows,
                          self.np_store, self.tprefix, queue, self.logger))
            tasks[i].start()
        for i in range (0, n_nodes):
            tasks[i].join()

        en_loc1 = time.perf_counter_ns()
        self.logger.info("S3NumpyReader: created numpys in {} seconds".format((en_loc1-st_loc1)/nan_per_sec))

        # These files were parsed inside the tasks. Could come useful for caching
        for i in range (0, n_nodes):
            for event_file in event_files_to_read[i]:
                self._parsed_files.add(event_file)

        min_row = num_rows
        max_row = 0
        min_time = end_time
        max_time = 0

        jagged_metadata = [None]*n_nodes
        while queue.empty() is False:
            node_id, loc_min_row, loc_max_row,\
                    loc_min_time, loc_max_time, jagged_loc = queue.get()
            min_row = min(min_row, loc_min_row)
            max_row = max(max_row, loc_max_row)
            min_time = min(min_time, loc_min_time)
            max_time = max(max_time, loc_max_time)
            jagged_metadata[node_id] = jagged_loc

        """
        Adjust min_row and max_row to multiples of group_size,
        similar for times
        """
        #print ("num_rows {}, min_row {}, max_row {}, group_size {}".format(num_rows, min_row, max_row, group_size))
        min_row = (min_row//group_size)*group_size
        max_row = ((max_row)//group_size)*group_size + (group_size-1)
        min_time = (min_time//freq_delta_group)*freq_delta_group
        max_time = min_time + ((max_row-min_row)//group_size)*freq_delta_group
        #print ("num_rows {}, min_row {}, max_row {}, group_size {}".format(num_rows, min_row, max_row, group_size))
        assert max_row < num_rows

        np_arrs = []

        # Could multiprocess the backfill as well. Not a bottleneck for now
        post_process = True 
        """
        Logic for fl below. 
        We want to backfill, but not when the user pauses profiling
        Also, the user may have switched profiling frequencies, max being 60000,
        our original being possibly 100. So we deem as "interruption"
        if we do not see a signal within 
        fudge_factor * 60000 * "max freq" / "our preq" 
        worst case is 3 minutes
        """
        fl = int(max(5, 3*60000/self.frequency))
        for i in range (0, n_nodes):
            shm = shared_memory.SharedMemory(name=shared_mem_ids[i])
            arr_from_sh = np.ndarray(num_rows*num_cols,
                                      dtype=np.float64, buffer=shm.buf)
            np_arr = arr_from_sh.reshape((num_rows, num_cols))

            np_arr = np_arr[min_row:max_row+1,:]
            assert (np_arr.shape[0]//group_size)*group_size == np_arr.shape[0]

            if post_process:
                mask = np.isnan(np_arr)
                self.logger.info("S3NumpyReader: {} nans out of {}".format(mask.sum(), np_arr.size))
                st_loc = time.perf_counter_ns()
                temp_df = pd.DataFrame(np_arr)

                # Fill at most 5 missing values, if more, there is a gap
                temp_df.fillna(method='ffill', axis=0, limit = fl, inplace=True)
                temp_df.fillna(method='bfill', axis=0, limit = fl, inplace=True)
                # If anything is left to fill, we had a profiling gap. Zero it
                temp_df.fillna(0, axis=0, inplace=True)
                temp_df.fillna(0, axis=0, inplace=True)

                fill_val = S3NumpySystemMetricsReader.fill_val
                temp_df.fillna(fill_val, axis=0, inplace=True)
                np_arr = temp_df.values

                n_binned_rows = np_arr.shape[0]//group_size
                binned_arr = np.full((n_binned_rows, num_cols), np.nan)

                for j in range(0, n_binned_rows):
                    binned_arr[j, :] =\
                       np_arr[j*group_size:(j+1)*group_size, :].mean(axis=0)

                np_arr = binned_arr.astype(np.uint8)
                en_loc = time.perf_counter_ns()
                self.logger.info("S3NumpyReader: DF Non naning took {} seconds".format((en_loc-st_loc)/nan_per_sec))

            np_arrs.append(np.copy(np_arr)) #np.copy? relationship to "unlink"?
            shm.close()
            shm.unlink()

        self.logger.info("S3NumpyReader: returned min/max times: {}/{}".format(min_time, max_time+freq_delta_group))

        n = np_arr.shape[0]
        # Reason for returning an array of tuples: toggling macro profiler on/off
        # Add freq_delta for slicing consistency
        return np_arrs, [[min_time, max_time+freq_delta_group]], jagged_metadata, jagged_accu_metadata

    def collect_accu_metadata(self, start_time: int, end_time: int, forward: bool):
        if self.accu is False or forward is False:
            return None

        if self.last_accu_start_time == 0:
            # Initialize
            self.last_accu_start_time = start_time

        assert end_time - start_time < self.accu_delta # Accumulate smaller chunks

        if end_time - self.last_accu_start_time <=  self.accu_delta:
            return None # Did not accumulate enough

        """
        Write files to disk 
        collect jagged metadata,
        reset accu_cnts
        reset last_accu_start_time and return jagged_accu_metadata
        """

        self.logger.info ("S3NumpyReader: writting accumulated data")

        # both col_names based and extra_col_names based indicators go on disk:
        num_cols = len(self.col_names) + len(self.extra_col_names)
        num_rows = self.accu_n_rows
        np_store = self.np_store
        tprefix = self.tprefix
        separator = S3NumpySystemMetricsReader.separator
        accu_prefix = S3NumpySystemMetricsReader.accu_prefix

        jagged_metadata_supp = [None]*self.n_nodes

        for i in range (0, self.n_nodes):
            jagged_metadata_loc = []
            assert self.accu_cnt_mem_ids[i] is not None
            shm_count = shared_memory.SharedMemory(name=self.accu_cnt_mem_ids[i])
            counts = np.ndarray((num_cols,), dtype=np.int32, buffer=shm_count.buf)
            num_effective_rows = np.amax(counts)
            assert num_effective_rows <= num_rows
            self.logger.info ("S3NumpyReader: rows: {} effective rows: {}".\
                    format(num_rows, num_effective_rows))

            shm_val = shared_memory.SharedMemory(name=self.accu_val_mem_ids[i])
            vals = np.ndarray((num_rows, num_cols), dtype=np.int32, buffer=shm_val.buf)
            effective_vals = vals[0:num_effective_rows,:]

            shm_time = shared_memory.SharedMemory(name=self.accu_time_mem_ids[i])
            times = np.ndarray((num_rows, num_cols), dtype=np.int64, buffer=shm_time.buf)
            effective_times = times[0:num_effective_rows,:]

            # Get the minimum occuring time to be used in filename
            first_times = effective_times[0,:]
            min_time = np.amax(first_times)
            for j in range (0, num_cols):
                if first_times[j] > 0:
                    min_time = min(first_times[j], min_time)

            shp = (num_effective_rows, num_cols)

            S3NumpySystemMetricsReader.store_vals_times(
                        i, min_time, np_store, tprefix,
                        shp, effective_vals, effective_times, accu_prefix)
            
            jagged_metadata_loc.append(min_time)
            jagged_metadata_loc.append(np.copy(counts))
            jagged_metadata_supp[i] = jagged_metadata_loc
            
            # Reset counts
            counts[:] = 0

            shm_count.close()
            shm_val.close()
            shm_time.close()

        # Reset accumulator start time
        self.last_accu_start_time = start_time
 
        return jagged_metadata_supp

    def get_group_size(self):
        return self.group_size

    def set_group_size(self, group_size):
        # TODO: protect setting, we should not be mi-read
        self.group_size = group_size

    def refresh_event_file_list(self):
        list_dir = ListRequest(
            Bucket=self.bucket_name,
            Prefix=self.prefix,
            StartAfter=self._startAfter_prefix if self._startAfter_prefix else self.prefix,
        )
        self._refresh_event_file_list_s3_mode(list_dir)

    @staticmethod
    def node_name_from_index(node_id):
        node_name = "algo-"+str(node_id+1)
        return node_name

    @staticmethod
    def split_s3_path(s3_path):
        path_parts=s3_path.replace("s3://","").split("/")
        bucket=path_parts.pop(0)
        key="/".join(path_parts)
        return bucket, key

    @staticmethod
    def stored_fnames(node_id, file_min_time, tprefix, accu_prefix = None):
        separator = S3NumpySystemMetricsReader.separator
        val_filename = str(file_min_time) + separator + str(node_id+1) + ".npy"
        time_filename = tprefix + separator + val_filename

        if accu_prefix is not None:
            val_filename = accu_prefix + val_filename
            time_filename = accu_prefix + time_filename

        return val_filename, time_filename

    @staticmethod
    def store_vals_times(node_ind, min_time_in_chunk, np_store, tprefix,
                         shp, np_val, np_time, accu_prefix = None):
        val_filename, time_filename =\
            S3NumpySystemMetricsReader.stored_fnames(
                    node_ind, min_time_in_chunk, tprefix, accu_prefix)
        node_name = S3NumpySystemMetricsReader.node_name_from_index(node_ind)

        if np_store.startswith("s3://"):
            S3NumpySystemMetricsReader.dump_to_s3(np_store, node_name, val_filename, np_val)
            S3NumpySystemMetricsReader.dump_to_s3(np_store, node_name, time_filename, np_time)
        else:
            S3NumpySystemMetricsReader.dump_to_disk(np_store, node_name, val_filename, np_val, shp, dtype=np.int32)
            S3NumpySystemMetricsReader.dump_to_disk(np_store, node_name, time_filename, np_time, shp, dtype=np.int64)

    @staticmethod
    def store_vals(node_ind, np_store, shp, np_data, val_type="", dtype=np.int32):
        node_name = S3NumpySystemMetricsReader.node_name_from_index(node_ind)
        separator = S3NumpySystemMetricsReader.separator
        filename =  val_type + separator + str(node_ind+1) + ".npy"
        if np_store.startswith("s3://"):
            S3NumpySystemMetricsReader.dump_to_s3(np_store, node_name, filename, np_data)
        else:
            S3NumpySystemMetricsReader.dump_to_disk(np_store, node_name, filename, np_data, shp, dtype=dtype)
            
    @staticmethod
    def dump_to_s3(s3_storage_loc, node_name, filename, np_data):
        s3_client = boto3.client('s3')
        bucket, key = S3NumpySystemMetricsReader.split_s3_path(s3_storage_loc)
        filepath = os.path.join(key, node_name, filename)

        data_stream = io.BytesIO()
        pickle.dump(np_data, data_stream)
        data_stream.seek(0)
        s3_client.upload_fileobj(data_stream, bucket, filepath)

    @staticmethod
    def dump_to_disk(disk_storage_loc, node_name, filename, np_data, shp, dtype=np.int32):
            directory = os.path.join(disk_storage_loc, node_name)
            filepath = os.path.join(disk_storage_loc, node_name, filename)
        
            if not os.path.exists(directory):
                os.makedirs(directory)
            fp_numpy = np.memmap(filepath,
                               dtype=dtype, offset=0, mode='w+', shape = shp)

            fp_numpy[:] = np_data
            fp_numpy.flush()
