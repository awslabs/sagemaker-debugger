class TensorLocation:
    def __init__(self, tname, event_file_name, start_idx, length):
        self.tensorname = tname
        self.event_file_name = event_file_name
        self.start_idx = start_idx
        self.length = length

    def serialize(self):
        return format(f'{self.tensorname}, {self.event_file_name}, {self.start_idx},{self.length}')

    @staticmethod
    def deserialize(manifest_line_str, manifest_key_name):
        arr = manifest_line_str.split(",")
        return TensorLocation(arr[0], arr[1], arr[2], arr[3])


class IndexUtil:
    # These functions are common to index reader and index writer
    MAX_INDEX_FILE_NUM_IN_INDEX_PREFIX = 1000

    @staticmethod
    def get_index_prefix_for_step(step_num):
        index_prefix_for_step = step_num // IndexUtil.MAX_INDEX_FILE_NUM_IN_INDEX_PREFIX
        return format(index_prefix_for_step, '09')

    @staticmethod
    def next_index_prefix_for_step(step_num):
        index_prefix_for_step = step_num // IndexUtil.MAX_INDEX_FILE_NUM_IN_INDEX_PREFIX
        return format(index_prefix_for_step + 1, '09')

    @staticmethod
    def indexS3Key(trial_prefix, index_prefix_for_step_str, step_num, worker_name, gpurank=0):
        step_num_str = format(step_num, '012')
        gpurank_str = format(gpurank, '04')
        index_filename = format(f"{step_num_str}_{worker_name}_{gpurank_str}.csv")
        index_key = format(f"{trial_prefix}/index/{index_prefix_for_step_str}/{index_filename}")
        return index_key

    # for a step_num index files lies in prefix step_num/MAX_INDEX_FILE_NUM_IN_INDEX_PREFIX

    @staticmethod
    def get_index_key_for_step(trial_prefix, step_num, worker_name, gpurank=0):
        index_prefix_for_step_str = IndexUtil.get_index_prefix_for_step(step_num)
        return IndexUtil.indexS3Key(trial_prefix, index_prefix_for_step_str, step_num, worker_name,
                                    gpurank)  # let's assume worker_rank and gpu_rank is 0 for now, that is no distibuted support
        # We need to think on naming conventions and access patterns for -
        # 1) muti-node training --> data parallel
        # 2) multi gpu training --> model parallel

    @staticmethod
    def get_step_from_idx_filename(index_file_name):
        i = index_file_name.find("_")
        k = index_file_name[i + 1:].find("_")
        step_num_str = index_file_name[i + 1: i + 1 + k]
        return int(step_num_str)