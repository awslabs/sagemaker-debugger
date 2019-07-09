import struct
from tornasole_core.tfevent.event_pb2 import Event
from tornasole_core.tfevent.event_file_reader import get_tensor_data
from tornasole_core.tfevent.util import make_tensor_proto
from tornasole_core.tfevent.summary_pb2 import Summary, SummaryMetadata
from tornasole_core.tfrecord.record_reader import masked_crc32c, u32
from tornasole_core.tfrecord.record_writer import CHECKSUM_MAGIC_BYTES

class TensorReader:
    def __init__(self, data):
        self.position = 0
        self.data = data
        self.max_pos = len(self.data)

    def _read(self, n):
        data = self.data[self.position:self.position + n]
        self.position += n
        return data

    def read_record(self, check=False):
        payload = None
        strlen_bytes = self._read(8)
        # will give you payload for the record, which is essentially the event.
        strlen = struct.unpack('Q', strlen_bytes)[0]
        saved_len_crc = struct.unpack('I', self._read(4))[0]
        if check:
            computed_len_crc = masked_crc32c(strlen_bytes)
            assert saved_len_crc == computed_len_crc
        payload = self._read(strlen)
        saved_payload_crc = struct.unpack('I', self._read(4))[0]
        if check:
            computed_payload_crc = masked_crc32c(payload)
            assert saved_payload_crc == computed_payload_crc
        else:
            computed_payload_crc = masked_crc32c(CHECKSUM_MAGIC_BYTES)
            assert saved_payload_crc == computed_payload_crc
        return payload

    def read_tensors(self, read_data=False, check=False):
        for (step,summ) in self.read_summaries(check=check):
            for v in summ.value:
                assert v.WhichOneof('value') == 'tensor'
                tensor_name = v.tag
                # We have found the right tensor at the right step
                if read_data:
                    tensor_data = get_tensor_data(v.tensor)
                else:
                    tensor_data = None
                yield (tensor_name, step, tensor_data)

    def read_summaries(self, check=False):
        for ev in self.read_events(check=check):
            #assert ev.HasField('step')
            if not ev.HasField('summary'):
                continue
            assert ev.HasField('summary')
            yield (ev.step, ev.summary)

    def read_events(self, check=False):
        while self.has_data():
            rec = self.read_record(check=check)
            event = Event()
            event.ParseFromString(rec)
            yield event

    def has_data(self):
        return self.position < self.max_pos
