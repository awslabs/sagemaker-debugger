# Standard Library
import struct

# First Party
from smdebug.core.logger import get_logger
from smdebug.core.modes import MODE_PLUGIN_NAME, MODE_STEP_PLUGIN_NAME, ModeKeys
from smdebug.core.tfevent.event_file_reader import get_tensor_data
from smdebug.core.tfevent.proto.event_pb2 import Event
from smdebug.core.tfrecord.record_reader import masked_crc32c
from smdebug.core.tfrecord.record_writer import CHECKSUM_MAGIC_BYTES

logger = get_logger()


class TensorReader:
    def __init__(self, data):
        self.position = 0
        self.data = data
        self.max_pos = len(self.data)

    def _read(self, n):
        data = self.data[self.position : self.position + n]
        self.position += n
        return data

    def read_record(self, check="minimal"):
        payload = None
        strlen_bytes = self._read(8)
        # will give you payload for the record, which is essentially the event.
        strlen = struct.unpack("Q", strlen_bytes)[0]
        saved_len_crc = struct.unpack("I", self._read(4))[0]

        if check in ["minimal", "full"]:
            computed_len_crc = masked_crc32c(strlen_bytes)
            assert saved_len_crc == computed_len_crc

        payload = self._read(strlen)
        saved_payload_crc = struct.unpack("I", self._read(4))[0]
        if check == "full":
            computed_payload_crc = masked_crc32c(payload)
            assert saved_payload_crc == computed_payload_crc
        elif check == "minimal":
            computed_payload_crc = masked_crc32c(CHECKSUM_MAGIC_BYTES)
            assert saved_payload_crc == computed_payload_crc
        else:
            # no check
            pass
        return payload

    def _get_mode_modestep(self, step, plugin_data):
        mode_step = step
        mode = ModeKeys.GLOBAL
        for metadata in plugin_data:
            if metadata.plugin_name == MODE_STEP_PLUGIN_NAME:
                mode_step = int(metadata.content)
            if metadata.plugin_name == MODE_PLUGIN_NAME:
                mode = ModeKeys(int(metadata.content))
        return mode, mode_step

    def read_tensors(self, check="minimal"):
        for (step, summ) in self.read_summaries(check=check):
            for v in summ.value:
                val = v.WhichOneof("value")
                assert val == "tensor"
                tensor_name = v.tag
                # We have found the right tensor at the right step
                tensor_data = get_tensor_data(v.tensor)
                mode, mode_step = self._get_mode_modestep(step, v.metadata.plugin_data)
                yield (tensor_name, step, tensor_data, mode, mode_step)

    def read_summaries(self, check="minimal"):
        for ev in self.read_events(check=check):
            # assert ev.HasField('step')
            if not ev.HasField("summary"):
                continue
            assert ev.HasField("summary")
            yield (ev.step, ev.summary)

    def read_events(self, check="minimal"):
        while self.has_data():
            rec = self.read_record(check=check)
            event = Event()
            event.ParseFromString(rec)
            yield event

    def has_data(self):
        return self.position < self.max_pos
