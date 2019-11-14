# Standard Library
import json
import os
import time

# First Party
from smdebug.core.config_constants import DEFAULT_SAGEMAKER_METRICS_PATH

METRICS_DIR = os.environ.get(DEFAULT_SAGEMAKER_METRICS_PATH, ".")


class _RawMetricData(object):
    def __init__(self, metric_name, value, iteration_number, timestamp):
        self.MetricName = metric_name
        self.Value = value
        self.Timestamp = timestamp
        self.IterationNumber = iteration_number


class SageMakerFileMetricsWriter(object):
    def __init__(self, filename=None):
        self._file = open(filename or self._metrics_file_name(), "a")
        self._indexes = {}
        self._closed = False

    def _metrics_file_name(self):
        return "{}/{}.json".format(METRICS_DIR, str(os.getpid()))

    def _write_metric_value(self, file, raw_metric_data):
        try:
            self._file.write(json.dumps(raw_metric_data.__dict__))
            self._file.write("\n")
        except AttributeError:
            if self._closed:
                raise ValueError("log_metric called on a closed writer")
            elif not self._file:
                self._file = open(self._metrics_file_name(), "a")
                self._file.write(json.dumps(raw_metric_data.__dict__))
                self._file.write("\n")
            else:
                raise

    def log_metric(self, metric_name, value, iteration_number=None, timestamp=None):
        timestamp = int(round(time.time())) if timestamp is None else int(timestamp)
        resolved_index = int(
            self._indexes.get(metric_name, 0) if iteration_number is None else iteration_number
        )

        value = float(value)
        assert isinstance(resolved_index, int)
        assert isinstance(timestamp, int)

        self._write_metric_value(
            self._file, _RawMetricData(metric_name, value, iteration_number, timestamp)
        )
        if not iteration_number:
            self._indexes[metric_name] = resolved_index + 1

    def close(self):
        if not self._closed and self._file:
            self._file.close()
            self._file = None
        self._closed = True

    def __enter__(self):
        """Return self"""
        return self

    def __exit__(self, type, value, traceback):
        """Execute self.close()"""
        self.close()

    def __del__(self):
        """Execute self.close()"""
        self.close()
