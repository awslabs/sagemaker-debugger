# Standard Library
import os

# First Party
from smdebug.core.utils import is_s3

# Local
from .local_trial import LocalTrial
from .profiler_trial import ProfilerTrial
from .s3_trial import S3Trial


def create_trial(
    path, name=None, profiler=False, output_dir="/opt/ml/processing/outputs/", **kwargs
):
    path = path.strip()  # Remove any accidental leading/trailing whitespace input by the user
    if name is None:
        name = os.path.basename(path)
    s3, bucket_name, prefix_name = is_s3(path)
    if profiler:
        return ProfilerTrial(name=name, trial_dir=path, output_dir=output_dir, **kwargs)
    if s3:
        return S3Trial(name=name, bucket_name=bucket_name, prefix_name=prefix_name, **kwargs)
    else:
        return LocalTrial(name=name, dirname=path, **kwargs)
