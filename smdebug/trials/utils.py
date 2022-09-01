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
    """
    Args:
        path (str): A local path or an S3 path of the form ``s3://bucket/prefix``. You should see
            directories such as ``collections``, ``events`` and ``index`` at this
            path once the training job starts.

        name (str): A name for a trial.
            It is to help you manage different trials. This is an optional
            parameter, which defaults to the basename of the path if not passed.
            Make sure to give it a unique name to prevent duplication.

    Returns:
        :class:`~smdebug.trials.trial.Trial`:
        An SMDebug trial instance

    The following examples show how to create an SMDebug trial object.

    **Example: Creating an S3 trial**

    .. code:: python

        from smdebug.trials import create_trial
        trial = create_trial(
            path='s3://smdebug-testing-bucket/outputs/resnet',
            name='resnet_training_run'
        )

    **Example: Creating a local trial**

    .. code:: python

        from smdebug.trials import create_trial
        trial = create_trial(
            path='/home/ubuntu/smdebug_outputs/resnet',
            name='resnet_training_run'
        )

    **Example: Restricting analysis to a range of steps**

    You can optionally pass ``range_steps`` to restrict your analysis to a
    certain range of steps. Note that if you do so, Trial will not load data
    from other steps.

    - ``range_steps=(100, None)``: This will load all steps after 100

    - ``range_steps=(None, 100)``: This will load all steps before 100

    - ``range_steps=(100, 200)`` : This will load steps between 100 and 200

    - ``range_steps=None``: This will load all steps

    .. code:: python

        from smdebug.trials import create_trial
        trial = create_trial(
            path='s3://smdebug-testing-bucket/outputs/resnet',
            name='resnet_training',
            range_steps=(100, 200)
        )

    """

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
