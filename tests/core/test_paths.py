# Standard Library
import os
import uuid
from tempfile import TemporaryDirectory

# First Party
from smdebug.core.access_layer.file import SMDEBUG_TEMP_PATH_SUFFIX, get_temp_path
from smdebug.core.access_layer.utils import training_has_ended
from smdebug.core.hook_utils import verify_and_get_out_dir
from smdebug.core.utils import SagemakerSimulator, ScriptSimulator
from smdebug.exceptions import SMDebugRuntimeError
from smdebug.trials import create_trial

# Local
from ..analysis.utils import dummy_trial_creator


def test_whitespace_handling_in_path_str():
    _id = str(uuid.uuid4())
    path = os.path.join("ts_output/train/", _id)
    dummy_trial_creator(trial_dir=path, num_workers=1, job_ended=True)

    # Test Leading Whitespace Handling
    create_trial("   " + path)

    # Test Trailing Whitespace Handling
    create_trial(path + "  ")


def test_outdir_non_sagemaker():
    id = str(uuid.uuid4())
    path = "/tmp/tests/" + id
    out_dir = verify_and_get_out_dir(path)
    assert out_dir == path
    os.makedirs(path)
    training_has_ended(path)
    try:
        verify_and_get_out_dir(path)
        # should raise exception as dir present
        assert False
    except SMDebugRuntimeError as e:
        pass


def test_temp_paths():
    with SagemakerSimulator() as sim:
        for path in [
            "/opt/ml/output/tensors/events/a",
            "/opt/ml/output/tensors/a",
            "/opt/ml/output/tensors/events/a/b",
        ]:
            temp_path = get_temp_path(path)
            assert temp_path.endswith(SMDEBUG_TEMP_PATH_SUFFIX)

    with ScriptSimulator() as sim:
        for path in ["/a/b/c", "/opt/ml/output/a", "a/b/c"]:
            temp_path = get_temp_path(path)
            assert temp_path.endswith(SMDEBUG_TEMP_PATH_SUFFIX)


def test_s3_path_that_exists_without_end_of_job():
    path = "s3://smdebug-testing/resources/s3-path-without-end-of-job"
    verify_and_get_out_dir(path)
    # should not raise error as dir present but does not have the end of job file
    verify_and_get_out_dir(path)


def test_outdir_sagemaker(monkeypatch):
    with TemporaryDirectory() as dir_name:
        json_file_contents = f"""
                {{
                    "S3OutputPath": "s3://sagemaker-test",
                    "LocalPath": "{dir_name}",
                    "HookParameters" : {{
                        "save_interval": "2",
                        "include_workers": "all"
                    }}
                }}
                """
        from smdebug.tensorflow import get_hook

        with SagemakerSimulator(json_file_contents=json_file_contents) as sim:
            hook = get_hook("keras", create_if_not_exists=True)
            assert hook.out_dir == dir_name
