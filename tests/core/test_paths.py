from tornasole.core.hook_utils import verify_and_get_out_dir
from tornasole.core.json_config import DEFAULT_SAGEMAKER_TORNASOLE_PATH
from tornasole.core.access_layer.file import get_temp_path, SAGEMAKER_TEMP_PATH_SUFFIX, NON_SAGEMAKER_TEMP_PATH_PREFIX
import os
import uuid


def test_outdir_non_sagemaker():
    id = str(uuid.uuid4())
    path = '/tmp/tests/' + id
    out_dir = verify_and_get_out_dir(path)
    assert out_dir == path
    os.makedirs(path)
    try:
        verify_and_get_out_dir(path)
        # should raise exception as dir present
        assert False
    except RuntimeError as e:
        pass


def test_outdir_sagemaker():
    os.environ['TRAINING_JOB_NAME'] = 'a'
    id = str(uuid.uuid4())
    paths = ['/tmp/tests/' + id, 's3://tmp/tests/' + id]
    for path in paths:
        out_dir = verify_and_get_out_dir(path)
        assert out_dir == DEFAULT_SAGEMAKER_TORNASOLE_PATH
    del os.environ['TRAINING_JOB_NAME']


def test_temp_paths():
    for path in ['/opt/ml/output/tensors/events/a',
                 '/opt/ml/output/tensors/a',
                 '/opt/ml/output/tensors/events/a/b']:
        tp = get_temp_path(path)
        assert tp.endswith(SAGEMAKER_TEMP_PATH_SUFFIX)
        assert not tp.startswith(NON_SAGEMAKER_TEMP_PATH_PREFIX)

    for path in ['/a/b/c', '/opt/ml/output/a', 'a/b/c']:
        tp = get_temp_path(path)
        assert not SAGEMAKER_TEMP_PATH_SUFFIX in tp
        assert tp.startswith(NON_SAGEMAKER_TEMP_PATH_PREFIX)
