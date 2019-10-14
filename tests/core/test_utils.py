import pytest

from tornasole.core.utils import is_s3, check_dir_exists
from tornasole.core.json_config import DEFAULT_SAGEMAKER_TORNASOLE_PATH, collect_tornasole_config_params
from tornasole.core.collection_manager import CollectionManager

def test_normal():
  rval = is_s3('a/b/c')
  assert not rval[0]

def test_s3():
  rval = is_s3('s3://a/b')
  assert rval[0]
  assert rval[1] == 'a'
  assert rval[2] == 'b'

def test_s3_noprefix():
  rval = is_s3('s3://a')
  assert rval[0]
  assert rval[1] == 'a'
  assert rval[2] == ''

def test_s3_noprefix2():
  rval = is_s3('s3://a/')
  assert rval[0]
  assert rval[1] == 'a'
  assert rval[2] == ''

def test_check_dir_exists_no_local():
  check_dir_exists('/home/ubuntu/asasdas')

def test_check_dir_exists():
  try:
    check_dir_exists('/home/ubuntu/')
    assert False
  except Exception as e:
    pass

def test_check_dir_exists_no_s3():
  check_dir_exists('s3://tornasole-testing/pleasedontexist')

def test_check_dir_exists_s3():
  try:
    check_dir_exists('s3://tornasole-binaries-use1/tornasole_tf/')
    assert False
  except Exception as e:
    pass

def test_check_dir_exists_no():
  try:
    check_dir_exists('s3://tornasole-binaries-use1')
    assert False
  except Exception as e:
    pass

@pytest.mark.skip(reason="If no config file is found, then SM doesn't want a TornasoleHook")
def test_collect_tornasole_config_params():
  tornasole_params = collect_tornasole_config_params(collection_manager=CollectionManager())
  assert(tornasole_params["out_dir"] == DEFAULT_SAGEMAKER_TORNASOLE_PATH)
