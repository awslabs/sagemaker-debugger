from tornasole_core.utils import is_s3, check_dir_exists

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