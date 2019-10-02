import pytest
import uuid
from tests.analysis.utils import generate_data
from tornasole.trials import create_trial
from tornasole.analysis.utils import no_refresh


def help_test_refresh_with_range(path):
  trial_name = str(uuid.uuid4())
  num_steps = 8
  num_tensors = 10
  for i in range(num_steps):
    generate_data(path=path, trial=trial_name, num_tensors=num_tensors,
                  step=i, tname_prefix='foo', worker='algo-1', shape=(3, 3, 3))
  tr = create_trial(path + trial_name, range_steps=(0,5))
  assert len(tr.available_steps()) == 5
  for i in range(num_steps, num_steps*2):
    generate_data(path=path, trial=trial_name, num_tensors=num_tensors,
                  step=i, tname_prefix='foo', worker='algo-1', shape=(3, 3, 3), export_colls=False)
  assert len(tr.available_steps()) == 5

def help_test_refresh(path):
  trial_name = str(uuid.uuid4())
  num_steps = 8
  num_tensors = 10
  for i in range(num_steps):
    generate_data(path=path, trial=trial_name, num_tensors=num_tensors,
                  step=i, tname_prefix='foo', worker='algo-1', shape=(3, 3, 3))
  tr = create_trial(path + trial_name)

  assert 'foo_' + str(num_tensors+1) not in tr.tensors()
  assert 'foo_1' in tr.tensors()
  assert len(tr.available_steps()) == num_steps
  assert len(tr.tensor('foo_1').steps()) == num_steps

  for i in range(num_steps, num_steps*2):
    generate_data(path=path, trial=trial_name, num_tensors=num_tensors,
                  step=i, tname_prefix='foo', worker='algo-1', shape=(3, 3, 3), export_colls=False)
  assert len(tr.tensor('foo_1').steps()) == num_steps*2
  assert len(tr.available_steps()) == num_steps*2

  generate_data(path=path, trial=trial_name, num_tensors=num_tensors,
                step=num_steps*2 + 1,
                tname_prefix='foo', worker='algo-1', shape=(3, 3, 3), export_colls=False)
  assert len(tr.available_steps()) == num_steps * 2 + 1

  generate_data(path=path, trial=trial_name, num_tensors=num_tensors + 3,
                step=num_steps * 2 + 2,
                tname_prefix='foo', worker='algo-1', shape=(3, 3, 3), export_colls=False)
  assert tr.tensor('foo_' + str(num_tensors+1)) is not None

def help_test_no_refresh(path):
  trial_name = str(uuid.uuid4())
  num_steps = 8
  num_tensors = 10

  for i in range(num_steps):
    generate_data(path=path, trial=trial_name, num_tensors=num_tensors,
                  step=i, tname_prefix='foo', worker='algo-1', shape=(3, 3, 3))
  tr = create_trial(path + trial_name)

  assert 'foo_' + str(num_tensors+1) not in tr.tensors()
  assert 'foo_1' in tr.tensors()
  assert len(tr.available_steps()) == num_steps
  assert len(tr.tensor('foo_1').steps()) == num_steps

  for i in range(num_steps, num_steps*2):
    generate_data(path=path, trial=trial_name, num_tensors=num_tensors,
                  step=i, tname_prefix='foo', worker='algo-1', shape=(3, 3, 3), export_colls=False)

  with no_refresh([tr]) as [tr]:
    assert len(tr.tensor('foo_1').steps()) == num_steps
    assert len(tr.available_steps()) == num_steps

  with no_refresh([tr]):
    assert len(tr.tensor('foo_1').steps()) == num_steps
    assert len(tr.available_steps()) == num_steps

  with no_refresh(tr):
    assert len(tr.tensor('foo_1').steps()) == num_steps
    assert len(tr.available_steps()) == num_steps

def test_no_refresh_local():
  help_test_no_refresh('ts_output/train/')

@pytest.mark.slow # 0:37 to run
def test_no_refresh_s3():
  help_test_no_refresh('s3://tornasole-testing/rules/ts_output/train/')

def test_refresh_with_range_local():
  help_test_refresh_with_range('ts_output/train/')

@pytest.mark.slow # 0:36 to run
def test_refresh_with_range_s3():
  help_test_refresh_with_range('s3://tornasole-testing/rules/ts_output/train/')

def test_refresh_local():
  help_test_refresh('ts_output/train/')

@pytest.mark.slow # 0:47 to run
def test_refresh_s3():
  help_test_refresh('s3://tornasole-testing/rules/ts_output/train/')
