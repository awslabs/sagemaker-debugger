# Standard Library
import json
import os
import shutil
import uuid
from pathlib import Path

# Third Party
import pytest

# First Party
from smdebug.core.collection_manager import CollectionManager
from smdebug.core.config_constants import INCOMPLETE_STEP_WAIT_WINDOW_KEY
from smdebug.core.locations import IndexFileLocationUtils
from smdebug.core.modes import ModeKeys
from smdebug.core.tensor import StepState
from smdebug.exceptions import NoMoreData, StepUnavailable
from smdebug.trials import create_trial


def dummy_trial_creator(trial_dir, num_workers, job_ended):
    Path(trial_dir).mkdir(parents=True, exist_ok=True)
    cm = CollectionManager()
    for i in range(num_workers):
        collection_file_name = f"worker_{i}_collections.json"
        cm.export(trial_dir, collection_file_name)
    if job_ended:
        Path(os.path.join(trial_dir, "training_job_end.ts")).touch()


def dummy_step_creator(trial_dir, global_step, mode, mode_step, worker_name):
    static_step_data = (
        '{"meta": {"mode": "TRAIN", "mode_step": 0, "event_file_name": ""}, '
        '"tensor_payload": ['
        '{"tensorname": "gradients/dummy:0", "start_idx": 0, "length": 1}'
        "]}"
    )

    step = json.loads(static_step_data)
    step["meta"]["mode"] = mode
    step["meta"]["mode_step"] = mode_step

    index_file_location = IndexFileLocationUtils.get_index_key_for_step(
        trial_dir, global_step, worker_name
    )
    Path(os.path.dirname(index_file_location)).mkdir(parents=True, exist_ok=True)
    with open(index_file_location, "w") as f:
        json.dump(step, f)


@pytest.mark.slow
def test_single_writer_all_steps_written_complete_job():
    """Test Scenario Description"
     workers : [a]
     steps :{
        1: [a], 2: [a], 3: [a], 4: [a], 5: [a], 6: [a]
        }
    END_OF_JOB.ts --> Present
    """

    path = "s3://smdebug-testing/resources/has_step_scenarios/single-writer-all-steps-written-complete-job"
    trial = create_trial(path)
    num_workers = len(trial.workers())
    assert num_workers == 1
    assert trial.loaded_all_steps is True
    all_steps = trial.steps(show_incomplete_steps=True)
    completed_steps = trial.steps()
    assert all_steps == [0, 1, 2, 3, 4, 5, 6]
    assert completed_steps == all_steps
    assert trial.has_passed_step(3) == StepState.AVAILABLE
    assert trial.has_passed_step(8) == StepState.UNAVAILABLE
    assert (
        trial.last_index_token
        == "resources/has_step_scenarios/single-writer-all-steps-written-complete-job/index/000000000/000000000006_worker_0.json"
    )
    assert trial.last_complete_step == 6


@pytest.mark.slow
def test_single_writer_all_steps_written_complete_job_two_modes():
    """Test Scenario Description"
     workers : [a]
     modes: TRAIN, EVAL
     steps :{
        0: [worker:a, mode: TRAIN, mode_step: 0],
        10: [worker:a, mode: TRAIN, mode_step: 10],
        20: [worker:a, mode: TRAIN, mode_step: 20],
        30: [worker:a, mode: TRAIN, mode_step: 30],
        40: [worker:a, mode: EVAL, mode_step: 0],
        50: [worker:a, mode: EVAL, mode_step: 10],
        60: [worker:a, mode: EVAL, mode_step: 20],
        70: [worker:a, mode: EVAL, mode_step: 30]
        }
    END_OF_JOB.ts --> Present
    """

    path = os.path.join("ts_output/train/", str(uuid.uuid4()))
    dummy_trial_creator(trial_dir=path, num_workers=1, job_ended=True)
    for i in range(0, 31, 10):
        dummy_step_creator(
            trial_dir=path, global_step=i, mode="TRAIN", mode_step=i, worker_name="worker_0"
        )

    for i in range(0, 31, 10):
        dummy_step_creator(
            trial_dir=path, global_step=i + 40, mode="EVAL", mode_step=i, worker_name="worker_0"
        )

    trial = create_trial(path)
    num_workers = len(trial.workers())
    assert num_workers == 1
    assert trial.loaded_all_steps is True
    all_steps = trial.steps(show_incomplete_steps=True)
    completed_steps = trial.steps()
    assert all_steps == [0, 10, 20, 30, 40, 50, 60, 70]
    assert completed_steps == all_steps
    assert trial.has_passed_step(30) == StepState.AVAILABLE
    assert trial.has_passed_step(23, mode=ModeKeys.TRAIN) == StepState.UNAVAILABLE
    assert trial.has_passed_step(40, mode=ModeKeys.TRAIN) == StepState.UNAVAILABLE
    assert trial.has_passed_step(30, mode=ModeKeys.EVAL) == StepState.AVAILABLE
    assert trial.has_passed_step(23, mode=ModeKeys.EVAL) == StepState.UNAVAILABLE
    assert trial.has_passed_step(80) == StepState.UNAVAILABLE
    assert trial.has_passed_step(80, mode=ModeKeys.TRAIN) == StepState.UNAVAILABLE
    assert trial.has_passed_step(80, mode=ModeKeys.EVAL) == StepState.UNAVAILABLE
    assert trial.last_index_token == os.path.join(
        path, "index/000000000/000000000070_worker_0.json"
    )
    assert trial.last_complete_step == 70
    shutil.rmtree(path, ignore_errors=True)


@pytest.mark.slow
def test_single_writer_all_steps_written_incomplete_job_two_modes():
    """Test Scenario Description"
     workers : [a]
     modes: TRAIN, EVAL
     steps :{
        0: [worker:a, mode: TRAIN, mode_step: 0],
        10: [worker:a, mode: TRAIN, mode_step: 10],
        20: [worker:a, mode: TRAIN, mode_step: 20],
        30: [worker:a, mode: TRAIN, mode_step: 30],
        40: [worker:a, mode: EVAL, mode_step: 0],
        50: [worker:a, mode: EVAL, mode_step: 10],
        60: [worker:a, mode: EVAL, mode_step: 20],
        70: [worker:a, mode: EVAL, mode_step: 30]
        }
    END_OF_JOB.ts --> Absent
    """

    path = os.path.join("ts_output/train/", str(uuid.uuid4()))
    dummy_trial_creator(trial_dir=path, num_workers=1, job_ended=False)
    for i in range(0, 31, 10):
        dummy_step_creator(
            trial_dir=path, global_step=i, mode="TRAIN", mode_step=i, worker_name="worker_0"
        )

    for i in range(0, 31, 10):
        dummy_step_creator(
            trial_dir=path, global_step=i + 40, mode="EVAL", mode_step=i, worker_name="worker_0"
        )

    trial = create_trial(path)
    num_workers = len(trial.workers())
    assert num_workers == 1
    assert trial.loaded_all_steps is False
    all_steps = trial.steps(show_incomplete_steps=True)
    completed_steps = trial.steps()
    assert all_steps == [0, 10, 20, 30, 40, 50, 60, 70]
    assert completed_steps == all_steps
    assert trial.has_passed_step(30) == StepState.AVAILABLE
    assert trial.has_passed_step(23, mode=ModeKeys.TRAIN) == StepState.UNAVAILABLE
    assert trial.has_passed_step(40, mode=ModeKeys.TRAIN) == StepState.NOT_YET_AVAILABLE
    assert trial.has_passed_step(30, mode=ModeKeys.EVAL) == StepState.AVAILABLE
    assert trial.has_passed_step(23, mode=ModeKeys.EVAL) == StepState.UNAVAILABLE
    assert trial.has_passed_step(80) == StepState.NOT_YET_AVAILABLE
    assert trial.has_passed_step(80, mode=ModeKeys.TRAIN) == StepState.NOT_YET_AVAILABLE
    assert trial.has_passed_step(80, mode=ModeKeys.EVAL) == StepState.NOT_YET_AVAILABLE
    assert trial.last_index_token == os.path.join(
        path, "index/000000000/000000000070_worker_0.json"
    )
    assert trial.last_complete_step == 70
    shutil.rmtree(path, ignore_errors=True)


@pytest.mark.slow
def test_single_writer_all_steps_written_incomplete_job():
    """Test Scenario Description"
     workers : [a]
     steps :{
        1: [a], 2: [a], 3: [a], 4: [a], 5: [a], 6: [a]
        }
    END_OF_JOB.ts --> Absent
    """

    path = "s3://smdebug-testing/resources/has_step_scenarios/single-writer-all-steps-written-incomplete-job"
    trial = create_trial(path)
    num_workers = len(trial.workers())
    assert num_workers == 1
    assert trial.loaded_all_steps is False
    all_steps = trial.steps(show_incomplete_steps=True)
    completed_steps = trial.steps()
    assert all_steps == [0, 1, 2, 3, 4, 5, 6]
    assert all_steps == completed_steps
    assert trial.has_passed_step(3) == StepState.AVAILABLE
    assert trial.has_passed_step(8) == StepState.NOT_YET_AVAILABLE
    assert (
        trial.last_index_token
        == "resources/has_step_scenarios/single-writer-all-steps-written-incomplete-job/index/000000000/000000000006_worker_0.json"
    )
    assert trial.last_complete_step == 6

    try:
        trial.wait_for_steps([0, 1, 2, 3, 4, 5, 6])
    except Exception:
        # All the requested steps are available, do not raise an exception
        assert False


@pytest.mark.slow
def test_single_writer_not_all_steps_written_complete_job():
    """Test Scenario Description"
     workers : [a]
     steps :{
        1: [a], 2: [a], 3: [a], 4: [], 5: [a], 6: [a]
        }
    END_OF_JOB.ts --> Present
    """

    path = "s3://smdebug-testing/resources/has_step_scenarios/single-writer-not-all-steps-written-complete"
    trial = create_trial(path)
    num_workers = len(trial.workers())
    assert num_workers == 1
    assert trial.loaded_all_steps is True
    all_steps = trial.steps(show_incomplete_steps=True)
    completed_steps = trial.steps()
    assert all_steps == [0, 1, 2, 3, 5, 6]  # step 4 is missing
    assert completed_steps == all_steps
    assert trial.has_passed_step(3) == StepState.AVAILABLE
    assert trial.has_passed_step(4) == StepState.UNAVAILABLE
    assert trial.has_passed_step(8) == StepState.UNAVAILABLE
    assert (
        trial.last_index_token
        == "resources/has_step_scenarios/single-writer-not-all-steps-written-complete/index/000000000/000000000006_worker_0.json"
    )
    assert trial.last_complete_step == 6

    try:
        trial.wait_for_steps([0, 1, 2, 3, 5, 6])
    except Exception:
        # All the requested steps are available, do not raise an exception
        assert False

    try:
        trial.wait_for_steps([0, 1, 2, 3, 4, 5, 6])
        assert False
    except StepUnavailable:
        # Step 4 is Unavailable
        pass

    try:
        trial.wait_for_steps([0, 1, 2, 3, 5, 6, 7])
        assert False
    except NoMoreData:
        # Step 7 is Unavailable
        # But since training job has ended, we should raise NoMoreData
        pass


@pytest.mark.slow
def test_single_writer_not_all_steps_written_incomplete_job():
    """Test Scenario Description"
     workers : [a]
     steps :{
        1: [a], 2: [a], 3: [a], 4: [], 5: [a], 6: [a]
        }
    END_OF_JOB.ts --> Absent
    """

    path = "s3://smdebug-testing/resources/has_step_scenarios/single-writer-not-all-steps-written-incomplete"
    trial = create_trial(path)
    num_workers = len(trial.workers())
    assert num_workers == 1
    assert trial.loaded_all_steps is False
    all_steps = trial.steps(show_incomplete_steps=True)
    completed_steps = trial.steps()
    assert all_steps == [0, 1, 2, 3, 5, 6]  # step 4 is missing
    assert completed_steps == all_steps
    assert trial.has_passed_step(3) == StepState.AVAILABLE
    assert trial.has_passed_step(4) == StepState.UNAVAILABLE
    assert trial.has_passed_step(8) == StepState.NOT_YET_AVAILABLE
    assert (
        trial.last_index_token
        == "resources/has_step_scenarios/single-writer-not-all-steps-written-incomplete/index/000000000/000000000006_worker_0.json"
    )
    assert trial.last_complete_step == 6


@pytest.mark.slow
def test_three_writers_all_steps_written_complete_job():
    """Test Scenario Description"
     workers : [a,b,c]
     steps :{
        1: [a,b,c], 2: [a,b,c], 3: [a,b,c], 4: [a,b,c], 5: [a,b,c], 6: [a,b,c]
        }
    END_OF_JOB.ts --> Present
    """

    path = "s3://smdebug-testing/resources/has_step_scenarios/three-writers-allsteps-written-complete-job"
    trial = create_trial(path)
    num_workers = len(trial.workers())
    assert num_workers == 3
    assert trial.loaded_all_steps is True
    all_steps = trial.steps(show_incomplete_steps=True)
    completed_steps = trial.steps()
    assert all_steps == [0, 1, 2, 3, 4, 5, 6]
    assert completed_steps == all_steps
    assert trial.last_complete_step == 6
    assert trial.has_passed_step(3) == StepState.AVAILABLE
    assert trial.has_passed_step(4) == StepState.AVAILABLE
    assert trial.has_passed_step(8) == StepState.UNAVAILABLE
    assert (
        trial.last_index_token
        == "resources/has_step_scenarios/three-writers-allsteps-written-complete-job/index/000000000/000000000006_worker_2.json"
    )


@pytest.mark.slow
def test_three_writers_all_steps_written_incomplete_job():
    """Test Scenario Description"
     workers : [a,b,c]
     steps :{
        1: [a,b,c], 2: [a,b,c], 3: [a,b,c], 4: [a,b,c], 5: [a,b,c], 6: [a,b,c]
        }
    END_OF_JOB.ts --> Absent
    """

    path = "s3://smdebug-testing/resources/has_step_scenarios/three-writers-all-steps-written-incomplete-job"
    trial = create_trial(path)
    num_workers = len(trial.workers())
    assert num_workers == 3
    assert trial.loaded_all_steps is False
    all_steps = trial.steps(show_incomplete_steps=True)
    completed_steps = trial.steps()
    assert all_steps == [0, 1, 2, 3, 4, 5, 6]
    assert all_steps == completed_steps
    assert trial.last_complete_step == 6
    assert trial.has_passed_step(3) == StepState.AVAILABLE
    assert trial.has_passed_step(4) == StepState.AVAILABLE
    assert trial.has_passed_step(8) == StepState.NOT_YET_AVAILABLE
    assert (
        trial.last_index_token
        == "resources/has_step_scenarios/three-writers-all-steps-written-incomplete-job/index/000000000/000000000006_worker_2.json"
    )


@pytest.mark.slow
def test_three_writers_not_all_steps_written_complete_job():
    """Test Scenario Description"
     workers : [a,b,c]
     steps :{
        1: [a,b,c], 2: [a,b,c], 3: [a,c], 4: [a,c], 5: [a,c], 6: [a,c]
        }
    END_OF_JOB.ts --> Present
    """

    path = "s3://smdebug-testing/resources/has_step_scenarios/three-writers-not-all-steps-written-complete-job"
    trial = create_trial(path)
    num_workers = len(trial.workers())
    assert num_workers == 3
    assert trial.loaded_all_steps is True
    all_steps = trial.steps(show_incomplete_steps=True)
    completed_steps = trial.steps()
    assert all_steps == [0, 1, 2, 3, 4, 5, 6]
    assert completed_steps == all_steps
    assert trial.has_passed_step(3) == StepState.AVAILABLE
    assert trial.last_complete_step == 6
    assert trial.has_passed_step(4) == StepState.AVAILABLE
    assert trial.has_passed_step(6) == StepState.AVAILABLE
    assert trial.has_passed_step(8) == StepState.UNAVAILABLE
    assert (
        trial.last_index_token
        == "resources/has_step_scenarios/three-writers-not-all-steps-written-complete-job/index/000000000/000000000002_worker_2.json"
    )


@pytest.mark.slow
def test_three_writers_not_all_steps_written_incomplete_job():
    """Test Scenario Description"
     workers : [a,b,c]
     steps :{
        1: [a,b,c], 2: [a,b,c], 3: [a,c], 4: [a,c], 5: [a,c], 6: [a,c]
        }
    END_OF_JOB.ts --> Absent
    """
    path = "s3://smdebug-testing/resources/has_step_scenarios/three-writers-not-all-steps-written-incomplete-job"
    trial = create_trial(path)
    num_workers = len(trial.workers())
    assert num_workers == 3
    assert trial.loaded_all_steps is False
    all_steps = trial.steps(show_incomplete_steps=True)
    completed_steps = trial.steps()
    assert all_steps == [0, 1, 2, 3, 4, 5, 6]
    assert completed_steps == [0, 1, 2]
    assert trial.has_passed_step(2) == StepState.AVAILABLE
    assert trial.last_complete_step == 2
    assert trial.has_passed_step(4) == StepState.NOT_YET_AVAILABLE
    assert trial.has_passed_step(6) == StepState.NOT_YET_AVAILABLE
    assert trial.has_passed_step(8) == StepState.NOT_YET_AVAILABLE
    assert (
        trial.last_index_token
        == "resources/has_step_scenarios/three-writers-not-all-steps-written-incomplete-job/index/000000000/000000000002_worker_2.json"
    )


@pytest.mark.slow
def test_three_writers_not_all_steps_written_but_later_step_written_incomplete_job():
    """Test Scenario Description"
     workers : [a,b,c]
     steps :{
        1: [a,b,c], 2: [a,b,c], 3: [a,c], 4: [a,c], 5: [a,c], 6: [a,b,c]
        }
    END_OF_JOB.ts --> Absent
    """
    path = "s3://smdebug-testing/resources/has_step_scenarios/three-writers-not-all-steps-written-but-later-step-written-incomplete-job"
    trial = create_trial(path)
    num_workers = len(trial.workers())
    assert num_workers == 3
    assert trial.loaded_all_steps is False
    all_steps = trial.steps(show_incomplete_steps=True)
    completed_steps = trial.steps()
    assert all_steps == [0, 1, 2, 3, 4, 5, 6]
    assert completed_steps == all_steps
    assert trial.has_passed_step(2) == StepState.AVAILABLE
    assert trial.last_complete_step == 6
    assert trial.has_passed_step(4) == StepState.AVAILABLE
    assert trial.has_passed_step(6) == StepState.AVAILABLE
    assert trial.has_passed_step(8) == StepState.NOT_YET_AVAILABLE
    assert (
        trial.last_index_token
        == "resources/has_step_scenarios/three-writers-not-all-steps-written-but-later-step-written-incomplete-job/index/000000000/000000000006_worker_2.json"
    )


@pytest.mark.slow
def test_three_writers_one_step_missing_but_later_steps_written_incomplete_job():
    """Test Scenario Description"
     workers : [a,b,c]
     steps :{
        1: [a,b,c], 2: [a,b,c], 3: [], 4: [a,c], 5: [a,c], 6: [a,b,c]
        }
    END_OF_JOB.ts --> Absent
    """
    path = "s3://smdebug-testing/resources/has_step_scenarios/three_writers_one_step_missing_but_later_steps_written_incomplete_job"
    trial = create_trial(path)
    num_workers = len(trial.workers())
    assert num_workers == 3
    assert trial.loaded_all_steps is False
    all_steps = trial.steps(show_incomplete_steps=True)
    completed_steps = trial.steps()
    assert all_steps == [0, 1, 2, 4, 5, 6]
    assert completed_steps == all_steps
    assert trial.has_passed_step(2) == StepState.AVAILABLE
    assert trial.has_passed_step(3) == StepState.UNAVAILABLE
    assert trial.last_complete_step == 6
    assert trial.has_passed_step(4) == StepState.AVAILABLE
    assert trial.has_passed_step(6) == StepState.AVAILABLE
    assert trial.has_passed_step(8) == StepState.NOT_YET_AVAILABLE
    assert (
        trial.last_index_token
        == "resources/has_step_scenarios/three_writers_one_step_missing_but_later_steps_written_incomplete_job/index/000000000/000000000006_worker_2.json"
    )


@pytest.mark.slow
def test_three_writers_one_step_missing_but_later_steps_written_partially_incomplete_job():
    """Test Scenario Description"
     workers : [a,b,c]
     steps :{
        1: [a,b,c], 2: [a,b,c], 3: [], 4: [a,c], 5: [a,c], 6: [a,c]
        }
    END_OF_JOB.ts --> Absent
    """
    path = "s3://smdebug-testing/resources/has_step_scenarios/three_writers_one_step_missing_but_later_steps_written_partially_incomplete_job"
    trial = create_trial(path)
    num_workers = len(trial.workers())
    assert num_workers == 3
    assert trial.loaded_all_steps is False
    all_steps = trial.steps(show_incomplete_steps=True)
    completed_steps = trial.steps()
    assert all_steps == [0, 1, 2, 4, 5, 6]
    assert completed_steps == [0, 1, 2]
    assert trial.has_passed_step(2) == StepState.AVAILABLE
    assert trial.has_passed_step(3) == StepState.NOT_YET_AVAILABLE
    assert trial.last_complete_step == 2
    assert trial.has_passed_step(4) == StepState.NOT_YET_AVAILABLE
    assert trial.has_passed_step(6) == StepState.NOT_YET_AVAILABLE
    assert trial.has_passed_step(8) == StepState.NOT_YET_AVAILABLE
    assert (
        trial.last_index_token
        == "resources/has_step_scenarios/three_writers_one_step_missing_but_later_steps_written_partially_incomplete_job/index/000000000/000000000002_worker_2.json"
    )


@pytest.mark.slow
def test_three_writers_one_step_missing_but_later_steps_written_partially_complete_job():
    """Test Scenario Description"
     workers : [a,b,c]
     steps :{
        1: [a,b,c], 2: [a,b,c], 3: [], 4: [a,c], 5: [a,c], 6: [a,c]
        }
    END_OF_JOB.ts --> Present
    """
    path = "s3://smdebug-testing/resources/has_step_scenarios/three_writers_one_step_missing_but_later_steps_written_partially_complete_job"
    trial = create_trial(path)
    num_workers = len(trial.workers())
    assert num_workers == 3
    assert trial.loaded_all_steps is True
    all_steps = trial.steps(show_incomplete_steps=True)
    completed_steps = trial.steps()
    assert all_steps == [0, 1, 2, 4, 5, 6]
    assert completed_steps == [0, 1, 2, 4, 5, 6]
    assert all_steps == completed_steps
    assert trial.has_passed_step(2) == StepState.AVAILABLE
    assert trial.has_passed_step(3) == StepState.UNAVAILABLE
    assert trial.last_complete_step == 6
    assert trial.has_passed_step(4) == StepState.AVAILABLE
    assert trial.has_passed_step(6) == StepState.AVAILABLE
    assert trial.has_passed_step(8) == StepState.UNAVAILABLE
    assert (
        trial.last_index_token
        == "resources/has_step_scenarios/three_writers_one_step_missing_but_later_steps_written_partially_complete_job/index/000000000/000000000002_worker_2.json"
    )


@pytest.mark.slow
@pytest.mark.skip(reason="Re-enable later when we can figure out why it's hanging")
def test_three_writers_not_all_steps_written_but_later_step_written_complete_job():
    """Test Scenario Description"
     workers : [a,b,c]
     steps :{
        1: [a,b,c], 2: [a,b,c], 3: [a,c], 4: [a,c], 5: [a,c], 6: [a,b,c]
        }
    END_OF_JOB.ts --> Present
    """
    path = "s3://smdebug-testing/resources/has_step_scenarios/three-writers-not-all-steps-written-but-later-step-written-complete-job"
    trial = create_trial(path)
    num_workers = len(trial.workers())
    assert num_workers == 3
    assert trial.loaded_all_steps is True
    all_steps = trial.steps(show_incomplete_steps=True)
    completed_steps = trial.steps()
    assert all_steps == [0, 1, 2, 3, 4, 5, 6]
    assert completed_steps == all_steps
    assert trial.has_passed_step(2) == StepState.AVAILABLE
    assert trial.last_complete_step == 6
    assert trial.has_passed_step(4) == StepState.AVAILABLE
    assert trial.has_passed_step(6) == StepState.AVAILABLE
    assert trial.has_passed_step(8) == StepState.UNAVAILABLE
    assert (
        trial.last_index_token
        == "resources/has_step_scenarios/three-writers-not-all-steps-written-but-later-step-written-complete-job/index/000000000/000000000006_worker_2.json"
    )


@pytest.mark.slow
def test_override_if_too_many_steps_skipped(monkeypatch):
    """Test Scenario Description"
     workers : [a,b,c]
     steps :{
        0: [a], 2: [a], 3: [a], 4: [a], 5: [a], ..., 19: [a]
        }
    END_OF_JOB.ts --> Absent
    Note: This test needs INCOMPLETE_STEP_WAIT_WINDOW to be set to 10 to pass

    This test checks the logic of the sliding window that waits for steps to be complete
    before marking them as complete.

    The last_completed_step is initialized to -1 at the start of the job.

    The logic executed by the sliding window is as follows:
        if total_steps (partial or complete) - last_complete_step +1 > WINDOW_THRESHOLD
            last_complete_step += WINDOW_THRESHOLD //2

    For this test, we set the threshold to 10.

    When the 20 steps are loaded, the last_completed_step is updated from -1 to 4 ( 10/2 + (-1))

    We then perform the query trial.workers(), which calls the refresh operation.

    This performs a second update on last_completed_step to 9

    The subsequent trial.tensors() queries do not change the value of last_completed_step, because the
    window is smaller than the set threshold
    """

    monkeypatch.setenv(INCOMPLETE_STEP_WAIT_WINDOW_KEY, "10")

    path = "s3://smdebug-testing/resources/has_step_scenarios/too-many-steps-skipped"
    trial = create_trial(path)
    assert trial.last_complete_step == 4
    assert (
        trial.last_index_token
        == "resources/has_step_scenarios/too-many-steps-skipped/index/000000000/000000000004_worker_2.json"
    )
    num_workers = len(trial.workers())
    assert num_workers == 3
    assert trial.last_complete_step == 9
    assert (
        trial.last_index_token
        == "resources/has_step_scenarios/too-many-steps-skipped/index/000000000/000000000009_worker_2.json"
    )
    assert trial.loaded_all_steps is False
    all_steps = trial.steps(show_incomplete_steps=True)
    assert all_steps == [i for i in range(20)]
    assert trial.last_complete_step == 9
    assert (
        trial.last_index_token
        == "resources/has_step_scenarios/too-many-steps-skipped/index/000000000/000000000009_worker_2.json"
    )
    trial.tensor_names()
    assert trial.last_complete_step == 9
    assert (
        trial.last_index_token
        == "resources/has_step_scenarios/too-many-steps-skipped/index/000000000/000000000009_worker_2.json"
    )
    trial.tensor_names()
    trial.tensor_names()
    assert trial.last_complete_step == 9
    assert (
        trial.last_index_token
        == "resources/has_step_scenarios/too-many-steps-skipped/index/000000000/000000000009_worker_2.json"
    )


@pytest.mark.slow
def test_partially_written_tensors():
    """
    The trial data accessed by this test was generated with the following hook configs:

        hook.get_collection("gradients").save_config = {
            smd.modes.TRAIN: smd.SaveConfigMode(save_interval=1, start_step=5, end_step=10),
        }


        hook.get_collection("losses").save_config = {
            smd.modes.TRAIN: smd.SaveConfigMode(save_interval=1, end_step=5),
        }

        hook = smd.SessionHook(
                    ...,
                include_collections=["weights", "gradients", "losses"],
                save_config=smd.SaveConfig(save_interval=1, end_step=10),
        )

        The training job was executed by two workers.

        The data was manipulated post completion in the following ways:

            1. END_OF_JOB.ts was deleted.
            2. Index_files for steps: [3, 4, 8, 9] were deleted for one worker
    """

    path = "s3://smdebug-testing/resources/has_step_scenarios/partially_written_tensors/"
    trial = create_trial(path)

    assert trial.steps(show_incomplete_steps=True) == [i for i in range(10)]  # [0, 1, 2, ..., 9]
    assert trial.steps() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert trial.last_complete_step == 7

    """
        Here, we expect steps and all_steps to be equal,
        even though steps 3 and 4 are incomplete because
        trial.last_complete_step > 4.

        So we do not expect steps 3 and 4 to ever be written.

    """
    loss = trial.tensor("loss:0")
    assert loss.steps(show_incomplete_steps=True) == [0, 1, 2, 3, 4]
    assert loss.steps() == [0, 1, 2, 3, 4]

    gradient = trial.tensor("gradients/MatMul_grad/tuple/control_dependency_1:0")
    assert gradient.steps(show_incomplete_steps=True) == [5, 6, 7, 8, 9]
    assert gradient.steps() == [5, 6, 7]

    weight = trial.tensor("foobar/weight1:0")
    assert weight.steps(show_incomplete_steps=True) == trial.steps(show_incomplete_steps=True)
    assert weight.steps() == trial.steps()
