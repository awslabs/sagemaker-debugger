import os

os.environ[
    "TORNASOLE_CONFIG_MAX_WAIT_STEPS"
] = "10"  # Set to this value for test_override_if_too_many_steps_skipped

import pytest

from tornasole.trials import create_trial
from tornasole.core.tensor import StepState


@pytest.mark.slow
def test_single_writer_all_steps_written_complete_job():
    """Test Scenario Description"
     workers : [a]
     steps :{
        1: [a], 2: [a], 3: [a], 4: [a], 5: [a], 6: [a]
        }
    END_OF_JOB.ts --> Present
    """

    path = "s3://tornasole-testing/has_step_scenarios/single-writer-all-steps-written-complete-job"
    trial = create_trial(path)
    num_workers = len(trial.workers())
    assert num_workers == 1
    assert trial.loaded_all_steps is True
    available_steps = trial.available_steps()
    assert available_steps == [0, 1, 2, 3, 4, 5, 6]
    assert trial.has_passed_step(3) == StepState.AVAILABLE
    assert trial.has_passed_step(8) == StepState.UNAVAILABLE
    assert (
        trial.last_index_token
        == "has_step_scenarios/single-writer-all-steps-written-complete-job/index/000000000/000000000006_worker_0.json"
    )


@pytest.mark.slow
def test_single_writer_all_steps_written_incomplete_job():
    """Test Scenario Description"
     workers : [a]
     steps :{
        1: [a], 2: [a], 3: [a], 4: [a], 5: [a], 6: [a]
        }
    END_OF_JOB.ts --> Absent
    """

    path = (
        "s3://tornasole-testing/has_step_scenarios/single-writer-all-steps-written-incomplete-job"
    )
    trial = create_trial(path)
    num_workers = len(trial.workers())
    assert num_workers == 1
    assert trial.loaded_all_steps is False
    available_steps = trial.available_steps()
    assert available_steps == [0, 1, 2, 3, 4, 5, 6]
    assert trial.has_passed_step(3) == StepState.AVAILABLE
    assert trial.has_passed_step(8) == StepState.NOT_YET_AVAILABLE
    assert (
        trial.last_index_token
        == "has_step_scenarios/single-writer-all-steps-written-incomplete-job/index/000000000/000000000006_worker_0.json"
    )


@pytest.mark.slow
def test_single_writer_not_all_steps_written_complete_job():
    """Test Scenario Description"
     workers : [a]
     steps :{
        1: [a], 2: [a], 3: [a], 4: [], 5: [a], 6: [a]
        }
    END_OF_JOB.ts --> Present
    """

    path = "s3://tornasole-testing/has_step_scenarios/single-writer-not-all-steps-written-complete"
    trial = create_trial(path)
    num_workers = len(trial.workers())
    assert num_workers == 1
    assert trial.loaded_all_steps is True
    available_steps = trial.available_steps()
    assert available_steps == [0, 1, 2, 3, 5, 6]  # step 4 is missing
    assert trial.has_passed_step(3) == StepState.AVAILABLE
    assert trial.has_passed_step(4) == StepState.UNAVAILABLE
    assert trial.has_passed_step(8) == StepState.UNAVAILABLE
    assert (
        trial.last_index_token
        == "has_step_scenarios/single-writer-not-all-steps-written-complete/index/000000000/000000000006_worker_0.json"
    )


@pytest.mark.slow
def test_single_writer_not_all_steps_written_incomplete_job():
    """Test Scenario Description"
     workers : [a]
     steps :{
        1: [a], 2: [a], 3: [a], 4: [], 5: [a], 6: [a]
        }
    END_OF_JOB.ts --> Absent
    """

    path = (
        "s3://tornasole-testing/has_step_scenarios/single-writer-not-all-steps-written-incomplete"
    )
    trial = create_trial(path)
    num_workers = len(trial.workers())
    assert num_workers == 1
    assert trial.loaded_all_steps is False
    available_steps = trial.available_steps()
    assert available_steps == [0, 1, 2, 3, 5, 6]  # step 4 is missing
    assert trial.has_passed_step(3) == StepState.AVAILABLE
    assert trial.has_passed_step(4) == StepState.UNAVAILABLE
    assert trial.has_passed_step(8) == StepState.NOT_YET_AVAILABLE
    assert (
        trial.last_index_token
        == "has_step_scenarios/single-writer-not-all-steps-written-incomplete/index/000000000/000000000006_worker_0.json"
    )


@pytest.mark.slow
def test_three_writers_all_steps_written_complete_job():
    """Test Scenario Description"
     workers : [a,b,c]
     steps :{
        1: [a,b,c], 2: [a,b,c], 3: [a,b,c], 4: [a,b,c], 5: [a,b,c], 6: [a,b,c]
        }
    END_OF_JOB.ts --> Present
    """

    path = "s3://tornasole-testing/has_step_scenarios/three-writers-allsteps-written-complete-job"
    trial = create_trial(path)
    num_workers = len(trial.workers())
    assert num_workers == 3
    assert trial.loaded_all_steps is True
    available_steps = trial.available_steps()
    assert available_steps == [0, 1, 2, 3, 4, 5, 6]
    assert trial.last_complete_step == 6
    assert trial.has_passed_step(3) == StepState.AVAILABLE
    assert trial.has_passed_step(4) == StepState.AVAILABLE
    assert trial.has_passed_step(8) == StepState.UNAVAILABLE
    assert (
        trial.last_index_token
        == "has_step_scenarios/three-writers-allsteps-written-complete-job/index/000000000/000000000006_worker_2.json"
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

    path = (
        "s3://tornasole-testing/has_step_scenarios/three-writers-all-steps-written-incomplete-job"
    )
    trial = create_trial(path)
    num_workers = len(trial.workers())
    assert num_workers == 3
    assert trial.loaded_all_steps is False
    available_steps = trial.available_steps()
    assert available_steps == [0, 1, 2, 3, 4, 5, 6]
    assert trial.last_complete_step == 6
    assert trial.has_passed_step(3) == StepState.AVAILABLE
    assert trial.has_passed_step(4) == StepState.AVAILABLE
    assert trial.has_passed_step(8) == StepState.NOT_YET_AVAILABLE
    assert (
        trial.last_index_token
        == "has_step_scenarios/three-writers-all-steps-written-incomplete-job/index/000000000/000000000006_worker_2.json"
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

    path = (
        "s3://tornasole-testing/has_step_scenarios/three-writers-not-all-steps-written-complete-job"
    )
    trial = create_trial(path)
    num_workers = len(trial.workers())
    assert num_workers == 3
    assert trial.loaded_all_steps is True
    available_steps = trial.available_steps()
    assert available_steps == [0, 1, 2, 3, 4, 5, 6]
    assert trial.has_passed_step(3) == StepState.AVAILABLE
    assert trial.last_complete_step == 2
    assert trial.has_passed_step(4) == StepState.AVAILABLE
    assert trial.has_passed_step(6) == StepState.AVAILABLE
    assert trial.has_passed_step(8) == StepState.UNAVAILABLE
    assert (
        trial.last_index_token
        == "has_step_scenarios/three-writers-not-all-steps-written-complete-job/index/000000000/000000000002_worker_2.json"
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
    path = "s3://tornasole-testing/has_step_scenarios/three-writers-not-all-steps-written-incomplete-job"
    trial = create_trial(path)
    num_workers = len(trial.workers())
    assert num_workers == 3
    assert trial.loaded_all_steps is False
    available_steps = trial.available_steps()
    assert available_steps == [0, 1, 2, 3, 4, 5, 6]
    assert trial.has_passed_step(2) == StepState.AVAILABLE
    assert trial.last_complete_step == 2
    assert trial.has_passed_step(4) == StepState.NOT_YET_AVAILABLE
    assert trial.has_passed_step(6) == StepState.NOT_YET_AVAILABLE
    assert trial.has_passed_step(8) == StepState.NOT_YET_AVAILABLE
    assert (
        trial.last_index_token
        == "has_step_scenarios/three-writers-not-all-steps-written-incomplete-job/index/000000000/000000000002_worker_2.json"
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
    path = "s3://tornasole-testing/has_step_scenarios/three-writers-not-all-steps-written-but-later-step-written-incomplete-job"
    trial = create_trial(path)
    num_workers = len(trial.workers())
    assert num_workers == 3
    assert trial.loaded_all_steps is False
    available_steps = trial.available_steps()
    assert available_steps == [0, 1, 2, 3, 4, 5, 6]
    assert trial.has_passed_step(2) == StepState.AVAILABLE
    assert trial.last_complete_step == 6
    assert trial.has_passed_step(4) == StepState.AVAILABLE
    assert trial.has_passed_step(6) == StepState.AVAILABLE
    assert trial.has_passed_step(8) == StepState.NOT_YET_AVAILABLE
    assert (
        trial.last_index_token
        == "has_step_scenarios/three-writers-not-all-steps-written-but-later-step-written-incomplete-job/index/000000000/000000000006_worker_2.json"
    )


@pytest.mark.slow
def test_three_writers_not_all_steps_written_but_later_step_written_complete_job():
    """Test Scenario Description"
     workers : [a,b,c]
     steps :{
        1: [a,b,c], 2: [a,b,c], 3: [a,c], 4: [a,c], 5: [a,c], 6: [a,b,c]
        }
    END_OF_JOB.ts --> Present
    """
    path = "s3://tornasole-testing/has_step_scenarios/three-writers-not-all-steps-written-but-later-step-written-complete-job"
    trial = create_trial(path)
    num_workers = len(trial.workers())
    assert num_workers == 3
    assert trial.loaded_all_steps is True
    available_steps = trial.available_steps()
    assert available_steps == [0, 1, 2, 3, 4, 5, 6]
    assert trial.has_passed_step(2) == StepState.AVAILABLE
    assert trial.last_complete_step == 6
    assert trial.has_passed_step(4) == StepState.AVAILABLE
    assert trial.has_passed_step(6) == StepState.AVAILABLE
    assert trial.has_passed_step(8) == StepState.UNAVAILABLE
    assert (
        trial.last_index_token
        == "has_step_scenarios/three-writers-not-all-steps-written-but-later-step-written-complete-job/index/000000000/000000000006_worker_2.json"
    )


@pytest.mark.slow
def test_override_if_too_many_steps_skipped():
    """Test Scenario Description"
     workers : [a,b,c]
     steps :{
        0: [a], 2: [a], 3: [a], 4: [a], 5: [a], ..., 19: [a]
        }
    END_OF_JOB.ts --> Absent
    Note: This test needs TORNASOLE_CONFIG_MAX_WAIT_STEPS to be set to 20 to pass

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

    path = "s3://tornasole-testing/has_step_scenarios/too-many-steps-skipped"
    trial = create_trial(path)
    assert trial.last_complete_step == 4
    assert (
        trial.last_index_token
        == "has_step_scenarios/too-many-steps-skipped/index/000000000/000000000004_worker_2.json"
    )
    num_workers = len(trial.workers())
    assert num_workers == 3
    assert trial.last_complete_step == 9
    assert (
        trial.last_index_token
        == "has_step_scenarios/too-many-steps-skipped/index/000000000/000000000009_worker_2.json"
    )
    assert trial.loaded_all_steps is False
    available_steps = trial.available_steps()
    assert available_steps == [i for i in range(20)]
    assert trial.last_complete_step == 9
    assert (
        trial.last_index_token
        == "has_step_scenarios/too-many-steps-skipped/index/000000000/000000000009_worker_2.json"
    )
    trial.tensors()
    assert trial.last_complete_step == 9
    assert (
        trial.last_index_token
        == "has_step_scenarios/too-many-steps-skipped/index/000000000/000000000009_worker_2.json"
    )
    trial.tensors()
    trial.tensors()
    assert trial.last_complete_step == 9
    assert (
        trial.last_index_token
        == "has_step_scenarios/too-many-steps-skipped/index/000000000/000000000009_worker_2.json"
    )


os.environ["TORNASOLE_CONFIG_MAX_WAIT_STEPS"] = "1000"  # Reset to default
