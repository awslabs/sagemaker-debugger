import os

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
    all_steps = trial.steps(show_incomplete_steps=True)
    completed_steps = trial.steps()
    assert all_steps == [0, 1, 2, 3, 4, 5, 6]
    assert completed_steps == all_steps
    assert trial.has_passed_step(3) == StepState.AVAILABLE
    assert trial.has_passed_step(8) == StepState.UNAVAILABLE
    assert (
        trial.last_index_token
        == "has_step_scenarios/single-writer-all-steps-written-complete-job/index/000000000/000000000006_worker_0.json"
    )
    assert trial.last_complete_step == 6


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
    all_steps = trial.steps(show_incomplete_steps=True)
    completed_steps = trial.steps()
    assert all_steps == [0, 1, 2, 3, 4, 5, 6]
    assert all_steps == completed_steps
    assert trial.has_passed_step(3) == StepState.AVAILABLE
    assert trial.has_passed_step(8) == StepState.NOT_YET_AVAILABLE
    assert (
        trial.last_index_token
        == "has_step_scenarios/single-writer-all-steps-written-incomplete-job/index/000000000/000000000006_worker_0.json"
    )
    assert trial.last_complete_step == 6


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
    all_steps = trial.steps(show_incomplete_steps=True)
    completed_steps = trial.steps()
    assert all_steps == [0, 1, 2, 3, 5, 6]  # step 4 is missing
    assert completed_steps == all_steps
    assert trial.has_passed_step(3) == StepState.AVAILABLE
    assert trial.has_passed_step(4) == StepState.UNAVAILABLE
    assert trial.has_passed_step(8) == StepState.UNAVAILABLE
    assert (
        trial.last_index_token
        == "has_step_scenarios/single-writer-not-all-steps-written-complete/index/000000000/000000000006_worker_0.json"
    )
    assert trial.last_complete_step == 6


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
    all_steps = trial.steps(show_incomplete_steps=True)
    completed_steps = trial.steps()
    assert all_steps == [0, 1, 2, 3, 5, 6]  # step 4 is missing
    assert completed_steps == all_steps
    assert trial.has_passed_step(3) == StepState.AVAILABLE
    assert trial.has_passed_step(4) == StepState.UNAVAILABLE
    assert trial.has_passed_step(8) == StepState.NOT_YET_AVAILABLE
    assert (
        trial.last_index_token
        == "has_step_scenarios/single-writer-not-all-steps-written-incomplete/index/000000000/000000000006_worker_0.json"
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

    path = "s3://tornasole-testing/has_step_scenarios/three-writers-allsteps-written-complete-job"
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
        == "has_step_scenarios/three-writers-not-all-steps-written-but-later-step-written-incomplete-job/index/000000000/000000000006_worker_2.json"
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
    path = "s3://tornasole-testing/has_step_scenarios/three_writers_one_step_missing_but_later_steps_written_incomplete_job"
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
        == "has_step_scenarios/three_writers_one_step_missing_but_later_steps_written_incomplete_job/index/000000000/000000000006_worker_2.json"
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
    path = "s3://tornasole-testing/has_step_scenarios/three_writers_one_step_missing_but_later_steps_written_partially_incomplete_job"
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
        == "has_step_scenarios/three_writers_one_step_missing_but_later_steps_written_partially_incomplete_job/index/000000000/000000000002_worker_2.json"
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
    path = "s3://tornasole-testing/has_step_scenarios/three_writers_one_step_missing_but_later_steps_written_partially_complete_job"
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
        == "has_step_scenarios/three_writers_one_step_missing_but_later_steps_written_partially_complete_job/index/000000000/000000000002_worker_2.json"
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
    Note: This test needs TORNASOLE_INCOMPLETE_STEP_WAIT_WINDOW to be set to 10 to pass

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
    os.environ["TORNASOLE_INCOMPLETE_STEP_WAIT_WINDOW"] = "10"

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
    all_steps = trial.steps(show_incomplete_steps=True)
    assert all_steps == [i for i in range(20)]
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

    del os.environ["TORNASOLE_INCOMPLETE_STEP_WAIT_WINDOW"]


def test_partially_written_tensors():
    """
    The trial data accessed by this test was generated with the following hook configs:

        ts.get_collection("gradients").save_config = {
            ts.modes.TRAIN: ts.SaveConfigMode(save_interval=1, start_step=5, end_step=10),
        }


        ts.get_collection("losses").save_config = {
            ts.modes.TRAIN: ts.SaveConfigMode(save_interval=1, end_step=5),
        }

        hook = ts.TornasoleHook(
                    ...,
                include_collections=["weights", "gradients", "losses"],
                save_config=ts.SaveConfig(save_interval=1, end_step=10),
        )

        The training job was executed by two workers.

        The data was manipulated post completion in the following ways:

            1. END_OF_JOB.ts was deleted.
            2. Index_files for steps: [3, 4, 8, 9] were deleted for one worker
    """

    path = "s3://tornasole-testing/has_step_scenarios/partially_written_tensors/"
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
