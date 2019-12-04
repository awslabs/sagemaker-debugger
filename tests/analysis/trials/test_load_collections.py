# Standard Library

# Third Party
import pytest

# First Party
from smdebug.exceptions import MissingCollectionFiles
from smdebug.trials import create_trial


@pytest.mark.slow
def test_load_collection_files_from_completed_job():
    """
    Number of collection files : 2001
    Training_has_ended.ts : Present

    All the collection files have been written in the test dataset
    and the training_has_ended file is present
    :return:
    """
    path = "s3://smdebug-testing/outputs/collection-tests/all-collection-files-present/"
    try:
        trial = create_trial(path)
    except MissingCollectionFiles:
        assert False
    assert len(trial.workers()) == 2001


@pytest.mark.slow
def test_load_collection_files_from_completed_job_with_missing_files():
    """
    Number of collection files : 1446
    Training_has_ended.ts : Present

    Some of the collection files have been removed in the test dataset.
    The number of expected collection files is supposed to 2001
    but the training_has_ended file is present so we stop waiting
    :return:
    """
    path = "s3://smdebug-testing/outputs/collection-tests/collection-files-missing/"
    try:
        trial = create_trial(path)
        assert False
    except MissingCollectionFiles:
        assert True


@pytest.mark.slow
def test_load_collection_files_from_incomplete_job():
    """
    Number of collection files : 2001
    Training_has_ended.ts : Absent

    All the collection files have been written in the test dataset
    and the training_has_ended file is absent


    :return:
    """
    path = (
        "s3://smdebug-testing/outputs/collection-tests/all-collection-files-present-job-incomplete/"
    )
    try:
        trial = create_trial(path)
    except MissingCollectionFiles:
        assert False
    assert len(trial.workers()) == 2001
