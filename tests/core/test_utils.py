# Third Party
# Standard Library
import os
import shutil
import tempfile
import time
from multiprocessing import Manager, Process
from os import makedirs

import boto3
import pytest
import requests

# First Party
from smdebug.core.access_layer import (
    DEFAULT_GRACETIME_FOR_RULE_STOP_SEC,
    ENV_RULE_STOP_SIGNAL_FILENAME,
    check_dir_exists,
    is_rule_signalled_gracetime_passed,
)
from smdebug.core.collection_manager import CollectionManager
from smdebug.core.config_constants import (
    PAPERMILL_EXECUTION_ENV_VAR,
    PROFILER_REPORT_VERSION,
    PROFILER_TELEMETRY_URL,
)
from smdebug.core.index_reader import ReadIndexFilesCache
from smdebug.core.json_config import (
    DEFAULT_SAGEMAKER_OUTDIR,
    add_collections_to_manager,
    collect_hook_config_params,
    get_include_collections,
    get_json_config_as_dict,
)
from smdebug.core.locations import IndexFileLocationUtils
from smdebug.core.utils import (
    SagemakerSimulator,
    _prepare_telemetry_url,
    get_aws_region_from_processing_job_arn,
    is_first_process,
    is_s3,
    setup_profiler_report,
)


def test_normal():
    rval = is_s3("a/b/c")
    assert not rval[0]


def test_s3():
    rval = is_s3("s3://a/b")
    assert rval[0]
    assert rval[1] == "a"
    assert rval[2] == "b"


def test_s3_noprefix():
    rval = is_s3("s3://a")
    assert rval[0]
    assert rval[1] == "a"
    assert rval[2] == ""


def test_s3_noprefix2():
    rval = is_s3("s3://a/")
    assert rval[0]
    assert rval[1] == "a"
    assert rval[2] == ""


def test_check_dir_not_exists_local():
    check_dir_exists("/home/ubuntu/asasdas")


def test_check_dir_exists():
    try:
        check_dir_exists("/home/ubuntu/")
        assert False
    except Exception as e:
        pass


def test_check_dir_not_exists_s3():
    check_dir_exists("s3://smdebug-testing/resources/doesnotexist")


def test_check_dir_exists_s3():
    # This file should exist in the bucket for proper testing
    check_dir_exists("s3://smdebug-testing/resources/exists")


def setup_rule_stop_file(temp_file, time_str, monkeypatch, write=True):
    dir = os.path.dirname(temp_file.name)
    rel_filename = os.path.relpath(temp_file.name, start=dir)
    if write is True:
        # write timestamp in temp file
        temp_file.write(str(time_str))
        temp_file.flush()
    monkeypatch.setenv(ENV_RULE_STOP_SIGNAL_FILENAME, rel_filename)


def test_is_rule_signalled_gracetime_not_passed(monkeypatch):
    temp_file = tempfile.NamedTemporaryFile(mode="w+")
    time_str = str(int(time.time()))
    setup_rule_stop_file(temp_file, time_str, monkeypatch)
    dir = os.path.dirname(temp_file.name)
    assert is_rule_signalled_gracetime_passed(dir) is False


def test_is_rule_signalled_gracetime_passed(monkeypatch):
    temp_file = tempfile.NamedTemporaryFile(mode="w+")
    time_str = str(int(time.time() - 2 * DEFAULT_GRACETIME_FOR_RULE_STOP_SEC))
    setup_rule_stop_file(temp_file, time_str, monkeypatch)
    dir = os.path.dirname(temp_file.name)
    assert is_rule_signalled_gracetime_passed(dir) is True


def test_is_rule_signalled_no_env_var_set(monkeypatch):
    assert is_rule_signalled_gracetime_passed("/fake-file") is False


def test_is_rule_signalled_no_signal_file(monkeypatch):
    temp_file = tempfile.NamedTemporaryFile(mode="w+")
    time_str = str(int(time.time() - 2 * DEFAULT_GRACETIME_FOR_RULE_STOP_SEC))
    setup_rule_stop_file(temp_file, time_str, monkeypatch, write=False)
    dir = os.path.dirname(temp_file.name)
    # env variable is set, remove the file.
    temp_file.close()
    assert is_rule_signalled_gracetime_passed(dir) is False


def test_is_rule_signalled_invalid_gracetime(monkeypatch):
    temp_file = tempfile.NamedTemporaryFile(mode="w+")
    setup_rule_stop_file(temp_file, "Invalid_time", monkeypatch)
    dir = os.path.dirname(temp_file.name)
    assert is_rule_signalled_gracetime_passed(dir) is True


@pytest.mark.skip(reason="It's unclear what this is testing.")
def test_check_dir_not_exists():
    with pytest.raises(Exception):
        check_dir_exists("s3://smdebug-testing")


def test_index_files_cache():
    """
    Test to verify that the index file cache is behaving as it should.
        1. The cache should not save elements already present
        2. The cache should remove its old elements when it attempts to save more elements
        that its set limit.
    """
    index_file_cache = ReadIndexFilesCache()
    index_file_cache.add("file_1", None)
    index_file_cache.add("file_1", None)
    assert len(index_file_cache.lookup_set) == 1
    assert index_file_cache.has_not_read("file_1") is False
    assert index_file_cache.has_not_read("file_2") is True
    index_file_cache.add("file_2", None)
    index_file_cache.add("file_3", None)
    index_file_cache.add("file_4", None)
    assert len(index_file_cache.lookup_set) == 4

    # Test cache eviction logic

    index_file_cache.cache_limit = 2  # override cache limit
    index_file_cache.add("file_5", "file_1")
    assert len(index_file_cache.lookup_set) == 5  # No elements evicted
    index_file_cache.add("file_6", "file_4")
    assert (
        len(index_file_cache.lookup_set) == 3
    )  # Elements in the cache will be file_4, file_5, file_6


def test_index_files_cache_insert_many_elements_in_the_first_read():
    cache = ReadIndexFilesCache()
    cache.cache_limit = 5
    elements = ["a", "b", "c", "d", "e", "f", "g", "h"]
    for e in elements:
        cache.add(e, None)

    # No files should be evicted because start_after_key has not been set
    assert len(cache.lookup_set) == len(elements)


def test_get_prefix_from_index_file():
    local_index_filepath = "/opt/ml/testing/run_1/index/000000000/000000000000_worker_0.json"
    prefix = IndexFileLocationUtils.get_prefix_from_index_file(local_index_filepath)

    assert prefix == "/opt/ml/testing/run_1"

    s3_index_filepath = (
        "s3://bucket-that-does-not-exist/run_1/index/000000000/000000000000_worker_0.json"
    )
    prefix = IndexFileLocationUtils.get_prefix_from_index_file(s3_index_filepath)

    assert prefix == "s3://bucket-that-does-not-exist/run_1"


def test_json_params():
    params_dict = get_json_config_as_dict(
        json_config_path="tests/core/json_configs/all_params.json"
    )
    hook_params = collect_hook_config_params(params_dict)
    include_collections = get_include_collections(params_dict)
    coll_manager = CollectionManager()
    add_collections_to_manager(coll_manager, params_dict, hook_params)
    assert hook_params["include_workers"] == "one"
    assert hook_params["save_all"] is True
    assert coll_manager.get("weights").save_histogram is False
    assert coll_manager.get("gradients").save_histogram is False
    assert "weights" in include_collections
    assert "gradients" in include_collections
    assert len(include_collections) == 2
    assert hook_params["export_tensorboard"] == True
    assert hook_params["tensorboard_dir"] == "/tmp/tensorboard"


def test_json_params_sagemaker():
    with SagemakerSimulator() as sim:
        params_dict = get_json_config_as_dict(
            json_config_path="tests/core/json_configs/all_params.json"
        )
        hook_params = collect_hook_config_params(params_dict)
        include_collections = get_include_collections(params_dict)
        coll_manager = CollectionManager()
        add_collections_to_manager(coll_manager, params_dict, hook_params)
        assert hook_params["include_workers"] == "one"
        assert hook_params["save_all"] is True
        assert coll_manager.get("weights").save_histogram is False
        assert coll_manager.get("gradients").save_histogram is False
        assert "weights" in include_collections
        assert "gradients" in include_collections
        assert len(include_collections) == 2
        assert hook_params["export_tensorboard"] == True
        assert hook_params["tensorboard_dir"] == sim.tensorboard_dir


@pytest.mark.parametrize("dir", [True, False])
def test_is_first_process(dir):
    s3_path = "s3://this/is/a/valid/path"
    assert is_first_process(s3_path)

    # This section tests local path
    for _ in range(10):
        helper_test_is_first_process(dir)


def helper_test_is_first_process(dir):
    temp_dir = tempfile.TemporaryDirectory()
    path = temp_dir.name
    shutil.rmtree(path, ignore_errors=True)
    if dir:
        makedirs(temp_dir.name)
    process_list = []

    def helper(fn, arg, shared_list):
        shared_list.append(fn(arg))

    manager = Manager()
    results = manager.list()
    for i in range(100):
        p = Process(target=helper, args=(is_first_process, path, results))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    assert results.count(True) == 1, f"Failed for path: {path}"


def _get_all_aws_regions():
    ec2 = boto3.client("ec2")
    response = ec2.describe_regions()
    regions = [r["RegionName"] for r in response["Regions"]]
    return regions


@pytest.fixture()
def test_arn():
    arn = "arn:aws:sagemaker:{region}:012345678910:processing-job/random-test-arn"
    return arn


def fake_get(*args, **kwargs):
    # The goal of fake GET is to check if this function was indeed called or not
    assert False


@pytest.mark.parametrize("region", _get_all_aws_regions())
def test_telemetry_url_preparation(test_arn, region):
    arn_with_region = test_arn.format(region=region)
    assert get_aws_region_from_processing_job_arn(arn_with_region) == region
    url = _prepare_telemetry_url(arn_with_region)
    assert url == PROFILER_TELEMETRY_URL.format(
        region=region
    ) + "/?x-artifact-id={report_version}&x-arn={arn}".format(
        report_version=PROFILER_REPORT_VERSION, arn=arn_with_region
    )


def test_setup_profiler_report(monkeypatch, test_arn):
    test_arn = test_arn.format(region="us-east-1")
    monkeypatch.setattr(requests, "get", fake_get)

    # setup_profiler_report is expected to be executed when called with a correct ARN
    with pytest.raises(AssertionError):
        setup_profiler_report(test_arn)

    with pytest.raises(AssertionError):
        setup_profiler_report(test_arn, opt_out=False)

    # setup_profiler_report is NOT expected to be executed when called with opt_out=True
    setup_profiler_report(test_arn, opt_out=True)

    # setup_profiler_report is NOT expected to be executed when env PAPERMILL_EXECUTION is set
    monkeypatch.setenv(PAPERMILL_EXECUTION_ENV_VAR, "1")
    setup_profiler_report(test_arn, opt_out=False)
