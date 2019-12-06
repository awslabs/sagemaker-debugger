# Third Party
import pytest

# First Party
from smdebug.core.access_layer import check_dir_exists
from smdebug.core.collection_manager import CollectionManager
from smdebug.core.index_reader import ReadIndexFilesCache
from smdebug.core.json_config import (
    DEFAULT_SAGEMAKER_OUTDIR,
    add_collections_to_manager,
    collect_hook_config_params,
    get_include_collections,
    get_json_config_as_dict,
)
from smdebug.core.locations import IndexFileLocationUtils
from smdebug.core.utils import SagemakerSimulator, is_s3


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
