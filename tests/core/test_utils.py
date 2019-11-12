# Third Party
import pytest

# First Party
from smdebug.core.access_layer import check_dir_exists
from smdebug.core.collection_manager import CollectionManager
from smdebug.core.index_reader import ReadIndexFilesCache
from smdebug.core.json_config import DEFAULT_SAGEMAKER_OUTDIR, collect_config_params
from smdebug.core.locations import IndexFileLocationUtils
from smdebug.core.utils import is_s3


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


def test_check_dir_exists_no_local():
    check_dir_exists("/home/ubuntu/asasdas")


def test_check_dir_exists():
    try:
        check_dir_exists("/home/ubuntu/")
        assert False
    except Exception as e:
        pass


def test_check_dir_exists_no_s3():
    check_dir_exists("s3://tornasole-testing/pleasedontexist")


def test_check_dir_exists_s3():
    try:
        check_dir_exists("s3://tornasole-binaries-use1/tornasole_tf/")
        assert False
    except Exception as e:
        pass


def test_check_dir_exists_no():
    try:
        check_dir_exists("s3://tornasole-binaries-use1")
        assert False
    except Exception as e:
        pass


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
    local_index_filepath = (
        "/opt/ml/tornasole-testing/run_1/index/000000000/000000000000_worker_0.json"
    )
    prefix = IndexFileLocationUtils.get_prefix_from_index_file(local_index_filepath)

    assert prefix == "/opt/ml/tornasole-testing/run_1"

    s3_index_filepath = "s3://tornasole-testing/run_1/index/000000000/000000000000_worker_0.json"
    prefix = IndexFileLocationUtils.get_prefix_from_index_file(s3_index_filepath)

    assert prefix == "s3://tornasole-testing/run_1"


@pytest.mark.skip(reason="If no config file is found, then SM doesn't want a SessionHook")
def test_collect_config_params():
    params = collect_config_params(collection_manager=CollectionManager())
    assert params["out_dir"] == DEFAULT_SAGEMAKER_OUTDIR
