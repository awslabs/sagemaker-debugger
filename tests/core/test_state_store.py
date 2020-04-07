# Standard Library
import os
import shutil
import time

# First Party
from smdebug.core.state_store import StateStore


def setup_test1():
    try:
        shutil.rmtree("checkpoints_test_dir/")
    except:
        pass
    os.mkdir("checkpoints_test_dir/")
    os.mkdir("checkpoints_test_dir/subdir1")
    os.mkdir("checkpoints_test_dir/subdir2")
    dir_path = os.path.abspath("checkpoints_test_dir")
    with open(dir_path + "/metadata.json", "w") as f:
        pass
    with open(dir_path + "/config_test.json", "w") as f:
        pass
    with open(dir_path + "/subdir1/checkpoint_test1.txt", "w") as f:
        pass
    with open(dir_path + "/subdir2/checkpoint_test2.txt", "w") as f:
        pass
    return dir_path


def setup_test2():
    try:
        shutil.rmtree("checkpoints_test_dir/")
    except:
        pass
    os.mkdir("checkpoints_test_dir/")
    dir_path = os.path.abspath("checkpoints_test_dir")
    with open(dir_path + "/metadata.json", "w") as f:
        pass
    return dir_path


def test_is_checkpoint_updated():
    s1 = StateStore()
    s1._checkpoint_update_timestamp = time.time()
    s1._checkpoint_dir = setup_test1()
    # the checkpoint update time is greater than _checkpoint_update_timestamp. is_checkpoint_updated should return true.
    assert s1.is_checkpoint_updated()

    s1._checkpoint_update_timestamp = time.time()
    # the checkpoint update time is lesser than _checkpoint_update_timestamp. is_checkpoint_updated should return false.
    assert not s1.is_checkpoint_updated()
    shutil.rmtree(s1._checkpoint_dir)

    s2 = StateStore()
    s2._checkpoint_update_timestamp = time.time()
    s2._checkpoint_dir = setup_test2()
    # setup_test2 has only metadata.json in the checkpoints_dir. So no checkpoints file was created or updated. It should return false.
    assert not s2.is_checkpoint_updated()

    s2._checkpoint_update_timestamp = time.time()
    # setup_test2 has only metadata.json in the checkpoints_dir. So no checkpoints file was created or updated. It should return false.
    assert not s2.is_checkpoint_updated()
    shutil.rmtree(s2._checkpoint_dir)
