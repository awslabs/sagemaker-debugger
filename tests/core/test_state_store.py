# Standard Library
import json
import os
import shutil

# First Party
from smdebug.core.state_store import StateStore


def setup_test():
    try:
        shutil.rmtree("checkpoints_test_dir/")
    except:
        pass
    # create the checkpoints directory
    os.mkdir("checkpoints_test_dir/")
    dir_path = os.path.abspath("checkpoints_test_dir")

    # create the config file and set the corresponding environment variable.
    mock_config = {"LocalPath": dir_path}
    with open("mock_config.json", "w") as f:
        json.dump(mock_config, f)
    os.environ["CHECKPOINT_CONFIG_FILE_PATH"] = os.path.abspath("mock_config.json")

    # create the metadata file inside the checkpoints directory.
    mock_metadata = [
        {
            "training-run": "",
            "latest-global-step-saved": "",
            "latest-global-step-seen": "",
            "latest-mode-step": "",
        }
    ]
    with open(dir_path + "/metadata.json", "w") as f:
        json.dump(mock_metadata, f)
    return dir_path, os.path.abspath("mock_config.json")


def cleanup(checkpoints_dir_path, config_path):
    shutil.rmtree(checkpoints_dir_path)
    os.remove(config_path)


def test_is_checkpoint_updated():
    s1 = StateStore()
    # There is no checkpoint_dir. is_checkpoint_updated should return False.
    assert s1.is_checkpoint_updated() is False

    # call setup_test to create checkpoints_dir and metadata file.
    checkpoints_dir_path, config_path = setup_test()
    s2 = StateStore()
    # checkpoints_dir only has metadata.json. So no checkpoints file was created or updated. It should return false.
    assert s2.is_checkpoint_updated() is False

    s2.update_state("test-state1")
    # checkpoints_dir still has only metadata.json. It should return false.
    assert s2.is_checkpoint_updated() is False

    os.mkdir(checkpoints_dir_path + "/subdir1")
    with open(checkpoints_dir_path + "/subdir1/checkpoint_test1.txt", "w") as f:
        pass
    # the checkpoint update time is greater than _checkpoint_update_timestamp. is_checkpoint_updated should return true.
    assert s2.is_checkpoint_updated()

    s2.update_state("test-state")
    # the state_file has been updated. The lastest checkpoint update time is lesser than _checkpoint_update_timestamp.
    # is_checkpoint_updated should return false.
    assert s2.is_checkpoint_updated() is False

    with open(checkpoints_dir_path + "/subdir1/checkpoint_test1.txt", "a") as f:
        f.write("test-string")
    # A checkpoint file has been updated. The checkpoint update time is greater than _checkpoint_update_timestamp.
    # is_checkpoint_updated should return true.
    assert s2.is_checkpoint_updated()

    s2.update_state("test-state1")
    os.mkdir(checkpoints_dir_path + "/subdir2")
    with open(checkpoints_dir_path + "/subdir2/checkpoint_test2.txt", "w") as f:
        pass
    # A new checkpoint file has been created. The checkpoint update time is greater than _checkpoint_update_timestamp.
    # is_checkpoint_updated should return true.
    assert s2.is_checkpoint_updated()
    cleanup(checkpoints_dir_path, config_path)
