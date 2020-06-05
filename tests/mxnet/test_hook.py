# Standard Library
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

# First Party
from smdebug import SaveConfig
from smdebug.core.access_layer.utils import has_training_ended
from smdebug.core.json_config import CONFIG_FILE_PATH_ENV_STR
from smdebug.mxnet.hook import Hook as t_hook
from smdebug.profiler.profiler_constants import DEFAULT_PREFIX

# Local
from .mnist_gluon_model import run_mnist_gluon_model


def test_hook():
    save_config = SaveConfig(save_steps=[0, 1, 2, 3])
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    out_dir = "/tmp/newlogsRunTest/" + run_id
    hook = t_hook(out_dir=out_dir, save_config=save_config)
    assert has_training_ended(out_dir) == False
    run_mnist_gluon_model(
        hook=hook, num_steps_train=10, num_steps_eval=10, register_to_loss_block=True
    )
    shutil.rmtree(out_dir)


def test_hook_from_json_config():
    out_dir = "/tmp/newlogsRunTest1/test_hook_from_json_config"
    shutil.rmtree(out_dir, True)
    os.environ[
        CONFIG_FILE_PATH_ENV_STR
    ] = "tests/mxnet/test_json_configs/test_hook_from_json_config.json"
    hook = t_hook.create_from_json_file()
    assert has_training_ended(out_dir) == False
    run_mnist_gluon_model(
        hook=hook, num_steps_train=10, num_steps_eval=10, register_to_loss_block=True
    )
    shutil.rmtree(out_dir, True)


def test_hook_from_json_config_full():
    out_dir = "/tmp/newlogsRunTest2/test_hook_from_json_config_full"
    shutil.rmtree(out_dir, True)
    os.environ[
        CONFIG_FILE_PATH_ENV_STR
    ] = "tests/mxnet/test_json_configs/test_hook_from_json_config_full.json"
    hook = t_hook.create_from_json_file()
    assert has_training_ended(out_dir) == False
    run_mnist_gluon_model(
        hook=hook, num_steps_train=10, num_steps_eval=10, register_to_loss_block=True
    )
    shutil.rmtree(out_dir, True)


def test_hook_timeline_file_write(set_up_smprofiler_config_path, out_dir):
    """
    This test is meant to test TimelineFileWriter through a MXNet hook.
    """
    hook = t_hook(out_dir=out_dir)

    for i in range(1, 11):
        n = "event" + str(i)
        hook.record_trace_events(
            training_phase="MXNet_TimelineFileWriteTest",
            op_name=n,
            step_num=i,
            timestamp=time.time(),
        )

    # need to explicitly close hook for the test here so that the JSON file is written and
    # can be read back below.
    # In training scripts, this is not necessary as _cleanup will take care of closing the trace file.
    hook.close()

    files = []
    for path in Path(out_dir + "/" + DEFAULT_PREFIX).rglob("*.json"):
        files.append(path)

    assert len(files) == 1

    with open(files[0]) as timeline_file:
        events_dict = json.load(timeline_file)

    assert events_dict
