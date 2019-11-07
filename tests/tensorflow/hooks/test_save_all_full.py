from .utils import *
from tornasole.tensorflow import reset_collections, get_collections, CollectionManager, Collection
import glob
import shutil
from tornasole.core.reader import FileReader
from tornasole.core.json_config import TORNASOLE_CONFIG_FILE_PATH_ENV_STR
from tornasole.core.config_constants import TORNASOLE_DEFAULT_COLLECTIONS_FILE_NAME
from tornasole.core.utils import get_path_to_collections


def test_save_all_full(hook=None, trial_dir=None):
    run_id = "trial_" + datetime.now().strftime("%Y%m%d-%H%M%S%f")
    if trial_dir is None:
        trial_dir = os.path.join(TORNASOLE_TF_HOOK_TESTS_DIR, run_id)
    tf.reset_default_graph()
    hook_created = False
    if hook is None:
        reset_collections()
        hook = TornasoleHook(
            out_dir=trial_dir, save_all=True, save_config=SaveConfig(save_interval=2)
        )
        hook_created = True

    simple_model(hook)
    files = get_collection_files(trial_dir)
    dirs, _ = get_dirs_files(os.path.join(trial_dir, "events"))

    coll = get_collections()
    assert all(
        [x in coll.keys() for x in ["all", "weights", "gradients", "losses", "optimizer_variables"]]
    )
    assert len(coll["weights"].tensor_names) == 1
    assert len(coll["gradients"].tensor_names) == 1
    assert len(coll["losses"].tensor_names) == 1

    assert TORNASOLE_DEFAULT_COLLECTIONS_FILE_NAME in files
    cm = CollectionManager.load(
        join(get_path_to_collections(trial_dir), TORNASOLE_DEFAULT_COLLECTIONS_FILE_NAME)
    )

    assert len(cm.collections) == len(coll), (coll, cm.collections)
    assert len(cm.collections["weights"].tensor_names) == 1
    assert len(cm.collections["losses"].tensor_names) == 1
    assert len(cm.collections["gradients"].tensor_names) == 1
    # as we hadn't asked to be saved
    assert len(cm.collections["optimizer_variables"].tensor_names) == 0
    assert len(cm.collections["all"].tensor_names) == 106
    num_tensors_loaded_collection = len(cm.collections["weights"].tensor_names) + len(
        cm.collections["gradients"].tensor_names
    )
    num_tensors_collection = len(coll["weights"].tensor_names) + len(coll["gradients"].tensor_names)
    assert num_tensors_collection == num_tensors_loaded_collection
    assert len(dirs) == 5
    for step in dirs:
        i = 0
        size = 0
        fs = glob.glob(join(trial_dir, "events", step, "**", "*.tfevents"), recursive=True)
        for f in fs:
            fr = FileReader(f)
            for x in fr.read_tensors():
                tensor_name, step, tensor_data, mode, mode_step = x
                i += 1
                print(tensor_name)
                size += tensor_data.nbytes
        assert i == 84
        assert size == 1462
    if hook_created:
        shutil.rmtree(trial_dir)


def test_hook_config_json():
    out_dir = "newlogsRunTest1/test_hook_from_json_config"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.environ[
        TORNASOLE_CONFIG_FILE_PATH_ENV_STR
    ] = "tests/tensorflow/hooks/test_json_configs/test_hook_from_json_config.json"
    reset_collections()
    hook = TornasoleHook.hook_from_config()
    test_save_all_full(hook, out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
