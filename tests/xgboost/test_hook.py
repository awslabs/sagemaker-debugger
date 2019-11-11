import os
import uuid
import numpy as np
import pytest
import shutil
import xgboost

from .run_xgboost_model import run_xgboost_model
from .json_config import get_json_config, get_json_config_full

from tornasole.xgboost import TornasoleHook, get_collection
from tornasole import SaveConfig
from tornasole.core.access_layer.utils import has_training_ended
from tornasole.core.json_config import CONFIG_FILE_PATH_ENV_STR, DEFAULT_SAGEMAKER_OUTDIR
from tornasole.xgboost import reset_collections
from tornasole.trials import create_trial


def test_hook(tmpdir):
    reset_collections()
    save_config = SaveConfig(save_steps=[0, 1, 2, 3])
    out_dir = os.path.join(tmpdir, str(uuid.uuid4()))
    hook = TornasoleHook(out_dir=out_dir, save_config=save_config)
    assert has_training_ended(out_dir) is False
    run_xgboost_model(hook=hook)


def test_hook_from_json_config(tmpdir, monkeypatch):
    reset_collections()
    out_dir = tmpdir.join("test_hook_from_json_config")
    config_file = tmpdir.join("config.json")
    config_file.write(get_json_config(str(out_dir)))
    monkeypatch.setenv(CONFIG_FILE_PATH_ENV_STR, str(config_file))
    hook = TornasoleHook.hook_from_config()
    assert has_training_ended(out_dir) is False
    run_xgboost_model(hook=hook)


def test_hook_from_json_config_full(tmpdir, monkeypatch):
    reset_collections()
    out_dir = tmpdir.join("test_hook_from_json_config_full")
    config_file = tmpdir.join("config.json")
    config_file.write(get_json_config_full(str(out_dir)))
    monkeypatch.setenv(CONFIG_FILE_PATH_ENV_STR, str(config_file))
    hook = TornasoleHook.hook_from_config()
    assert has_training_ended(out_dir) is False
    run_xgboost_model(hook=hook)


@pytest.mark.skip(reason="If no config file is found, then SM doesn't want a TornasoleHook")
def test_default_hook(monkeypatch):
    reset_collections()
    shutil.rmtree("/opt/ml/output/tensors", ignore_errors=True)
    monkeypatch.delenv(CONFIG_FILE_PATH_ENV_STR, raising=False)
    hook = TornasoleHook.hook_from_config()
    assert hook.out_dir == DEFAULT_SAGEMAKER_OUTDIR


def test_hook_save_all(tmpdir):
    reset_collections()
    save_config = SaveConfig(save_steps=[0, 1, 2, 3])
    out_dir = os.path.join(tmpdir, str(uuid.uuid4()))

    hook = TornasoleHook(out_dir=out_dir, save_config=save_config, save_all=True)
    run_xgboost_model(hook=hook)

    trial = create_trial(out_dir)
    collections = trial.collections()
    tensors = trial.tensors()
    assert len(tensors) > 0
    assert len(trial.steps()) == 4
    assert "all" in collections
    assert "metrics" in collections
    assert "feature_importance" in collections
    assert "train-rmse" in tensors
    assert any(t.endswith("/feature_importance") for t in tensors)
    assert any(t.startswith("trees/") for t in tensors)
    assert len(collections["all"].tensor_names) == len(tensors)


def test_hook_save_config_collections(tmpdir):
    reset_collections()
    out_dir = os.path.join(tmpdir, str(uuid.uuid4()))
    hook = TornasoleHook(out_dir=out_dir, include_collections=["metrics", "feature_importance"])

    get_collection("metrics").save_config = SaveConfig(save_interval=2)
    get_collection("feature_importance").save_config = SaveConfig(save_interval=3)

    run_xgboost_model(hook=hook)

    trial = create_trial(out_dir)
    metric_steps = trial.tensor("train-rmse").steps()
    assert all(step % 2 == 0 for step in metric_steps[:-1])
    fimps = [t for t in trial.tensors() if t.endswith("/feature_importance")]
    fimp_steps = trial.tensor(fimps[0]).steps()
    assert all(step % 3 == 0 for step in fimp_steps[:-1])


def test_hook_shap(tmpdir):
    np.random.seed(42)
    train_data = np.random.rand(10, 10)
    train_label = np.random.randint(2, size=10)
    dtrain = xgboost.DMatrix(train_data, label=train_label)

    reset_collections()
    out_dir = os.path.join(tmpdir, str(uuid.uuid4()))
    hook = TornasoleHook(out_dir=out_dir, include_collections=["average_shap"], train_data=dtrain)
    run_xgboost_model(hook=hook)

    trial = create_trial(out_dir)
    tensors = trial.tensors()
    assert len(tensors) > 0
    assert "average_shap" in trial.collections()
    assert any(t.endswith("/average_shap") for t in tensors)


def test_hook_validation(tmpdir):
    np.random.seed(42)
    train_data = np.random.rand(5, 10)
    train_label = np.random.randint(2, size=5)
    dtrain = xgboost.DMatrix(train_data, label=train_label)
    valid_data = np.random.rand(5, 10)
    valid_label = np.random.randint(2, size=5)
    dvalid = xgboost.DMatrix(valid_data, label=valid_label)

    reset_collections()
    out_dir = os.path.join(tmpdir, str(uuid.uuid4()))
    hook = TornasoleHook(
        out_dir=out_dir,
        include_collections=["labels", "predictions"],
        train_data=dtrain,
        validation_data=dvalid,
    )
    run_xgboost_model(hook=hook)

    trial = create_trial(out_dir)
    tensors = trial.tensors()
    assert len(tensors) > 0
    assert "labels" in trial.collections()
    assert "predictions" in trial.collections()
    assert "labels" in tensors
    assert "predictions" in tensors


def test_hook_tree_model(tmpdir):
    np.random.seed(42)
    train_data = np.random.rand(5, 10)
    train_label = np.random.randint(2, size=5)
    dtrain = xgboost.DMatrix(train_data, label=train_label)
    params = {"objective": "binary:logistic"}
    bst = xgboost.train(params, dtrain, num_boost_round=0)
    df = bst.trees_to_dataframe()

    reset_collections()
    out_dir = os.path.join(tmpdir, str(uuid.uuid4()))
    hook = TornasoleHook(out_dir=out_dir, include_collections=["trees"])
    run_xgboost_model(hook=hook)

    trial = create_trial(out_dir)
    tensors = trial.tensors()
    assert len(tensors) > 0
    assert "trees" in trial.collections()
    for col in df.columns:
        assert "trees/{}".format(col) in tensors


def test_hook_params(tmpdir):
    np.random.seed(42)
    train_data = np.random.rand(5, 10)
    train_label = np.random.randint(2, size=5)
    dtrain = xgboost.DMatrix(train_data, label=train_label)
    valid_data = np.random.rand(5, 10)
    valid_label = np.random.randint(2, size=5)
    dvalid = xgboost.DMatrix(valid_data, label=valid_label)
    params = {"objective": "binary:logistic", "num_round": 20, "eta": 0.1}

    reset_collections()
    out_dir = os.path.join(tmpdir, str(uuid.uuid4()))
    hook = TornasoleHook(
        out_dir=out_dir, include_collections=["hyperparameters"], hyperparameters=params
    )
    run_xgboost_model(hook=hook)

    trial = create_trial(out_dir)
    tensors = trial.tensors()
    assert len(tensors) > 0
    assert "hyperparameters" in trial.collections()
    assert trial.tensor("hyperparameters/objective").value(0) == "binary:logistic"
    assert trial.tensor("hyperparameters/num_round").value(0) == 20
    assert trial.tensor("hyperparameters/eta").value(0) == 0.1
