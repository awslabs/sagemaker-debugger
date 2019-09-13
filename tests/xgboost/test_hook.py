import os
import json
import uuid
import numpy as np
import xgboost

from .run_xgboost_model import run_xgboost_model
from .json_config import get_json_config, get_json_config_full

from tornasole.xgboost import TornasoleHook, get_collection
from tornasole import SaveConfig
from tornasole.core.access_layer.utils import has_training_ended
from tornasole.core.json_config import (
    TORNASOLE_CONFIG_FILE_PATH_ENV_STR, DEFAULT_SAGEMAKER_TORNASOLE_PATH)
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
    monkeypatch.setenv(TORNASOLE_CONFIG_FILE_PATH_ENV_STR, str(config_file))
    hook = TornasoleHook.hook_from_config()
    assert has_training_ended(out_dir) is False
    run_xgboost_model(hook=hook)


def test_hook_from_json_config_full(tmpdir, monkeypatch):
    reset_collections()
    out_dir = tmpdir.join("test_hook_from_json_config_full")
    config_file = tmpdir.join("config.json")
    config_file.write(get_json_config_full(str(out_dir)))
    monkeypatch.setenv(TORNASOLE_CONFIG_FILE_PATH_ENV_STR, str(config_file))
    hook = TornasoleHook.hook_from_config()
    assert has_training_ended(out_dir) is False
    run_xgboost_model(hook=hook)


def test_default_hook(monkeypatch):
    reset_collections()
    monkeypatch.delenv(TORNASOLE_CONFIG_FILE_PATH_ENV_STR, raising=False)
    hook = TornasoleHook.hook_from_config()
    assert hook.out_dir == DEFAULT_SAGEMAKER_TORNASOLE_PATH


def test_hook_save_all(tmpdir):
    reset_collections()
    save_config = SaveConfig(save_steps=[0, 1, 2, 3])
    out_dir = os.path.join(tmpdir, str(uuid.uuid4()))

    hook = TornasoleHook(
        out_dir=out_dir,
        save_config=save_config,
        save_all=True)
    run_xgboost_model(hook=hook)

    trial = create_trial(out_dir)
    tensors = trial.tensors()
    assert len(tensors) > 0
    assert len(trial.available_steps()) == 4
    assert "metric" in trial.collections()
    assert "feature_importance" in trial.collections()
    assert "train-rmse" in tensors
    assert any(t.endswith("/feature_importance") for t in tensors)


def test_hook_save_config_collections(tmpdir):
    reset_collections()
    out_dir = os.path.join(tmpdir, str(uuid.uuid4()))
    hook = TornasoleHook(out_dir=out_dir)

    get_collection("metric").set_save_config(
        SaveConfig(save_interval=2))
    get_collection("feature_importance").set_save_config(
        SaveConfig(save_interval=3))

    run_xgboost_model(hook=hook)

    trial = create_trial(out_dir)
    metric_steps = trial.tensor("train-rmse").steps()
    assert all(step % 2 == 0 for step in metric_steps[:-1])
    fimps = [t for t in trial.tensors() if t.endswith("/feature_importance")]
    fimp_steps = trial.tensor(fimps[0]).steps()
    assert all(step % 3 == 0 for step in fimp_steps[:-1])


def test_hook_shap(tmpdir):
    np.random.seed(42)
    train_data = np.random.rand(5, 10)
    train_label = np.random.randint(2, size=5)
    dtrain = xgboost.DMatrix(train_data, label=train_label)

    reset_collections()
    out_dir = os.path.join(tmpdir, str(uuid.uuid4()))
    hook = TornasoleHook(out_dir=out_dir, train_data=dtrain)
    run_xgboost_model(hook=hook)

    trial = create_trial(out_dir)
    tensors = trial.tensors()
    assert len(tensors) > 0
    assert "average_shap" in trial.collections()


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
    hook = TornasoleHook(out_dir=out_dir, train_data=dtrain, validation_data=dvalid)
    run_xgboost_model(hook=hook)

    trial = create_trial(out_dir)
    tensors = trial.tensors()
    assert len(tensors) > 0
    assert "validation" in trial.collections()
    assert "y/validation" in tensors
    assert "y_hat/validation" in tensors
    assert any(t.endswith("/validation") for t in tensors)
