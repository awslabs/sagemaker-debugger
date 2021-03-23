# Standard Library
import os
import uuid

# Third Party
import numpy as np
import pytest
import xgboost

# First Party
from smdebug import SaveConfig, modes
from smdebug.core.access_layer.utils import has_training_ended
from smdebug.core.collection import CollectionKeys
from smdebug.core.json_config import CONFIG_FILE_PATH_ENV_STR
from smdebug.trials import create_trial
from smdebug.xgboost import Hook

# Local
from .json_config import get_json_config, get_json_config_for_losses, get_json_config_full
from .run_xgboost_model import run_xgboost_model


def test_hook(tmpdir):
    save_config = SaveConfig(save_steps=[0, 1, 2, 3])
    out_dir = os.path.join(tmpdir, str(uuid.uuid4()))
    hook = Hook(out_dir=out_dir, save_config=save_config)
    assert has_training_ended(out_dir) is False
    run_xgboost_model(hook=hook)


def test_hook_from_json_config(tmpdir, monkeypatch):
    out_dir = tmpdir.join("test_hook_from_json_config")
    config_file = tmpdir.join("config.json")
    config_file.write(get_json_config(str(out_dir)))
    monkeypatch.setenv(CONFIG_FILE_PATH_ENV_STR, str(config_file))
    hook = Hook.create_from_json_file()
    assert has_training_ended(out_dir) is False
    run_xgboost_model(hook=hook)


def test_hook_from_json_config_full(tmpdir, monkeypatch):
    out_dir = tmpdir.join("test_hook_from_json_config_full")
    config_file = tmpdir.join("config.json")
    config_file.write(get_json_config_full(str(out_dir)))
    monkeypatch.setenv(CONFIG_FILE_PATH_ENV_STR, str(config_file))
    hook = Hook.create_from_json_file()
    assert has_training_ended(out_dir) is False
    run_xgboost_model(hook=hook)


@pytest.mark.parametrize(
    "params", [{"eval_metric": "rmse"}, {"eval_metric": "auc"}, {"eval_metric": "map"}]
)
def test_hook_from_json_config_for_losses(tmpdir, monkeypatch, params):
    out_dir = tmpdir.join("test_hook_from_json_config_for_losses")
    config_file = tmpdir.join("config.json")
    config_file.write(get_json_config_for_losses(str(out_dir)))
    monkeypatch.setenv(CONFIG_FILE_PATH_ENV_STR, str(config_file))
    hook = Hook.create_from_json_file()
    assert has_training_ended(out_dir) is False
    run_xgboost_model(hook=hook, params=params)
    trial = create_trial(str(out_dir))
    eval_metric = params["eval_metric"]
    test_metric = f"test-{eval_metric}"
    train_metric = f"train-{eval_metric}"
    if eval_metric == "rmse":
        assert train_metric in trial.tensor_names(collection=CollectionKeys.METRICS)
        assert train_metric in trial.tensor_names(collection=CollectionKeys.LOSSES)
        assert test_metric in trial.tensor_names(collection=CollectionKeys.METRICS)
        assert test_metric in trial.tensor_names(collection=CollectionKeys.LOSSES)
    if eval_metric == "auc" or eval_metric == "map":
        assert train_metric in trial.tensor_names(collection=CollectionKeys.METRICS)
        assert train_metric not in trial.tensor_names(collection=CollectionKeys.LOSSES)
        assert test_metric in trial.tensor_names(collection=CollectionKeys.METRICS)
        assert test_metric not in trial.tensor_names(collection=CollectionKeys.LOSSES)


def test_hook_save_every_step(tmpdir):
    save_config = SaveConfig(save_interval=1)
    out_dir = os.path.join(tmpdir, str(uuid.uuid4()))
    hook = Hook(out_dir=out_dir, save_config=save_config)
    run_xgboost_model(hook=hook)
    trial = create_trial(out_dir)
    assert trial.steps() == list(range(10))


def test_hook_save_all(tmpdir):
    save_config = SaveConfig(save_steps=[0, 1, 2, 3])
    out_dir = os.path.join(tmpdir, str(uuid.uuid4()))

    hook = Hook(out_dir=out_dir, save_config=save_config, save_all=True)
    run_xgboost_model(hook=hook)

    trial = create_trial(out_dir)
    collections = trial.collections()
    tensors = trial.tensor_names()
    assert len(tensors) > 0
    assert len(trial.steps()) == 4
    assert "all" in collections
    assert "metrics" in collections
    assert "losses" in collections
    assert "feature_importance" in collections
    assert "train-rmse" in tensors
    assert any(t.startswith("feature_importance/") for t in tensors)
    assert any(t.startswith("trees/") for t in tensors)
    assert len(collections["all"].tensor_names) == len(tensors)


def test_hook_save_config_collections(tmpdir):
    out_dir = os.path.join(tmpdir, str(uuid.uuid4()))
    hook = Hook(out_dir=out_dir, include_collections=["metrics", "feature_importance"])

    hook.get_collection("metrics").save_config = SaveConfig(save_interval=2)
    hook.get_collection("feature_importance").save_config = SaveConfig(save_interval=3)

    run_xgboost_model(hook=hook)

    trial = create_trial(out_dir)
    metric_steps = trial.tensor("train-rmse").steps()
    assert all(step % 2 == 0 for step in metric_steps[:-1])
    fimps = [t for t in trial.tensor_names() if t.startswith("feature_importance/")]
    fimp_steps = trial.tensor(fimps[0]).steps()
    assert all(step % 3 == 0 for step in fimp_steps[:-1])


def test_hook_feature_importance(tmpdir):
    np.random.seed(42)
    train_data = np.random.rand(10, 10)
    train_label = np.random.randint(2, size=10)
    dtrain = xgboost.DMatrix(train_data, label=train_label)

    out_dir = os.path.join(tmpdir, str(uuid.uuid4()))
    hook = Hook(out_dir=out_dir, include_collections=["feature_importance"])
    run_xgboost_model(hook=hook)

    trial = create_trial(out_dir)
    tensors = trial.tensor_names()
    assert len(tensors) > 0
    assert "feature_importance" in trial.collections()
    assert any(t.startswith("feature_importance/") for t in tensors)
    assert any(t.startswith("feature_importance/weight/") for t in tensors)
    assert any(t.startswith("feature_importance/gain/") for t in tensors)
    assert any(t.startswith("feature_importance/cover/") for t in tensors)
    assert any(t.startswith("feature_importance/total_gain/") for t in tensors)
    assert any(t.startswith("feature_importance/total_cover/") for t in tensors)


def test_hook_shap(tmpdir):
    np.random.seed(42)
    train_data = np.random.rand(10, 10)
    train_label = np.random.randint(2, size=10)
    dtrain = xgboost.DMatrix(train_data, label=train_label)

    out_dir = os.path.join(tmpdir, str(uuid.uuid4()))
    hook = Hook(
        out_dir=out_dir, include_collections=["average_shap", "full_shap"], train_data=dtrain
    )
    run_xgboost_model(hook=hook)

    trial = create_trial(out_dir)
    tensors = trial.tensor_names()
    assert len(tensors) > 0
    assert "average_shap" in trial.collections()
    assert "full_shap" in trial.collections()
    assert any(t.startswith("average_shap/") for t in tensors)
    assert any(t.startswith("full_shap/") for t in tensors)
    assert not any(t.endswith("/bias") for t in tensors)
    average_shap_tensors = [t for t in tensors if t.startswith("average_shap/")]
    average_shap_tensor_name = average_shap_tensors.pop()
    assert trial.tensor(average_shap_tensor_name).value(0).shape == (1,)
    full_shap_tensors = [t for t in tensors if t.startswith("full_shap/")]
    full_shap_tensor_name = full_shap_tensors.pop()
    # full shap values should have 10 rows with 10 features + 1 bias
    assert trial.tensor(full_shap_tensor_name).value(0).shape == (10, 11)


def test_hook_validation(tmpdir):
    np.random.seed(42)
    train_data = np.random.rand(5, 10)
    train_label = np.random.randint(2, size=5)
    dtrain = xgboost.DMatrix(train_data, label=train_label)
    valid_data = np.random.rand(5, 10)
    valid_label = np.random.randint(2, size=5)
    dvalid = xgboost.DMatrix(valid_data, label=valid_label)

    out_dir = os.path.join(tmpdir, str(uuid.uuid4()))
    hook = Hook(
        out_dir=out_dir,
        include_collections=["labels", "predictions"],
        train_data=dtrain,
        validation_data=dvalid,
    )
    run_xgboost_model(hook=hook)

    trial = create_trial(out_dir)
    tensors = trial.tensor_names()
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

    out_dir = os.path.join(tmpdir, str(uuid.uuid4()))
    hook = Hook(out_dir=out_dir, include_collections=["trees"])
    run_xgboost_model(hook=hook)

    trial = create_trial(out_dir)
    tensors = trial.tensor_names()
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

    out_dir = os.path.join(tmpdir, str(uuid.uuid4()))
    hook = Hook(out_dir=out_dir, include_collections=["hyperparameters"], hyperparameters=params)
    run_xgboost_model(hook=hook)

    trial = create_trial(out_dir)
    tensors = trial.tensor_names()
    assert len(tensors) > 0
    assert "hyperparameters" in trial.collections()
    assert trial.tensor("hyperparameters/objective").value(0) == "binary:logistic"
    assert trial.tensor("hyperparameters/num_round").value(0) == 20
    assert trial.tensor("hyperparameters/eta").value(0) == 0.1


def test_hook_tensorboard_dir_created(tmpdir):
    out_dir = os.path.join(tmpdir, str(uuid.uuid4()))
    hook = Hook(out_dir=out_dir, export_tensorboard=True)
    run_xgboost_model(hook=hook)
    assert "tensorboard" in os.listdir(out_dir)


def test_setting_mode(tmpdir):
    out_dir = os.path.join(tmpdir, str(uuid.uuid4()))
    hook = Hook(out_dir=out_dir, export_tensorboard=True)
    hook.set_mode(modes.GLOBAL)
    with pytest.raises(ValueError):
        hook.set_mode("a")
