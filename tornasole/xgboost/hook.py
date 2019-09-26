import os
from typing import Optional, List, Union, Tuple, Dict
import numpy as np
from xgboost import DMatrix
from xgboost.core import CallbackEnv
from tornasole.core.collection import Collection
from tornasole.core.save_config import SaveConfig
from tornasole.core.save_manager import SaveManager
from tornasole.core.modes import ModeKeys, ALLOWED_MODES
from tornasole.core.access_layer.utils import training_has_ended
from tornasole.core.json_config import (
    TORNASOLE_CONFIG_DEFAULT_WORKER_NAME,
    create_hook_from_json_config)
from tornasole.core.writer import FileWriter
from tornasole.core.logger import get_logger
from tornasole.core.collection_manager import COLLECTIONS_FILE_NAME
from tornasole.core.hook_utils import verify_and_get_out_dir
from .collection import get_collection, get_collection_manager
from .utils import validate_data_file_path, get_content_type, get_dmatrix


DEFAULT_INCLUDE_COLLECTIONS = [
    "metric",
    "predictions",
    "labels",
    "feature_importance",
    "average_shap"]


class TornasoleHook:
    """Tornasole hook that represents a callback function in XGBoost."""

    def __init__(
            self,
            out_dir: Optional[str] = None,
            dry_run: bool = False,
            worker: str = TORNASOLE_CONFIG_DEFAULT_WORKER_NAME,
            reduction_config=None,
            save_config: Optional[SaveConfig] = None,
            include_regex: Optional[List[str]] = None,
            include_collections: Optional[List[str]] = None,
            save_all: bool = False,
            train_data: Union[None, Tuple[str, str], DMatrix] = None,
            validation_data: Union[None, Tuple[str, str], DMatrix] = None,
            ) -> None:
        """
        This class represents the hook which is meant to be used a callback
        function in XGBoost.

        Example
        -------
        >>> from tornasole.xgboost import TornasoleHook
        >>> tornasole_hook = TornasoleHook()
        >>> xgboost.train(prams, dtrain, callbacks=[tornasole_hook])

        Parameters
        ----------
        out_dir: A path into which tornasole outputs will be written.
        dry_run: When dry_run is True, behavior is only described in the log
            file, and evaluations are not actually saved.
        worker: name of worker in distributed setting.
        reduction_config: This parameter is not used.
            Placeholder to keep the API consistent with other hooks.
        save_config: A tornasole_core.SaveConfig object.
            See an example at https://github.com/awslabs/tornasole_core/blob/master/tests/test_save_config.py
        include_regex: Tensors matching these regular expressions will be
            available as part of the 'default' collection.
        include_collections: Tensors that should be saved.
            If not given, all known collections will be saved.
        save_all: If true, all evaluations are saved in the collection 'all'.
        train_data: When this parameter is a tuple (file path, content type) or
            an xboost.DMatrix instance, the average feature contributions
            (SHAP values) will be calcaulted against the provided data set.
            content type can be either 'csv' or 'libsvm', e.g.,
            train_data = ('/path/to/train/file', 'csv') or
            train_data = ('/path/to/validation/file', 'libsvm') or
            train_data = xgboost.DMatrix('train.svm.txt')
        validation_data: Same as train_data, but for validation data.
        """  # noqa: E501
        self.out_dir = verify_and_get_out_dir(out_dir)
        self.dry_run = dry_run
        self.worker = worker
        self.train_data = self._validate_data(train_data)
        self.validation_data = self._validate_data(validation_data)

        self.mode = ModeKeys.GLOBAL
        self.mode_steps = {ModeKeys.GLOBAL: -1}
        self.exported_collections = False

        self.logger = get_logger()
        self.logger.info('Saving to {}'.format(self.out_dir))

        if reduction_config is not None:
            msg = "'reduction_config' is not supported and will be ignored."
            self.logger.warning(msg)

        if include_collections is None:
            self.include_collections = DEFAULT_INCLUDE_COLLECTIONS
        else:
            self.include_collections = include_collections

        self._initialize_collectors(include_regex, save_all)

        if save_config is None:
            save_config = SaveConfig()
        if not isinstance(save_config, SaveConfig):
            raise ValueError(f"save_config={save_config} must be type SaveConfig")
        self.save_manager = SaveManager(
            collection_manager=get_collection_manager(),
            include_collections_names=self.include_collections,
            default_reduction_config=None,  # currently not supported in xgb
            default_save_config=save_config)
        self.prepared_save_manager = False

    def __call__(self, env: CallbackEnv) -> None:
        self._callback(env)

    @property
    def collections_path(self) -> str:
        return os.path.join(self.out_dir, COLLECTIONS_FILE_NAME)

    @classmethod
    def hook_from_config(cls):
        return create_hook_from_json_config(
            cls, get_collection_manager(), DEFAULT_INCLUDE_COLLECTIONS)

    def _initialize_collectors(self, include_regex, save_all) -> None:
        if include_regex is not None:
            get_collection("default").include(include_regex)
            if "default" not in self.include_collections:
                self.include_collections.append('default')

        if save_all:
            get_collection("all").include(r".*")
            if "all" not in self.include_collections:
                self.include_collections.append("all")

    def set_mode(self, mode):
        if mode in ALLOWED_MODES:
            self.mode = mode
        else:
            raise ValueError("Invalid mode {}. Valid modes are {}."
                             .format(mode, ','.join(ALLOWED_MODES)))
        if mode not in self.mode_steps:
            self.mode_steps[mode] = -1

    def _is_last_step(self, env: CallbackEnv) -> bool:
        return env.iteration + 1 == env.end_iteration

    def _callback(self, env: CallbackEnv) -> None:
        # env.rank: rabit rank of the node/process. master node has rank 0.
        # env.iteration: current boosting round.
        # env.begin_iteration: round # when training started. this is always 0.
        # env.end_iteration: round # when training will end. this is always num_round + 1.  # noqa: E501
        # env.model: model object.
        if not self.prepared_save_manager:
            # at this point we need all collections to be ready
            # this may not be the case at creation of hook
            # as user's code after hook might add collections
            self.save_manager.prepare()
            self.prepared_save_manager = True

        if not self.exported_collections or self._is_last_step(env):
            get_collection_manager().export(self.collections_path)
            self.exported_collections = True

        self.step = self.mode_steps[self.mode] = env.iteration

        if not self._is_last_step(env) and not self._process_step():
            self.logger.debug("Skipping iteration {}".format(self.step))
            return

        if env.rank > 0:
            self.worker = "worker{}".format(env.rank)

        self._initialize_writer()

        if self.collections_in_this_step.get("metric", False):
            self.write_metrics(env)

        if self.collections_in_this_step.get("predictions", False):
            self.write_predictions(env)

        if self.collections_in_this_step.get("labels", False):
            self.write_labels(env)

        if self.collections_in_this_step.get("feature_importance", False):
            self.write_feature_importances(env)

        if self.collections_in_this_step.get("average_shap", False):
            self.write_average_shap(env)

        if not self._is_last_step(env):
            self._flush_and_close_writer()

        if self._is_last_step(env):
            training_has_ended(self.out_dir)

        self.logger.info("Saved iteration {}.".format(self.step))

    def write_metrics(self, env: CallbackEnv):
        # Get metrics measured at current boosting round
        for metric_name, metric_data in env.evaluation_result_list:
            self._write_tensor(metric_name, metric_data)

    def write_predictions(self, env: CallbackEnv):
        # Write predictions y_hat from validation data
        if not self.validation_data:
            return
        self._write_tensor(
            "predictions",
            env.model.predict(self.validation_data))

    def write_labels(self, env: CallbackEnv):
        # Write labels y from validation data
        if not self.validation_data:
            return
        self._write_tensor("labels", self.validation_data.get_label())

    def write_feature_importances(self, env: CallbackEnv):
        # Get normalized feature importances (fraction of splits made in each
        # feature)
        feature_importances = env.model.get_fscore()
        total = sum(feature_importances.values())

        for feature_name in feature_importances:
            feature_data = feature_importances[feature_name] / total
            self._write_tensor(
                "{}/feature_importance".format(feature_name), feature_data)

    def write_average_shap(self, env: CallbackEnv):
        if not self.train_data:
            return
        # feature names will be in the format, 'f0', 'f1', 'f2', ..., numbered
        # according to the order of features in the data set.
        feature_names = env.model.feature_names + ["bias"]

        shap_values = env.model.predict(self.train_data, pred_contribs=True)
        dim = len(shap_values.shape)
        shap_avg = np.mean(shap_values, axis=tuple(range(dim - 1)))

        for feature_id, feature_name in enumerate(feature_names):
            if shap_avg[feature_id] > 0:
                self._write_tensor(
                    "{}/average_shap".format(feature_name),
                    shap_avg[feature_id])

    def _initialize_writer(self) -> None:
        if self.dry_run:
            return
        self._writer = FileWriter(
            trial_dir=self.out_dir, step=self.step, worker=self.worker)

    def _flush_and_close_writer(self) -> None:
        if self.dry_run:
            return
        self._writer.flush()
        self._writer.close()

    def _write_tensor(self, name, data) -> None:
        if self.dry_run:
            return

        save_collections = self.save_manager.from_collections(name)
        for save_collection in save_collections:
            if save_collection.name in self.collections_in_this_step.keys():
                self._writer.write_tensor(
                    tdata=data, tname=name,
                    mode=self.mode, mode_step=self.mode_steps[self.mode])
                return

    def _process_step(self) -> Dict[str, bool]:
        # returns dictionary of dictionaries: coll_name -> {step: True/False,
        # when_nan: True/False} there will be no entry in dictionary for
        # collections where both step and when_nan are False. This dictionary
        # is stored in self.collections_in_this_step so that we do not need to
        # call this # function in every callback invocation for a given step.
        self.collections_in_this_step = self.save_manager.collections_to_save(
            self.mode, self.mode_steps[self.mode])
        return self.collections_in_this_step

    @staticmethod
    def _validate_data(
            data: Union[None, Tuple[str, str], DMatrix] = None
            ) -> None:
        if data is None or isinstance(data, DMatrix):
            return data
        error_msg = (
            "'data' must be a tuple of strings representing "
            "(file path, content type) or an xgboost.DMatrix instance.")
        is_tuple = isinstance(data, tuple)
        is_first_item_str = isinstance(data[0], str)
        if not (is_tuple and is_first_item_str):
            raise ValueError(error_msg)
        file_path = os.path.expanduser(data[0])
        if not os.path.isfile(file_path):
            raise NotImplementedError(
                "Only local files are currently supported for SHAP.")
        try:
            content_type = get_content_type(data[1])
            validate_data_file_path(file_path, content_type)
            return get_dmatrix(file_path, content_type)
        except Exception:
            raise ValueError(error_msg)
