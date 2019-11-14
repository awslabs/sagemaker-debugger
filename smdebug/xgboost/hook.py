# Standard Library
import os
from typing import Any, Dict, List, Optional, Tuple, Union

# Third Party
import numpy as np
import xgboost as xgb
from xgboost import DMatrix
from xgboost.core import CallbackEnv

# First Party
from smdebug.core.collection import CollectionKeys
from smdebug.core.hook import CallbackHook
from smdebug.core.json_config import create_hook_from_json_config
from smdebug.core.save_config import SaveConfig
from smdebug.core.tfevent.util import make_numpy_array
from smdebug.xgboost.singleton_utils import set_hook

# Local
from .collection import get_collection_manager
from .utils import get_content_type, get_dmatrix, parse_tree_model, validate_data_file_path

DEFAULT_INCLUDE_COLLECTIONS = [CollectionKeys.METRICS]
DEFAULT_SAVE_CONFIG_INTERVAL = 10
DEFAULT_SAVE_CONFIG_START_STEP = 0
DEFAULT_SAVE_CONFIG_END_STEP = None
DEFAULT_SAVE_CONFIG_SAVE_STEPS = []


class Hook(CallbackHook):
    """Tornasole hook that represents a callback function in XGBoost."""

    def __init__(
        self,
        out_dir: Optional[str] = None,
        export_tensorboard: bool = False,
        tensorboard_dir: Optional[str] = None,
        dry_run: bool = False,
        reduction_config=None,
        save_config: Optional[SaveConfig] = None,
        include_regex: Optional[List[str]] = None,
        include_collections: Optional[List[str]] = None,
        save_all: bool = False,
        hyperparameters: Optional[Dict[str, Any]] = None,
        train_data: Union[None, Tuple[str, str], DMatrix] = None,
        validation_data: Union[None, Tuple[str, str], DMatrix] = None,
    ) -> None:
        """
        This class represents the hook which is meant to be used a callback
        function in XGBoost.

        Example
        -------
        >>> from smdebug.xgboost import SessionHook
        >>> hook = SessionHook()
        >>> xgboost.train(prams, dtrain, callbacks=[hook])

        Parameters
        ----------
        out_dir: A path into which tornasole outputs will be written.
        dry_run: When dry_run is True, behavior is only described in the log
            file, and evaluations are not actually saved.
        reduction_config: This parameter is not used.
            Placeholder to keep the API consistent with other hooks.
        save_config: A tornasole_core.SaveConfig object.
            See an example at https://github.com/awslabs/tornasole_core/blob/master/tests/test_save_config.py
        include_regex: Tensors matching these regular expressions will be
            available as part of the 'default' collection.
        include_collections: Tensors that should be saved.
            If not given, all known collections will be saved.
        save_all: If true, all evaluations are saved in the collection 'all'.
        hyperparameters: When this dictionary is given, the key-value pairs
            will be available in the 'hyperparameters' collection.
        train_data: When this parameter is a tuple (file path, content type) or
            an xboost.DMatrix instance, the average feature contributions
            (SHAP values) will be calcaulted against the provided data set.
            content type can be either 'csv' or 'libsvm', e.g.,
            train_data = ('/path/to/train/file', 'csv') or
            train_data = ('/path/to/validation/file', 'libsvm') or
            train_data = xgboost.DMatrix('train.svm.txt')
        validation_data: Same as train_data, but for validation data.
        """  # noqa: E501
        if save_config is None:
            save_config = SaveConfig(save_interval=DEFAULT_SAVE_CONFIG_INTERVAL)
        collection_manager = get_collection_manager()
        super().__init__(
            collection_manager=collection_manager,
            default_include_collections=DEFAULT_INCLUDE_COLLECTIONS,
            data_type_name=None,
            out_dir=out_dir,
            export_tensorboard=export_tensorboard,
            tensorboard_dir=tensorboard_dir,
            dry_run=dry_run,
            reduction_config=None,
            save_config=save_config,
            include_regex=include_regex,
            include_collections=include_collections,
            save_all=save_all,
        )
        if reduction_config is not None:
            msg = "'reduction_config' is not supported and will be ignored."
            self.logger.warning(msg)
        self.hyperparameters = hyperparameters
        self.train_data = self._validate_data(train_data)
        self.validation_data = self._validate_data(validation_data)
        self.worker = self.get_worker_name()
        set_hook(self)

    def __call__(self, env: CallbackEnv) -> None:
        self._callback(env)

    def get_num_workers(self):
        return xgb.rabit.get_world_size()

    def get_worker_name(self):
        return "worker_{}".format(xgb.rabit.get_rank())

    @classmethod
    def hook_from_config(cls, json_config_path=None):
        """Relies on the existence of a JSON file.

        First, check json_config_path. If it's not None,
            If the file exists, use that.
            If the file does not exist, throw an error.
        Otherwise, check the filepath set by a SageMaker environment variable.
            If the file exists, use that.
        Otherwise,
            return None.
        """
        default_values = dict(
            save_interval=DEFAULT_SAVE_CONFIG_INTERVAL,
            start_step=DEFAULT_SAVE_CONFIG_START_STEP,
            end_step=DEFAULT_SAVE_CONFIG_END_STEP,
            save_steps=DEFAULT_SAVE_CONFIG_SAVE_STEPS,
        )
        return create_hook_from_json_config(
            cls,
            get_collection_manager(),
            json_config_path=json_config_path,
            default_values=default_values,
        )

    def _is_last_step(self, env: CallbackEnv) -> bool:
        # env.iteration: current boosting round.
        # env.end_iteration: round # when training will end. this is always num_round + 1.  # noqa: E501
        return env.iteration + 1 == env.end_iteration

    def _increment_step(self, iteration):
        self.step = self.mode_steps[self.mode] = iteration
        self._collections_to_save_for_step = None

    def _callback(self, env: CallbackEnv) -> None:
        if not self.prepared_collections:
            # at this point we need all collections to be ready
            # this may not be the case at creation of hook
            # as user's code after hook might add collections
            self._prepare_collections()
            self.prepared_collections = True

        self._increment_step(env.iteration)

        if self.last_saved_step is not None and not self.exported_collections:
            self.export_collections()
            self.exported_collections = True

        if not self._get_collections_to_save_for_step():
            self.logger.debug("Skipping iteration {}".format(self.step))
            return

        self._initialize_writers()

        if self._is_collection_being_saved_for_step(CollectionKeys.HYPERPARAMETERS):
            self.write_hyperparameters(env)

        if self._is_collection_being_saved_for_step(CollectionKeys.METRICS):
            self.write_metrics(env)

        if self._is_collection_being_saved_for_step(CollectionKeys.PREDICTIONS):
            self.write_predictions(env)

        if self._is_collection_being_saved_for_step(CollectionKeys.LABELS):
            self.write_labels(env)

        if self._is_collection_being_saved_for_step(CollectionKeys.FEATURE_IMPORTANCE):
            self.write_feature_importances(env)

        if self._is_collection_being_saved_for_step(CollectionKeys.AVERAGE_SHAP):
            self.write_average_shap(env)

        if self._is_collection_being_saved_for_step(CollectionKeys.TREES):
            self.write_tree_model(env)

        self.last_saved_step = self.step

        self._close_writers()

    def write_hyperparameters(self, env: CallbackEnv):
        if not self.hyperparameters:
            return
        for param_name, param_value in self.hyperparameters.items():
            self._save_for_tensor("hyperparameters/{}".format(param_name), param_value)

    def write_metrics(self, env: CallbackEnv):
        # Get metrics measured at current boosting round
        for metric_name, metric_data in env.evaluation_result_list:
            self._save_for_tensor(metric_name, metric_data)

    def write_predictions(self, env: CallbackEnv):
        # Write predictions y_hat from validation data
        if not self.validation_data:
            return
        self._save_for_tensor("predictions", env.model.predict(self.validation_data))

    def write_labels(self, env: CallbackEnv):
        # Write labels y from validation data
        if not self.validation_data:
            return
        self._save_for_tensor("labels", self.validation_data.get_label())

    def write_feature_importances(self, env: CallbackEnv):
        # Get normalized feature importances (fraction of splits made in each
        # feature)
        feature_importances = env.model.get_fscore()
        total = sum(feature_importances.values())

        for feature_name in feature_importances:
            feature_data = feature_importances[feature_name] / total
            self._save_for_tensor("{}/feature_importance".format(feature_name), feature_data)

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
                self._save_for_tensor("{}/average_shap".format(feature_name), shap_avg[feature_id])

    def write_tree_model(self, env: CallbackEnv):
        if hasattr(env.model, "booster") and env.model.booster not in {"gbtree", "dart"}:
            self.logger.warning(
                "Tree model dump is not supported for Booster type %s", env.model.booster
            )
            return
        tree = parse_tree_model(env.model, env.iteration)
        for column_name, column_values in tree.items():
            tensor_name = "trees/{}".format(column_name)
            self._save_for_tensor(tensor_name, np.array(column_values))

    def _write_for_tensor(self, tensor_name, tensor_value, save_collections):
        self._write_raw_tensor(tensor_name, tensor_value, save_collections)

    @staticmethod
    def _get_reduction_of_data(reduction_name, tensor_value, tensor_name, abs):
        raise NotImplementedError("Reductions are not supported by XGBoost hook")

    @staticmethod
    def _make_numpy_array(tensor_value):
        return make_numpy_array(tensor_value)

    @staticmethod
    def _validate_data(data: Union[None, Tuple[str, str], DMatrix] = None) -> None:
        if data is None or isinstance(data, DMatrix):
            return data
        error_msg = (
            "'data' must be a tuple of strings representing "
            "(file path, content type) or an xgboost.DMatrix instance."
        )
        is_tuple = isinstance(data, tuple)
        is_first_item_str = isinstance(data[0], str)
        if not (is_tuple and is_first_item_str):
            raise ValueError(error_msg)
        file_path = os.path.expanduser(data[0])
        if not os.path.isfile(file_path):
            raise NotImplementedError("Only local files are currently supported for SHAP.")
        try:
            content_type = get_content_type(data[1])
            validate_data_file_path(file_path, content_type)
            return get_dmatrix(file_path, content_type)
        except Exception:
            raise ValueError(error_msg)
