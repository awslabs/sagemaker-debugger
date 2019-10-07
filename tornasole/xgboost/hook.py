import atexit
import os
from typing import Optional, List, Union, Tuple, Dict
import numpy as np
from xgboost import DMatrix
from xgboost.core import CallbackEnv
from tornasole.core.collection import Collection, CollectionKeys
from tornasole.core.save_config import SaveConfig
from tornasole.core.hook import CallbackHook
from tornasole.core.access_layer.utils import training_has_ended
from tornasole.core.json_config import create_hook_from_json_config

from .collection import get_collection_manager
from .utils import validate_data_file_path, get_content_type, get_dmatrix


DEFAULT_INCLUDE_COLLECTIONS = [
    CollectionKeys.METRIC,
    CollectionKeys.PREDICTIONS,
    CollectionKeys.LABELS,
    CollectionKeys.FEATURE_IMPORTANCE,
    CollectionKeys.AVERAGE_SHAP
]


class TornasoleHook(CallbackHook):
    """Tornasole hook that represents a callback function in XGBoost."""

    def __init__(
            self,
            out_dir: Optional[str] = None,
            dry_run: bool = False,
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
        super().__init__(collection_manager=get_collection_manager(),
                         default_include_collections=DEFAULT_INCLUDE_COLLECTIONS,
                         data_type_name=None,
                         out_dir=out_dir,
                         dry_run=dry_run,
                         reduction_config=None,
                         save_config=save_config,
                         include_regex=include_regex,
                         include_collections=include_collections,
                         save_all=save_all)
        if reduction_config is not None:
            msg = "'reduction_config' is not supported and will be ignored."
            self.logger.warning(msg)
        self.train_data = self._validate_data(train_data)
        self.validation_data = self._validate_data(validation_data)
        # as we do cleanup ourselves at end of job
        atexit.unregister(self._cleanup)

    def __call__(self, env: CallbackEnv) -> None:
        self._callback(env)

    def get_num_workers(self):
        # TODO :
        return 1

    def get_worker_name(self):
        # TODO :
        pass

    @classmethod
    def hook_from_config(cls):
        return create_hook_from_json_config(cls, get_collection_manager())

    def _cleanup(self):
        # todo: this second export should go
        self.export_collections()
        training_has_ended(self.out_dir)

    def _is_last_step(self, env: CallbackEnv) -> bool:
        return env.iteration + 1 == env.end_iteration

    def _is_collection_being_saved_for_step(self, name):
        return self.collection_manager.get(name) in self.collections_in_this_step

    def _callback(self, env: CallbackEnv) -> None:
        # env.rank: rabit rank of the node/process. master node has rank 0.
        # env.iteration: current boosting round.
        # env.begin_iteration: round # when training started. this is always 0.
        # env.end_iteration: round # when training will end. this is always num_round + 1.  # noqa: E501
        # env.model: model object.
        if not self.prepared_collections:
            # at this point we need all collections to be ready
            # this may not be the case at creation of hook
            # as user's code after hook might add collections
            self._prepare_collections()
            self.prepared_collections = True

        if not self.exported_collections:
            self.export_collections()
            self.exported_collections = True

        self.step = self.mode_steps[self.mode] = env.iteration

        if not self._is_last_step(env) and not self._process_step():
            self.logger.debug("Skipping iteration {}".format(self.step))
            return

        if env.rank > 0:
            self.worker = "worker_{}".format(env.rank)

        self._initialize_writer()

        if self._is_collection_being_saved_for_step(CollectionKeys.METRIC):
            self.write_metrics(env)

        if self._is_collection_being_saved_for_step(CollectionKeys.PREDICTIONS):
            self.write_predictions(env)

        if self._is_collection_being_saved_for_step(CollectionKeys.LABELS):
            self.write_labels(env)

        if self._is_collection_being_saved_for_step(
                CollectionKeys.FEATURE_IMPORTANCE):
            self.write_feature_importances(env)

        if self._is_collection_being_saved_for_step(
                CollectionKeys.AVERAGE_SHAP):
            self.write_average_shap(env)

        if not self._is_last_step(env):
            self._flush_and_close_writer()

        if self._is_last_step(env):
            self._cleanup()

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

    def _write_reductions(self, tensor_name, tensor_value, reduction_config):
        # not writing reductions for xgboost
        return

    @staticmethod
    def _get_reduction_of_data(reduction_name, tensor_value, tensor_name, abs):
        raise NotImplementedError('Reductions are not support by XGBoost hook')

    @staticmethod
    def _make_numpy_array(tensor_value):
        return tensor_value

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
