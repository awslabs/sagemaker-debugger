# Standard Library
import os
from abc import ABCMeta
from typing import List, Set

# Third Party
from tensorflow.python.distribute.distribute_lib import _DefaultDistributionStrategy

# First Party
from smdebug.core.config_constants import CONFIG_DEFAULT_WORKER_NAME
from smdebug.core.hook import BaseHook
from smdebug.core.modes import ModeKeys
from smdebug.core.reductions import get_numpy_reduction, get_reduction_tensor_name
from smdebug.core.tfevent.util import make_numpy_array
from smdebug.core.utils import serialize_tf_device
from smdebug.core.writer import FileWriter

# Local
from .collection import CollectionKeys, CollectionManager
from .singleton_utils import set_hook
from .utils import (
    TFDistributionStrategy,
    get_num_workers_from_tf_config,
    get_worker_id_from_tf_config,
    is_mirrored_strategy,
    is_parameter_server_strategy,
)

try:
    # as most of the v1 API is deprecated from the main tf namespace from 1.14
    import tensorflow.compat.v1 as tf
except ImportError:
    # For TF 1.13
    import tensorflow as tf

try:
    pass
except ImportError:
    pass


DEFAULT_INCLUDE_COLLECTIONS = [
    CollectionKeys.METRICS,
    CollectionKeys.LOSSES,
    CollectionKeys.SM_METRICS,
]


class TensorflowBaseHook(BaseHook):
    __metaclass__ = ABCMeta

    def __init__(
        self,
        out_dir,
        export_tensorboard=False,
        tensorboard_dir=None,
        init_step=0,
        dry_run=False,
        reduction_config=None,
        save_config=None,
        include_regex=None,
        include_collections=None,
        save_all=False,
        include_workers="one",
    ):
        collection_manager = CollectionManager()
        super().__init__(
            collection_manager=collection_manager,
            default_include_collections=DEFAULT_INCLUDE_COLLECTIONS,
            init_step=init_step,
            out_dir=out_dir,
            export_tensorboard=export_tensorboard,
            tensorboard_dir=tensorboard_dir,
            dry_run=dry_run,
            reduction_config=reduction_config,
            save_config=save_config,
            include_regex=include_regex,
            include_collections=include_collections,
            save_all=save_all,
            include_workers=include_workers,
        )
        self.optimizer = None
        self._gradients_set = False
        """self.device_map is a mapping between a tf device string to a serialized (filename-friendly) device string
                Example -> /job:worker/replica:0/task:1/device:GPU:0 : _job-worker_replica-0_task-1_device-GPU-0"""
        self.device_map = {}
        self.writer_map = {}
        self.distribution_strategy = None
        self.tf_config = os.getenv(
            "TF_CONFIG"
        )  # caches the TF_CONFIG for the parameter server strategy
        self._hook_supported = None
        set_hook(self)

    def _get_distribution_strategy(self) -> TFDistributionStrategy:
        try:
            import horovod.tensorflow as hvd

            if hvd.size():
                return TFDistributionStrategy.HOROVOD
        except (ModuleNotFoundError, ValueError, ImportError):
            pass

        if self.tf_config and is_parameter_server_strategy(self.tf_config):
            return TFDistributionStrategy.PARAMETER_SERVER_STRATEGY

        strat = tf.distribute.get_strategy()
        if is_mirrored_strategy(strat):
            return TFDistributionStrategy.MIRRORED_STRATEGY

        if isinstance(strat, _DefaultDistributionStrategy):
            # single device
            return TFDistributionStrategy.NONE

        return TFDistributionStrategy.UNSUPPORTED

    def _get_worker_name(self) -> str:
        """
        This function returns the name of the worker based on
        the distribution strategy.

        We do not use this function for MirroredStrategy.
        Device names are used as worker names for this MirroredStrategy.
        The names of the workers are managed by device_map in the case of this strategy.

        It is safe to return the CONFIG_DEFAULT_WORKER_NAME in this case.
        :return: str
        """
        try:
            import horovod.tensorflow as hvd

            if hvd.size():
                return f"worker_{hvd.rank()}"
        except (ModuleNotFoundError, ValueError, ImportError):
            pass

        tf_config = os.getenv("TF_CONFIG")
        if tf_config and is_parameter_server_strategy(tf_config):
            return get_worker_id_from_tf_config(tf_config)
        return CONFIG_DEFAULT_WORKER_NAME

    def export_collections(self):
        num_workers = self._get_num_workers()
        if self.save_all_workers is False:
            num_workers = 1
            if (
                self.distribution_strategy == TFDistributionStrategy.PARAMETER_SERVER_STRATEGY
                or self.distribution_strategy == TFDistributionStrategy.HOROVOD
            ):
                if self.worker != self.chief_worker:
                    return
        self.collection_manager.set_num_workers(num_workers)

        if len(self.device_map):

            for device, serialized_device in self.device_map.items():
                if self.save_all_workers is False and device != self.chief_worker:
                    continue
                collection_file_name = f"{serialized_device}_collections.json"
                self.collection_manager.export(self.out_dir, collection_file_name)
        else:
            collection_file_name = f"{self.worker}_collections.json"
            self.collection_manager.export(self.out_dir, collection_file_name)

    def _get_num_workers(self):
        try:
            import horovod.tensorflow as hvd

            if hvd.size():
                return hvd.size()
        except (ModuleNotFoundError, ValueError, ImportError):
            pass
        tf_config = os.getenv("TF_CONFIG")
        if tf_config and is_parameter_server_strategy(tf_config):
            return get_num_workers_from_tf_config(tf_config)
        strategy = tf.distribute.get_strategy()
        return strategy.num_replicas_in_sync

    def _export_model(self):
        tb_writer = self._maybe_get_tb_writer()
        if tb_writer:
            self.logger.info("Writing graph")
            tb_writer.write_graph(self.graph.as_graph_def(add_shapes=True))
        # don't close writer as it might be needed in the step that follows
        # else we will have to open the file again

    def _add_to_device_map(self, tensor):
        if tensor.device and "CPU" not in tensor.device and tensor.device not in self.device_map:
            self.device_map[tensor.device] = serialize_tf_device(tensor.device)

    def _get_writers(self, tensor_name, tensor_ref) -> List[FileWriter]:
        """
        For tensors generated during distributed tf jobs, we map the tensor to a writer
        with its device attribute.
        If the device attribute is CPU, we map it to all the writers.
        For all other frameworks and single worker jobs we return a list with a single worker.

        If include workers is False, we return a writer only if the
        chief device is attempting to write.
        :param tensor_name:
        :return: List[FileWriter]
        """
        if (
            len(self.device_map)
            and self.distribution_strategy != TFDistributionStrategy.PARAMETER_SERVER_STRATEGY
        ):
            if tensor_ref.tf_obj is not None:
                worker = tensor_ref.tf_obj.device
            else:
                # metrics in Keras
                worker = "CPU"

            if not bool(worker) or "CPU" in worker:
                return list(self.writer_map.values())
            if self.save_all_workers is False:
                if worker == self.chief_worker:
                    worker = self.device_map[worker]
                else:
                    return []
            else:
                worker = self.device_map[worker]
            return [self.writer_map[worker]]
        else:
            return [self.writer] if self.writer else []

    def _initialize_writers(self, only_initialize_if_missing=False) -> None:
        # In keras, sometimes we are not sure if writer is initialized
        # (such as metrics at end of epoch), that's why it passes the flag only_init_if_missing

        if self.dry_run:
            return

        if (
            self.save_all_workers is False
            and self.distribution_strategy != TFDistributionStrategy.MIRRORED_STRATEGY
        ):
            """
            If include_workers is False, we assign we check if the hook has been created by
            the chief worker. If not we do not initialize a writer.
            """
            if self.chief_worker != self.worker:
                return

        if (
            len(self.device_map)
            and self.distribution_strategy != TFDistributionStrategy.PARAMETER_SERVER_STRATEGY
        ):
            """
                Initialize one writer per device string
                If save_all_workers is False, we only initialize a writer
                for the chief worker
            """
            for device, device_string in self.device_map.items():
                if device_string in self.writer_map and only_initialize_if_missing is True:
                    continue
                if self.save_all_workers is True:
                    self.writer_map[device_string] = FileWriter(
                        trial_dir=self.out_dir, step=self.step, worker=device_string
                    )
                elif self.save_all_workers is False and device == self.chief_worker:
                    self.writer_map[device_string] = FileWriter(
                        trial_dir=self.out_dir, step=self.step, worker=device_string
                    )
        else:
            if self.writer is None or only_initialize_if_missing is False:
                self.writer = FileWriter(trial_dir=self.out_dir, step=self.step, worker=self.worker)

    def _close_writers(self) -> None:
        if self.dry_run:
            return

        # flush out sm_metric scalars to metrics file
        if self.metrics_writer is not None:
            self._write_scalars()

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

        # Delete all the dist training writers
        to_delete_writers = []
        for device, writer in self.writer_map.items():
            writer.flush()
            writer.close()
            to_delete_writers.append(device)

        for device in to_delete_writers:
            del self.writer_map[device]

    def _log_unsupported_optimizer(self, optimizer):
        self.logger.warning(
            f"Unsupported optimizer {optimizer} {optimizer.__class__}. "
            "Tornasole can not automatically find the gradients. "
            "Please specify the gradient tensors and optimizer variables "
            "using the methods hook.set_gradients and hook.set_optimizer_variables"
        )

    def _get_collections_with_tensor(self, tf_tensor_name) -> Set["Collection"]:
        self._assert_prep()
        return self.tensor_to_collections[tf_tensor_name]

    def _get_reduction_tensor_name(self, tensor_name, reduction_name, abs):
        return get_reduction_tensor_name(tensor_name, reduction_name, abs, remove_colon_index=False)

    def _write_for_tensor(self, tensor_name, tensor_value, save_collections, tensor_ref=None):
        # this tensor_name is tf tensor name, need to convert to export_name
        tensor_ref = self._get_tensor_ref(tensor_name, save_collections=save_collections)
        if tensor_ref:
            name = tensor_ref.export_name
            super()._write_for_tensor(
                name, tensor_value, save_collections=save_collections, tensor_ref=tensor_ref
            )

    def _get_tensor_ref(self, tf_tensor_name, save_collections=None):
        if save_collections is None:
            save_collections = self._get_collections_with_tensor(tf_tensor_name)
        if save_collections:
            return next(iter(save_collections)).get_tensor(tf_tensor_name)
        else:
            self.logger.warning(
                f"Hook attempted to save unknown tensor {tf_tensor_name}."
                f"This does not belong to any collection"
            )

    def _wrap_apply_gradients(self, optimizer):
        original_apply_gradients = optimizer.__class__.apply_gradients

        def new_apply_gradients(opt, grads_and_vars, global_step=None, name=None):
            # keras models can use tf optimizer through the wrapper
            # keras/optimizers/TFOptimizer
            self.set_gradients(gradients_and_variables=grads_and_vars)
            self.set_optimizer_variables(opt.variables())
            return original_apply_gradients(opt, grads_and_vars, global_step, name)

        optimizer.__class__.apply_gradients = new_apply_gradients
        return optimizer

    def set_gradients(self, gradients=None, gradients_and_variables=None):
        """
        This method allows Tornasole to find the gradient tensors.
        When this method is used for tf.train.Optimizer, gradients_and_variables is passed.
        When this method is used for tf.keras.Optimizer, gradients is passed.

        :param gradients: list of tf.Variables/tf.Tensors/tf.MirroredVariables
            the gradients wrt variables
        :param gradients_and_variables: list of tuples [(tf.Tensor/tf.Variable, tf.Tensor/tf.Variable)...]
            list of tuples representing gradients and weights
        """
        if self._gradients_set is False:
            if gradients is not None:
                self.collection_manager.get(CollectionKeys.GRADIENTS).add_for_mode(
                    gradients, ModeKeys.TRAIN
                )
            elif gradients_and_variables is not None:
                self.collection_manager.get(CollectionKeys.GRADIENTS).add_for_mode(
                    [g for g, v in gradients_and_variables], ModeKeys.TRAIN
                )
            self._gradients_set = True

    def set_optimizer_variables(self, optimizer_variables):
        """
        This method allows Tornasole to find the optimizer variables (such as momentum)
        :param optimizer_variables: list of tf.Variables/tf.Tensors/tf.MirroredVariables
        """
        # since this is done for each variable at a time for keras, not checking if set already
        self.collection_manager.get(CollectionKeys.OPTIMIZER_VARIABLES).add_for_mode(
            optimizer_variables, ModeKeys.TRAIN
        )

    def save_scalar(self, name, value, sm_metric=False):
        """
        save_scalar() not supported on Tensorflow
        """
        self.logger.warning(
            "save_scalar not supported on Tensorflow. "
            "Add the scalar to scalars or sm_metrics collection instead. "
        )
        return

    @staticmethod
    def _make_numpy_array(tensor_value):
        """
        Convert the tensor value into a numpy array.
        Here it's already numpy array
        """
        return make_numpy_array(tensor_value)

    @staticmethod
    def _get_reduction_of_data(reduction_name, tensor_value, tensor_name, abs):
        return get_numpy_reduction(reduction_name, tensor_value, abs)
