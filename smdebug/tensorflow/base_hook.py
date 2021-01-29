# Standard Library
import os
from abc import ABCMeta
from typing import List, Set, Tuple

# Third Party
import tensorflow.compat.v1 as tf
from tensorflow.python.distribute.distribute_lib import _DefaultDistributionStrategy
from tensorflow.python.framework import ops

# First Party
from smdebug.core.collection import DEFAULT_TF_COLLECTIONS
from smdebug.core.config_constants import DEFAULT_WORKER_NAME
from smdebug.core.hook import BaseHook
from smdebug.core.modes import ModeKeys
from smdebug.core.reductions import get_numpy_reduction, get_reduction_tensor_name
from smdebug.core.utils import check_smdataparallel_env, make_numpy_array, serialize_tf_device
from smdebug.core.writer import FileWriter

# Local
from .collection import CollectionKeys, CollectionManager
from .constants import TF_DEFAULT_SAVED_COLLECTIONS
from .singleton_utils import set_hook
from .utils import (
    TFDistributionStrategy,
    get_chief_worker_from_tf_config,
    get_num_workers_from_tf_config,
    get_worker_id_from_tf_config,
    is_mirrored_strategy,
    is_parameter_server_strategy,
    is_tf_version_2x,
    load_tf_config_json,
)

try:
    import smdistributed.modelparallel.tensorflow as smp  # noqa isort:skip

    _smp_imported = smp
except ImportError:
    _smp_imported = None


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
        profiler_config_parser=None,
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
            profiler_config_parser=profiler_config_parser,
        )
        self.optimizer = None
        self._custom_collections = None
        self._default_collections = None
        self._gradients_set = False
        """self.device_map is a mapping between a tf device string to a serialized (filename-friendly) device string
                Example -> /job:worker/replica:0/task:1/device:GPU:0 : _job-worker_replica-0_task-1_device-GPU-0"""
        self.device_map = {}
        self.writer_map = {}

        # This will be None if the var wasn't set, i.e. not param server
        self.tf_config_json = load_tf_config_json(os.getenv("TF_CONFIG"))
        self._hook_supported = None

        # Identify TF 2.x GradientTape
        self.tape = None
        self._exported_collections = False
        self._distribution_strategy = {
            ModeKeys.TRAIN: None,
            ModeKeys.EVAL: None,
            ModeKeys.PREDICT: None,
            ModeKeys.GLOBAL: None,
        }
        self._prepared_tensors = {
            ModeKeys.TRAIN: False,
            ModeKeys.EVAL: False,
            ModeKeys.PREDICT: False,
            ModeKeys.GLOBAL: False,
        }
        self._exported_model = {
            ModeKeys.TRAIN: False,
            ModeKeys.EVAL: False,
            ModeKeys.PREDICT: False,
            ModeKeys.GLOBAL: False,
        }
        set_hook(self)

    @property
    def distribution_strategy(self):
        return self._distribution_strategy[self.mode]

    @distribution_strategy.setter
    def distribution_strategy(self, distribution_strategy):
        self._distribution_strategy[self.mode] = distribution_strategy

    def _get_distribution_strategy(self) -> TFDistributionStrategy:
        try:
            import horovod.tensorflow as hvd

            if hvd.size():
                return TFDistributionStrategy.HOROVOD
        except (ModuleNotFoundError, ValueError, ImportError):
            pass

        # smdistributed.dataparallel should be invoked via `mpirun`.
        # It supports EC2 machines with 8 GPUs per machine.
        if check_smdataparallel_env():
            try:
                import smdistributed.dataparallel.tensorflow as smdataparallel

                # The total number of GPUs across all the nodes in the cluster
                if smdataparallel.size():
                    return TFDistributionStrategy.SMDATAPARALLEL
            except (ModuleNotFoundError, ValueError, ImportError):
                pass

        strat = tf.distribute.get_strategy()
        if is_mirrored_strategy(strat):
            return TFDistributionStrategy.MIRRORED

        if isinstance(strat, _DefaultDistributionStrategy):
            # single device
            return TFDistributionStrategy.NONE

        # Disable PS till we verify proper support of PS on SM
        # if self.tf_config_json and is_parameter_server_strategy(self.tf_config):
        #     return TFDistributionStrategy.PARAMETER_SERVER

        return TFDistributionStrategy.UNSUPPORTED

    def _assert_distribution_strategy(self):
        """
        The distribution strategy is initialized to None,
        as it's not available during hook construction.
        Later when the graph is ready, that's when correct distribution strategy is returned.
        """
        assert (
            self.distribution_strategy is not None
        ), "_get_distribution_strategy should be called before this method"

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
        self._assert_distribution_strategy()
        if self.distribution_strategy == TFDistributionStrategy.HOROVOD:
            if _smp_imported and _smp_imported.core.initialized:
                # when model parallel is being used, there will be multiple processes
                # with same hvd rank, hence use smp.rank
                return f"worker_{smp.rank()}"

            import horovod.tensorflow as hvd

            return f"worker_{hvd.rank()}"
        elif self.distribution_strategy == TFDistributionStrategy.SMDATAPARALLEL:
            import smdistributed.dataparallel.tensorflow as smdataparallel

            return f"worker_{smdataparallel.rank()}"
        elif self.distribution_strategy == TFDistributionStrategy.MIRRORED:
            # unused for this strategy
            return DEFAULT_WORKER_NAME
        elif self.distribution_strategy == TFDistributionStrategy.PARAMETER_SERVER:
            return get_worker_id_from_tf_config(self.tf_config_json)
        elif self.distribution_strategy == TFDistributionStrategy.NONE:
            return DEFAULT_WORKER_NAME
        elif self.distribution_strategy == TFDistributionStrategy.UNSUPPORTED:
            raise NotImplementedError

    def _get_default_collections(self):
        return DEFAULT_TF_COLLECTIONS

    def export_collections(self):
        # When TF 2.x GradientTape is used, prepare_layers() is not used
        # as the tensors provided by GradientTape are eager tensors and hence,
        # do not require preparing layers
        if not self.tape:
            assert self._prepared_tensors[self.mode]

        if self.save_all_workers is False:
            num_workers = 1
        else:
            num_workers = self._get_num_workers()
        self.collection_manager.set_num_workers(num_workers)

        if self.distribution_strategy in [
            TFDistributionStrategy.PARAMETER_SERVER,
            TFDistributionStrategy.HOROVOD,
            TFDistributionStrategy.SMDATAPARALLEL,
        ]:
            if self.save_all_workers is False and self.worker != self.chief_worker:
                return
        elif self.distribution_strategy == TFDistributionStrategy.MIRRORED:
            if len(self.device_map):
                for device, serialized_device in self.device_map.items():
                    if self.save_all_workers is True or device == self.chief_worker:
                        collection_file_name = f"{serialized_device}_collections.json"
                        self.collection_manager.export(self.out_dir, collection_file_name)
                return

        # below is used in these cases
        # if mirrored and device_map is empty (CPU training)
        # if horovod/param server and worker == chief worker
        collection_file_name = f"{self.worker}_collections.json"
        self.collection_manager.export(self.out_dir, collection_file_name)

    def has_default_hook_configuration(self):
        # Used in AWS TF to determine if the hook
        # is using the default hook configuration
        collections_being_saved = [x.name for x in self._collections_to_save]
        if set(collections_being_saved) == set(TF_DEFAULT_SAVED_COLLECTIONS):
            return True
        return False

    def _get_custom_and_default_collections(self) -> Tuple[Set["Collection"], Set["Collection"]]:
        if self._custom_collections is None:
            self._custom_collections = set()
            self._default_collections = set()
            for coll in self.collection_manager.get_collections().values():
                if coll.name not in DEFAULT_TF_COLLECTIONS:
                    self._custom_collections.add(coll)
                else:
                    self._default_collections.add(coll)

        return self._custom_collections, self._default_collections

    def _get_num_workers(self):
        self._assert_distribution_strategy()
        if self.distribution_strategy == TFDistributionStrategy.HOROVOD:
            if _smp_imported and smp.core.initialized:
                # when model parallel is being used, there will be multiple hvd process groups,
                # hence use smp.size
                return smp.size()

            import horovod.tensorflow as hvd

            return hvd.size()
        elif self.distribution_strategy == TFDistributionStrategy.SMDATAPARALLEL:
            import smdistributed.dataparallel.tensorflow as smdataparallel

            return smdataparallel.size()
        elif self.distribution_strategy == TFDistributionStrategy.MIRRORED:
            strategy = tf.distribute.get_strategy()
            return strategy.num_replicas_in_sync
        elif self.distribution_strategy == TFDistributionStrategy.PARAMETER_SERVER:
            return get_num_workers_from_tf_config(self.tf_config_json)
        elif self.distribution_strategy == TFDistributionStrategy.NONE:
            return 1
        elif self.distribution_strategy == TFDistributionStrategy.UNSUPPORTED:
            return 1

    def _set_chief_worker(self):
        self._assert_distribution_strategy()
        # this won't be used if save_all_workers is True
        if self.distribution_strategy == TFDistributionStrategy.HOROVOD:
            self.chief_worker = DEFAULT_WORKER_NAME
        elif self.distribution_strategy == TFDistributionStrategy.SMDATAPARALLEL:
            self.chief_worker = DEFAULT_WORKER_NAME
        elif self.distribution_strategy == TFDistributionStrategy.MIRRORED:
            assert self._prepared_tensors[self.mode]
            if len(self.device_map):
                self.chief_worker = sorted(self.device_map.keys())[0]
            else:
                self.chief_worker = DEFAULT_WORKER_NAME
        elif self.distribution_strategy == TFDistributionStrategy.PARAMETER_SERVER:
            self.chief_worker = get_chief_worker_from_tf_config(self.tf_config_json)
        elif self.distribution_strategy == TFDistributionStrategy.UNSUPPORTED:
            raise NotImplementedError

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
        if self.distribution_strategy in [
            TFDistributionStrategy.PARAMETER_SERVER,
            TFDistributionStrategy.HOROVOD,
            TFDistributionStrategy.SMDATAPARALLEL,
        ]:
            if self.save_all_workers is True or self.worker == self.chief_worker:
                return self._get_main_writer()
        elif self.distribution_strategy == TFDistributionStrategy.MIRRORED:
            if len(self.device_map):
                # else is for metrics in Keras
                if tensor_ref is not None and tensor_ref.tf_obj is not None:
                    worker = tensor_ref.tf_obj.device
                else:
                    worker = "CPU"
                # if device str is empty or cpu in worker
                if not bool(worker) or "CPU" in worker:
                    if self.save_all_workers:
                        return list(self.writer_map.values())
                    else:
                        return [self.writer_map[self.device_map[self.chief_worker]]]
                elif self.save_all_workers or worker == self.chief_worker:
                    return [self.writer_map[self.device_map[worker]]]
            else:
                # training on CPU when all device strings have cpu
                return self._get_main_writer()
        elif self.distribution_strategy == TFDistributionStrategy.NONE:
            return self._get_main_writer()
        else:
            raise NotImplementedError
        # when self.writer is None, returns empty list
        return []

    def _initialize_writers(self, only_initialize_if_missing=False) -> None:
        # In keras, sometimes we are not sure if writer is initialized
        # (such as metrics at end of epoch), that's why it passes the flag only_init_if_missing
        if self.dry_run:
            return

        if self.distribution_strategy in [
            TFDistributionStrategy.PARAMETER_SERVER,
            TFDistributionStrategy.HOROVOD,
            TFDistributionStrategy.SMDATAPARALLEL,
        ]:
            if self.save_all_workers is True or self.worker == self.chief_worker:
                if self.writer is None or only_initialize_if_missing is False:
                    self.writer = FileWriter(
                        trial_dir=self.out_dir, step=self.step, worker=self.worker
                    )
        elif self.distribution_strategy == TFDistributionStrategy.MIRRORED:
            if len(self.device_map):
                for device, device_string in self.device_map.items():
                    if device_string in self.writer_map and only_initialize_if_missing is True:
                        continue
                    if self.save_all_workers is True or device == self.chief_worker:
                        self.writer_map[device_string] = FileWriter(
                            trial_dir=self.out_dir, step=self.step, worker=device_string
                        )
            else:
                # training on CPU when all device strings have cpu
                if self.writer is None or only_initialize_if_missing is False:
                    self.writer = FileWriter(
                        trial_dir=self.out_dir, step=self.step, worker=self.worker
                    )

        elif self.distribution_strategy == TFDistributionStrategy.NONE:
            if self.writer is None or only_initialize_if_missing is False:
                self.writer = FileWriter(trial_dir=self.out_dir, step=self.step, worker=self.worker)
        else:
            raise NotImplementedError

    def _close_writers(self) -> None:
        if self.dry_run:
            return

        # flush out sm_metric scalars to metrics file
        self._write_scalars()

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

        self._close_given_writer_map(self.writer_map)
        self._close_given_writer_map(self.tb_writers)

    def _export_model(self):
        tb_writer = self._maybe_get_tb_writer()
        if tb_writer:
            tb_writer.write_graph(self.graph.as_graph_def(add_shapes=True))
        # don't close writer as it might be needed in the step that follows
        # else we will have to open the file again

    def _add_to_device_map(self, tensor):
        tensors = []

        # In TF 2.x eager mode, we cannot rely on input tensors to
        # populate this device map as these tensors cannot be saved.
        # Due to this, while executing MirroredStrategy on multiple GPUs,
        # weights and biases in the form of values.MirroredVariable are the
        # first tensors to reach this point. Since MirroredVariable is not
        # processed here, MirroredStrategy distributed training jobs failed
        # on GPU. Adding a check and processing MirroredVariable for TF 2.x
        # eager mode alone.
        if is_tf_version_2x() and tf.executing_eagerly():
            from tensorflow.python.distribute import values

            if isinstance(tensor, values.DistributedValues):
                tensors = [t for t in tensor._values]
        else:
            tensors = [tensor]

        for t in tensors:
            if t.device and "CPU" not in t.device and t.device not in self.device_map:
                self.device_map[t.device] = serialize_tf_device(t.device)

    def _log_unsupported_optimizer(self, optimizer):
        self.logger.warning(
            f"Unsupported optimizer {optimizer} {optimizer.__class__}, cannot automatically find "
            "gradients. Please specify the gradient tensors and optimizer variables "
            "using the methods hook.set_gradients() and hook.set_optimizer_variables()."
        )

    def _get_collections_with_tensor(self, tf_tensor_name) -> Set["Collection"]:
        self._assert_prep()
        # When TF 2.x GradientTape is used, layers are not prepared, hence
        # tensors are not matched with collections at preparation time.
        # Call core/hook.py's _get_collections_with_tensor() where tensors are
        # matched with collections by regex
        if self.tape or (
            tf_tensor_name not in self.tensor_to_collections
            and is_tf_version_2x()
            and tf.executing_eagerly()
        ):
            return super()._get_collections_with_tensor(tf_tensor_name)
        return self.tensor_to_collections[tf_tensor_name]

    def _get_reduction_tensor_name(self, tensor_name, reduction_name, abs):
        return get_reduction_tensor_name(tensor_name, reduction_name, abs, remove_colon_index=False)

    def _write_for_tensor(self, tensor_name, tensor_value, save_collections, tensor_ref=None):
        # When TF 2.x GradientTape is used, the tensors to be saved are of type
        # EagerTensor where tensor values are immediately available.
        # Calling core/hook.py's write_for_tensor directly in this case.
        if self.tape:
            super()._write_for_tensor(tensor_name, tensor_value, save_collections)
            return

        # this tensor_name is tf tensor name, need to convert to export_name
        tensor_ref = self._get_tensor_ref(tensor_name, save_collections=save_collections)
        if tensor_ref is not None:
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
        This method helps find the gradient tensors.
        When this method is used for tf.train.Optimizer, gradients_and_variables is passed.
        When this method is used for tf.keras.Optimizer, gradients is passed.

        :param gradients: list of tf.Variables/tf.Tensors/tf.MirroredVariables
            the gradients wrt variables
        :param gradients_and_variables: list of tuples [(tf.Tensor/tf.Variable, tf.Tensor/tf.Variable)...]
            list of tuples representing gradients and weights
        """
        # TF 2.x provides only symbolic gradient variables that do not provide access to their values.
        # Skipping set_gradients for Tf 2.x until there is
        # support to pass names and values from TF side.

        # From TF 2.2, executing_eagerly_outside_functions() can be used as
        # ops.executing_eagerly_outside_functions() or tf.compat.v1.executing_eagerly_outside_functions().
        # But in TF 2.1, only ops.executing_eagerly_outside_functions() is valid
        if is_tf_version_2x() and ops.executing_eagerly_outside_functions():
            return
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
        This method helps find the optimizer variables (such as momentum)
        :param optimizer_variables: list of tf.Variables/tf.Tensors/tf.MirroredVariables
        """
        # From TF 2.2, executing_eagerly_outside_functions() can be used as
        # ops.executing_eagerly_outside_functions() or tf.compat.v1.executing_eagerly_outside_functions().
        # But in TF 2.1, only ops.executing_eagerly_outside_functions() is valid
        # since this is done for each variable at a time for keras, not checking if set already
        self.collection_manager.get(CollectionKeys.OPTIMIZER_VARIABLES).add_for_mode(
            optimizer_variables, ModeKeys.TRAIN
        )

    @staticmethod
    def _make_numpy_array(tensor_value):
        """
        Convert the tensor value into a numpy array.
        Here it's already numpy array
        """
        if is_tf_version_2x() and tf.executing_eagerly():
            if (
                isinstance(tensor_value, tf.Variable) or isinstance(tensor_value, tf.Tensor)
            ) and hasattr(tensor_value, "numpy"):
                # TF 2.X eager mode
                return tensor_value.numpy()
        return make_numpy_array(tensor_value)

    @staticmethod
    def _get_reduction_of_data(reduction_name, tensor_value, tensor_name, abs):
        if hasattr(tensor_value, "numpy"):
            tensor_value = tensor_value.numpy()
        return get_numpy_reduction(reduction_name, tensor_value, abs)

    def add_to_collection(self, collection_name, variable):
        self.collection_manager.get(collection_name).add(variable)
