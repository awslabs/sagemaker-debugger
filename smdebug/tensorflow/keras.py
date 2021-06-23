# Standard Library
import atexit
import contextlib
import functools
import os
import time

# Third Party
import tensorflow.compat.v1 as tf
from tensorflow.python.distribute import values
from tensorflow.python.framework.indexed_slices import IndexedSlices
from tensorflow.python.util import nest

# First Party
from smdebug.core.locations import TraceFileLocation
from smdebug.core.modes import ModeKeys
from smdebug.core.utils import FRAMEWORK, error_handling_agent, match_inc
from smdebug.profiler.hvd_trace_file_rotation import HvdTraceFileRotation
from smdebug.profiler.profiler_config_parser import MetricsCategory
from smdebug.profiler.profiler_constants import (
    CONVERT_TO_MICROSECS,
    TF_DATALOADER_END_FLAG_FILENAME,
    TF_DATALOADER_START_FLAG_FILENAME,
)
from smdebug.profiler.utils import stop_tf_profiler
from smdebug.tensorflow.callable_cache import CallableCache
from smdebug.tensorflow.utils import InputOutputSaver, get_layer_call_fn

# Local
from .base_hook import TensorflowBaseHook, profiler_config_parser
from .collection import CollectionKeys
from .constants import SMDEBUG_GRADIENTS_KEY, SMDEBUG_LAYER_OUTPUTS_KEY, SMDEBUG_PREFIX
from .tensor_ref import TensorRef, get_tf_names
from .utils import (
    ModelInput,
    ModelInputs,
    ModelOutput,
    TFDistributionStrategy,
    get_export_name_for_keras,
    get_keras_layer_inputs,
    get_keras_layer_outputs,
    get_keras_mode,
    is_keras_optimizer,
    is_profiler_supported_for_tf_version,
    is_tf_version_2_3_x,
    is_tf_version_2x,
    is_tf_version_greater_than_2_4_x,
    supported_tf_variables,
)

# Enable python profiling if profiling is enabled.
profiler_config_parser.start_pre_step_zero_python_profiling()


class KerasHook(TensorflowBaseHook, tf.keras.callbacks.Callback):
    @error_handling_agent.catch_smdebug_errors()
    def __init__(
        self,
        out_dir,
        export_tensorboard=False,
        tensorboard_dir=None,
        dry_run=False,
        reduction_config=None,
        save_config=None,
        include_regex=None,
        include_collections=None,
        save_all=False,
        include_workers="one",
    ):
        TensorflowBaseHook.__init__(
            self,
            out_dir=out_dir,
            export_tensorboard=export_tensorboard,
            tensorboard_dir=tensorboard_dir,
            init_step=-1,
            dry_run=dry_run,
            reduction_config=reduction_config,
            save_config=save_config,
            include_regex=include_regex,
            include_collections=include_collections,
            save_all=save_all,
            include_workers=include_workers,
        )
        tf.keras.callbacks.Callback.__init__(self)
        self.tensor_refs_to_save_this_step = set()
        self._fetches_added = set()
        self.callable_cache = CallableCache()
        self.custom_tensors_to_save = (
            dict()
        )  # stores tensors custom tensors saved by users every step
        self.saved_layers = dict()
        self.has_registered_model = False

        # Profiling vars
        self.tf_profiler = None
        if is_profiler_supported_for_tf_version():
            from tensorflow.python.profiler import profiler_v2 as tf_profiler

            self.tf_profiler = tf_profiler
        self._log_dir = None
        self.is_detailed_profiling = False
        self.is_dataloader_profiling = False
        self.tf_profiler_start_time_in_micros = 0
        self.warm_up_completed = False
        # supports_tf_logs property was introduced in TF 2.3.0
        # it indicates to the framework that the callback is not
        # limited to reading only numpy logs
        self._supports_tf_logs = True
        # TF 2.3.0 has a callback ordering bug
        # this flag indicated to the train_batch_begin callback
        # the the step was already incremented in the on_train_begin callback
        self.step_incremented_in_on_train_begin = False
        self.has_logged_unsupported_tensors_in_non_eager_execution = False
        self.prepared_tf2_collections = False
        self.prepared_gradient_tape_collections = False
        # this flag indicates whether profiling is enabled for native TF2 training
        self.is_profiler_enabled_for_native_training = False
        # this flag indicates whether the hook has been closed or not
        self.is_hook_closed = False

        atexit.register(self.profiling_end)

    def _is_not_supported(self):
        if self.distribution_strategy is None:
            self.distribution_strategy = self._get_distribution_strategy()
        if self._hook_supported is None:
            self._hook_supported = True
            if not is_tf_version_2x() and (
                tf.executing_eagerly()
                or (hasattr(self.model, "run_eagerly") and self.model.run_eagerly)
            ):
                self.logger.info(
                    "Disabling SMDebug as it does not support eager mode" "for TF versions 1.x"
                )
                self._hook_supported = False
            elif self.distribution_strategy == TFDistributionStrategy.MIRRORED:
                try:
                    if is_tf_version_greater_than_2_4_x():
                        # distributed_training_utils.py renamed to distributed_training_utils_v1 in tf 2.4.0
                        from tensorflow.python.keras.distribute.distributed_training_utils_v1 import (
                            get_distributed_model,
                        )
                    else:
                        from tensorflow.python.keras.distribute.distributed_training_utils import (
                            get_distributed_model,
                        )

                except ImportError:
                    # for tf1.13 we can't import this, so we can't support mirrored strategy
                    self.logger.info(
                        "Disabling SMDebug as it does not support mirrored strategy"
                        "with TensorFlow version <1.14"
                    )
                    self._hook_supported = False
            elif self.distribution_strategy == TFDistributionStrategy.UNSUPPORTED:
                self.logger.info(
                    f"Disabling SMDebug as it does not support " f"{tf.distribute.get_strategy()}"
                )
                self._hook_supported = False
        return not self._hook_supported

    @error_handling_agent.catch_smdebug_errors()
    def register_model(self, model):
        # This function is called by the hook in the AWS TF codebase
        # It attaches a hook to every layer of the model to capture
        # layer values
        self.model = model
        self._wrap_model_with_input_output_saver()
        self.has_registered_model = True

    def _get_matching_collections(
        self, mode, tensor, tensor_type, ts_name, is_input_to_model=False, is_output_of_model=False
    ):
        colls_with_tensor = set()
        if tensor_type == "weight":
            if match_inc(
                tensor.name, self.collection_manager.get(CollectionKeys.BIASES).include_regex
            ):
                colls_with_tensor.add(self.collection_manager.get(CollectionKeys.BIASES))
            else:
                colls_with_tensor.add(self.collection_manager.get(CollectionKeys.WEIGHTS))
        elif is_input_to_model:
            colls_with_tensor.add(self.collection_manager.get(CollectionKeys.INPUTS))
        elif is_output_of_model:
            colls_with_tensor.add(self.collection_manager.get(CollectionKeys.OUTPUTS))

        for current_coll in self.collection_manager.get_collections().values():
            if current_coll.name in [
                CollectionKeys.WEIGHTS,
                CollectionKeys.BIASES,
                CollectionKeys.GRADIENTS,
                CollectionKeys.LAYERS,
            ]:
                # don't match regex for these as these are added specially above
                # we also don't want users to make mistakes configuring these collections
                continue

            if match_inc(ts_name, current_coll.include_regex):
                # In TF 2.x eager mode, we can't put tensors in a set/dictionary as tensor.__hash__()
                # is no longer available. tensor.ref() returns a hashable reference
                # object to this Tensor.
                if is_tf_version_2x() and tf.executing_eagerly():
                    if hasattr(tensor, "ref"):
                        # See: https://www.tensorflow.org/api_docs/python/tf/Tensor#ref
                        # experimental_ref is being deprecated for ref
                        tensor = tensor.ref()
                    elif hasattr(tensor, "experimental_ref"):
                        # tensor.experimental_ref is an experimental API
                        # and can be changed or removed.
                        # Ref: https://www.tensorflow.org/api_docs/python/tf/Tensor#experimental_ref
                        tensor = tensor.experimental_ref()
                    else:
                        raise Exception(
                            "Neither ref nor experimental_ref API present. Check TF version"
                        )
                if not current_coll.has_tensor(tensor):
                    # tensor will be added to this coll below
                    colls_with_tensor.add(current_coll)
                # don't recommend adding tensors externally as
                # they will have different internal name
                # but regardless, in such case we only use that tensor name to save data
                # instead of the keras-style-internal-names
        return colls_with_tensor

    def _check_and_add_layer_tensor(
        self, mode, layer, tensor_type, tensor, is_input_to_model=False, is_output_of_model=False
    ):
        if self.distribution_strategy == TFDistributionStrategy.MIRRORED and not tensor.device:
            # these are extra tensors which show up
            # ignoring this still allows us to access all replica's tensors
            # self.logger.debug(f"Skipping {layer} {tensor_type} {tensor}")
            return

        self._add_to_device_map(tensor)

        tf_names = get_tf_names(tensor)
        # multiple tfnames will only be returned for mirrored variable
        export_name = get_export_name_for_keras(layer, tensor_type, tensor)

        # if there are multiple tf_names, it's for mirrored variable.
        # in that case all the tensor ref objects mapping to tf_name in tensor_to_collections
        # have the same export name, although the objects are different
        # as they tf tensor object for different replica
        if tf_names[0] in self.tensor_to_collections:
            export_name = self._get_tensor_ref(tf_names[0]).export_name
            """
            if this tensor has been added already, it already has a export_name
            we need to use that.
            Cases:
            1. layer0_output0 == layer1_input0
            with this first come first ordering, we will hopefully be considering layer0/outputs/tensorname
            this may not work as intended for non sequential models. need to think of that later

            2. tensor added to collection outside of this prepare call, such as gradients
            there we need to use tfname for export_name

            3. same tensor added to collection in previous mode
            again we want to use previous export name.

            In each of these cases we want to set tensor_ref to be the same object as retrieved.
            """

        colls_with_tensor = self._get_matching_collections(
            mode,
            tensor,
            tensor_type,
            export_name,
            is_input_to_model=is_input_to_model,
            is_output_of_model=is_output_of_model,
        )

        self._create_tensors_for_matching_collections(
            mode, tensor, tf_names, export_name, colls_with_tensor
        )

    def _are_tensors_already_added(self, tf_names):
        # multiple tf_names will be here only for mirrored variable
        seen = 0
        for name in tf_names:
            seen += int(name in self.tensor_to_collections)
        if seen > 1:
            assert seen == len(tf_names)
        return seen > 0

    def _create_tensors_for_matching_collections(
        self, mode, tensor, tf_names, export_name, colls_with_tensor
    ):
        # if this tensor was already added to some collection in the previous call
        # do not use it as it is for previous mode
        if colls_with_tensor and not self._are_tensors_already_added(tf_names):
            # need to create new entry in tensor_to_collections dict for the tensor object
            tensor_refs = []
            for coll in colls_with_tensor:
                if not tensor_refs:
                    if isinstance(tensor, supported_tf_variables()):
                        tensor_refs.append(
                            coll.add_variable(tensor, export_name=export_name, mode=mode)
                        )
                    elif isinstance(tensor, tf.Tensor):
                        tensor_refs.append(coll.add_tensor(tensor, name=export_name, mode=mode))
                    elif isinstance(tensor, values.DistributedValues):
                        tensor_refs.extend(
                            coll.add_distributed_variable(
                                tensor, export_name=export_name, mode=mode
                            )
                        )
                    else:
                        raise NotImplementedError
                else:
                    # for second collection onwards
                    for t in tensor_refs:
                        coll.set_tensor_ref(t)
            for t in tensor_refs:
                self.tensor_to_collections[t.name] = colls_with_tensor
        elif colls_with_tensor:
            # we should only readd tensors which were already added if these are variables
            # other tensors are part of a different mode, and will cause a crash if fetched
            # because their input placeholders will not be passed.
            if any(
                [
                    c.name in [CollectionKeys.WEIGHTS, CollectionKeys.BIASES]
                    for c in colls_with_tensor
                ]
            ):
                # set mode of the tensorref object for these tensors
                # these are special because they are tf.Variables which require no input
                # they will be present in all modes
                for tf_name in tf_names:
                    tensor_ref = self._get_tensor_ref(tf_name)
                    tensor_ref.add_mode(mode)
        return

    def _get_distributed_model(self, mode):
        # not available in tf 1.13, code shouldn't reach here for 1.13
        # because of _is_not_supported
        from tensorflow.python.keras.distribute.distributed_training_utils import (
            get_distributed_model,
        )

        return get_distributed_model(self.model, get_keras_mode(mode))

    def _get_model(self, mode):
        if self.distribution_strategy == TFDistributionStrategy.MIRRORED:
            model = self._get_distributed_model(mode)
        else:
            model = self.model
        return model

    def _is_input_layer(self, mode, layer_inputs):
        model_inputs = []
        model = self._get_model(mode)
        # when in mirrored strategy
        if hasattr(model, "values"):
            for per_replica_model in model.values:
                model_inputs.extend(per_replica_model.inputs)
        else:
            model_inputs.extend(model.inputs)
        return any([i in model_inputs for i in layer_inputs])

    def _is_output_layer(self, mode, layer_outputs):
        model_outputs = []
        model = self._get_model(mode)
        # when in mirrored strategy
        if hasattr(model, "values"):
            for per_replica_model in model.values:
                model_outputs.extend(per_replica_model.outputs)
        else:
            model_outputs.extend(model.outputs)
        # In TF 2.X, calling `layer_outputs[0] in model_outputs gives the error:
        # *** tensorflow.python.framework.errors_impl.OperatorNotAllowedInGraphError: using a
        # `tf.Tensor` as a Python `bool` is not allowed in Graph execution. Use Eager execution or
        # decorate this function with @tf.function.
        # Calling `layer_outputs[0] == model_outputs[0]` gives <tf.Tensor 'Equal_1:0'>
        return any([i in model_outputs for i in layer_outputs])

    def _prepare_layers(self, mode):
        # adds any layer tensor (input, output and weight) to appropriate collection
        for layer in self.model.layers:
            # Cannot get input and output tensor values in TF 2.x eager mode.
            # therefore, adding input and output layers only in TF 1.x and
            # TF 2.x non-eager mode.
            if not is_tf_version_2x() or (is_tf_version_2x() and not tf.executing_eagerly()):
                layer_inputs = get_keras_layer_inputs(layer)
                is_input_layer = self._is_input_layer(mode, layer_inputs)
                for inp in layer_inputs:
                    self._check_and_add_layer_tensor(
                        mode, layer, "input", inp, is_input_to_model=is_input_layer
                    )

                layer_outputs = get_keras_layer_outputs(layer)

                is_output_layer = self._is_output_layer(mode, layer_outputs)
                for outp in layer_outputs:
                    self._check_and_add_layer_tensor(
                        mode, layer, "output", outp, is_output_of_model=is_output_layer
                    )

            # Weights can be retrieved in both
            weights = layer.weights

            for w in weights:
                self._check_and_add_layer_tensor(mode, layer, "weight", w)

    def _prepare_non_layer_tensors(self):
        for coll in self.collection_manager.get_collections().values():
            collection_values = coll.get_tensors()
            for tensor_ref in collection_values:
                if tensor_ref.name not in self.tensor_to_collections:
                    self.tensor_to_collections[tensor_ref.name] = {coll}
                elif coll not in self.tensor_to_collections[tensor_ref.name]:
                    self.tensor_to_collections[tensor_ref.name].add(coll)

    def _prepare_tensors_available_post_step(self):
        # for gradients, optimizer_variables
        custom_collections, _ = self._get_custom_and_default_collections()
        for coll in [
            self.get_collection(name=CollectionKeys.OPTIMIZER_VARIABLES),
            self.get_collection(name=CollectionKeys.GRADIENTS),
            self.get_collection(name=CollectionKeys.OUTPUTS),
            self.get_collection(name=CollectionKeys.INPUTS),
        ]:
            collection_values = coll.get_tensors()
            for tensor_ref in collection_values:
                if tensor_ref.name not in self.tensor_to_collections:
                    self.tensor_to_collections[tensor_ref.name] = {coll}
                elif coll not in self.tensor_to_collections[tensor_ref.name]:
                    self.tensor_to_collections[tensor_ref.name].add(coll)

                # Add tensor to custom collections
                for custom_coll in custom_collections:
                    if (
                        match_inc(tensor_ref.name, custom_coll.include_regex)
                        and tensor_ref.tf_obj is not None
                    ):
                        custom_coll.add_for_mode(tensor_ref.tf_obj, self.mode)
                        if custom_coll not in self.tensor_to_collections[tensor_ref.name]:
                            self.tensor_to_collections[tensor_ref.name].add(custom_coll)

    def _prepare_tensors_for_step(self, mode):
        self.tensor_refs_to_save_this_step = set()
        colls_to_save_for_step = self._get_collections_to_save_for_step()
        input_tensors_set = set(
            self.collection_manager.get(CollectionKeys.INPUTS).get_tensors(mode=mode)
        )
        for coll in colls_to_save_for_step:
            if coll.name in [CollectionKeys.METRICS, CollectionKeys.LOSSES, CollectionKeys.INPUTS]:
                # these should not be added to fetches, and can be retrieved after the step ends
                continue
            # below fetches even tensors which users might have added manually through collection API
            non_input_tensors = set(coll.get_tensors(mode=mode)).difference(input_tensors_set)
            self.tensor_refs_to_save_this_step.update(non_input_tensors)

    def _add_metric(self, metric_name, metric_value: tf.Tensor = None):
        if metric_name in self.tensor_to_collections:
            return

        if metric_name in ["loss", "val_loss"]:
            coll_name = CollectionKeys.LOSSES
        else:
            coll_name = CollectionKeys.METRICS
        coll = self.collection_manager.get(coll_name)
        if metric_value:
            coll.set_tensor_ref(metric_value, metric_name)
        else:
            coll.set_tensor_ref(TensorRef.from_non_graph_var(metric_name))
        self.tensor_to_collections[metric_name] = {coll}

    def _save_custom_tensors_post_step(self):
        # This saves all the values of custom tensors
        # that the user has saved with the save_tensor api
        for tensor_name in self.custom_tensors_to_save:
            tensor_value, collection_names = self.custom_tensors_to_save[tensor_name]
            self._save_tensor_to_file(tensor_name, tensor_value, collection_names)
        self.custom_tensors_to_save.clear()

    @error_handling_agent.catch_smdebug_errors()
    def should_save_layer(self, layer_name):
        # Called in AWS TF to determine
        # if a particular layer value
        # should be saved
        return self.should_save_tensor_or_collection(layer_name, CollectionKeys.LAYERS)

    def _save_tensor_to_file(self, tensor_name, tensor_value, collections):
        if isinstance(collections, set) is False:
            collections = {collections}
        # Since this function modifies the set, there is a possibility
        # of bugs if calling functions attempt to re-use the set passed
        # to this function
        collections_to_write = collections.copy()
        collections_to_save = self._get_collections_to_save_for_step()
        for c in collections_to_save:
            if match_inc(tensor_name, c.include_regex):
                collections_to_write.add(c)
        self._initialize_writers(only_initialize_if_missing=True)
        tensor_refs = []
        if isinstance(tensor_value, values.PerReplica):
            for t in tensor_value._values:
                tensor_ref = TensorRef.from_non_graph_var(tensor_name)
                tensor_refs.append((tensor_ref, t))
        else:
            tensor_ref = TensorRef.from_non_graph_var(tensor_name)
            tensor_refs.append((tensor_ref, tensor_value))

        for tensor_ref, t in tensor_refs:
            for collection in collections_to_write:
                if isinstance(collection, str):
                    collection = self.get_collection(collection)
                collection.set_tensor_ref(tensor_ref)
            self._save_for_tensor(tensor_name, t, check_before_write=True)

    def _save_gradients_from_logs(self, gradients):
        if gradients is not None:
            gradient_collection = self.get_collection(CollectionKeys.GRADIENTS)
            step_collections = self._get_collections_to_save_for_step()
            collections_to_write = (
                {gradient_collection} if gradient_collection in step_collections else set()
            )
            if gradients and isinstance(gradients[0], tuple) is False:
                gradients = zip(self.model.trainable_variables, gradients)
            for v, g in gradients:
                if isinstance(v, tf.Tensor):
                    # Tensor.name is meaningless with eager execution
                    layer_name = str(v.numpy(), "utf-8")
                elif isinstance(v, supported_tf_variables()):
                    layer_name = v.name
                elif isinstance(v, bytes):
                    layer_name = str(v, "utf-8")
                else:
                    layer_name = v
                layer_name = layer_name.split(":")[0]
                export_name = "gradients/" + layer_name + "Grad"
                if isinstance(g, IndexedSlices):
                    # This class is a simple wrapper for a pair of Tensor objects
                    # See: https://www.tensorflow.org/api_docs/python/tf/IndexedSlices
                    g = g.values
                self._save_tensor_to_file(export_name, g, collections_to_write)

    def _save_model_inputs_and_outputs_helper(self, collection_key, tensors_to_save, prefix):
        collections_to_write = (
            {self.get_collection(collection_key)}
            if self._is_collection_being_saved_for_step(collection_key)
            else set()
        )
        if isinstance(tensors_to_save, (dict, list)):
            tensors_to_save = nest.flatten(tensors_to_save)
            for idx, t_value in enumerate(tensors_to_save):
                t_name = f"{prefix}_{idx}"
                self._save_tensor_to_file(t_name, t_value, collections_to_write)
        else:
            self._save_tensor_to_file(prefix, tensors_to_save, collections_to_write)

    @error_handling_agent.catch_smdebug_errors()
    def save_smdebug_logs(self, logs):
        if logs is None:
            return

        for key in logs:
            if SMDEBUG_PREFIX in key:
                # Save Model Outputs
                if key == ModelOutput.LABELS:
                    self._save_model_inputs_and_outputs_helper(
                        CollectionKeys.OUTPUTS, logs[key], prefix="labels"
                    )
                elif key == ModelOutput.PREDICTIONS:
                    self._save_model_inputs_and_outputs_helper(
                        CollectionKeys.OUTPUTS, logs[key], prefix="predictions"
                    )
                # Save Gradients
                elif key == SMDEBUG_GRADIENTS_KEY:
                    self._save_gradients_from_logs(logs[key])
                # Save Intermediate Layers
                elif key == SMDEBUG_LAYER_OUTPUTS_KEY:
                    self._save_layer_values(logs[key])
                # Save Model Inputs
                elif key in ModelInputs:
                    self._save_model_inputs_and_outputs_helper(
                        CollectionKeys.INPUTS, logs[key], prefix="inputs"
                    )

    def _save_metrics(self, batch, logs, force_save=False):
        # if force_save is True, doesn't check whether collection needs to be saved for steps
        if logs is None:
            return

        if force_save or self._is_collection_being_saved_for_step(CollectionKeys.METRICS):
            self._initialize_writers(only_initialize_if_missing=True)
            logs["batch"] = batch
            for key in logs:
                if key in {"loss", "val_loss", "outputs"} or "smdebug_" in key:
                    # outputs is saved differently through outputs collection
                    continue
                self._add_metric(metric_name=key)
                self._save_for_tensor(key, logs[key], check_before_write=False)

        if force_save or self._is_collection_being_saved_for_step(CollectionKeys.LOSSES):
            self._initialize_writers(only_initialize_if_missing=True)
            for key in ["loss", "val_loss"]:
                if key in logs:
                    self._add_metric(metric_name=key)
                    self._save_for_tensor(key, logs[key], check_before_write=False)

    def _save_layer_input_and_outputs(self):
        if is_tf_version_2x() is False:
            return

        for layer_name in self.saved_layers:
            # Save Input
            input_collection = (
                {self.get_collection(CollectionKeys.LAYERS)}
                if self._is_collection_being_saved_for_step(CollectionKeys.LAYERS)
                else set()
            )
            if len(input_collection) > 0:
                c = self.get_collection(CollectionKeys.LAYERS)
                if match_inc(layer_name, c.include_regex) or c.include_regex == []:
                    pass
                else:
                    continue

            tensor = self.saved_layers[layer_name].layer_input
            export_name = get_export_name_for_keras(layer_name, tensor_type="input", tensor=tensor)
            t = tensor[0] if isinstance(tensor, list) and len(tensor) else tensor
            if hasattr(t, "numpy") is False:
                continue
            self._save_tensor_to_file(export_name, tensor, input_collection)

            # Save Output
            tensor = self.saved_layers[layer_name].layer_output
            export_name = get_export_name_for_keras(layer_name, tensor_type="output", tensor=tensor)
            self._is_collection_being_saved_for_step(CollectionKeys.LAYERS)
            output_collection = (
                {self.get_collection(CollectionKeys.LAYERS)}
                if self._is_collection_being_saved_for_step(CollectionKeys.LAYERS)
                else set()
            )
            t = tensor[0] if isinstance(tensor, list) and len(tensor) else tensor
            if hasattr(t, "numpy") is False:
                continue
            self._save_tensor_to_file(export_name, tensor, output_collection)

    def _save_tensors_post_step(self, batch, logs):
        # some tensors available as value from within hook are saved here
        # weights, metrics
        self._save_metrics(batch, logs)
        self.save_smdebug_logs(logs)
        self._save_custom_tensors_post_step()
        self._save_layer_input_and_outputs()

        if is_tf_version_2x() and tf.executing_eagerly():
            for tensor_ref in self.tensor_refs_to_save_this_step:
                tensor = tensor_ref.tf_obj
                self._save_for_tensor(
                    tensor_name=tensor.name, tensor_value=tensor.value(), check_before_write=False
                )

    def _get_exec_function(self, mode):
        # exec_function is None in 2.X; self.model exists but has no train_function, test_function, etc.
        if self.distribution_strategy in [
            TFDistributionStrategy.NONE,
            TFDistributionStrategy.HOROVOD,
            TFDistributionStrategy.SMDATAPARALLEL,
        ]:
            if mode == ModeKeys.TRAIN:
                x = self.model.train_function
            elif mode == ModeKeys.EVAL:
                x = self.model.test_function
            elif mode == ModeKeys.PREDICT:
                x = self.model.predict_function
            else:
                raise NotImplementedError
        else:
            x = self._get_distributed_model(mode)._distributed_function
        return x

    def _validate_exec_function(self, fn):
        if fn is None:
            self.logger.info(
                f"Could not save tensors for mode {self.mode.name} step {self.mode_steps[self.mode]} "
                f"as execution function has not yet been built."
            )
            return False
        else:
            return True

    def _save_tensor_callback(self, value, name, check):
        # this function changes the order of args so we can create a partial function for callback
        self._save_for_tensor(tensor_name=name, tensor_value=value, check_before_write=check)

    def _add_callbacks(self, mode):
        # safest if hook callback is the last
        # self.original_fetches = self._get_exec_function(mode).fetches.copy()

        x = self._get_exec_function(mode)  # Returns GraphExecutionFunction
        if self._validate_exec_function(x):
            for tensor_ref in self.tensor_refs_to_save_this_step:
                tensor = tensor_ref.tf_obj
                if tensor not in x.fetches and tensor not in x.fetch_callbacks:
                    x.fetches.append(tensor)
                    self._fetches_added.add(tensor)
                    x.fetch_callbacks[tensor] = functools.partial(
                        self._save_tensor_callback, name=tensor_ref.name, check=False
                    )
                else:
                    self.logger.warning(
                        f"Cannot save tensor {tensor.name} as there is already "
                        f"a callback registered for this tensor. "
                        f"Please remove the existing callback to save this tensor."
                    )

            callable_fn = self.callable_cache.get_fn(mode, x.fetches)
            if callable_fn is not None:
                x._fetches = list(x.fetches)
                x._callable_fn = callable_fn

    def _remove_fetches_and_callbacks(self, mode):
        x = self._get_exec_function(mode)

        # cache the callable for given fetches
        self.callable_cache.cache_fn(mode, fetches=x.fetches, callable_fn=x._callable_fn)

        for tf_obj in self._fetches_added:
            x.fetches.remove(tf_obj)
            x.fetch_callbacks.pop(tf_obj)
        self._fetches_added.clear()

    def _prepare_collections_for_tf2(self):
        self._prepare_collections()
        if self.has_default_hook_configuration():
            # wrapping the model is only supported if the hook does not have the default hook configuration
            self._unwrap_model_with_input_output_saver()
        self.prepared_tf2_collections = True

    def _start_detailed_profiling(self, current_step):
        """Start detailed profiling if the TF profiler is supported in the current TF version and detailed profiling
        wasn't already started.
        """
        if not is_profiler_supported_for_tf_version() or self.is_detailed_profiling:
            return

        self._log_dir = TraceFileLocation.get_detailed_profiling_log_dir(
            self.profiler_config_parser.config.local_path,
            FRAMEWORK.TENSORFLOW.value,
            self.mode_steps[ModeKeys.TRAIN],
        )
        self.logger.info(f"Enabling TF profiler on step: = {current_step}")
        if not self.warm_up_completed:
            # warming up profiler before it will be profiling.
            self.tf_profiler.warmup()
            self.warm_up_completed = True
        self.tf_profiler.start(self._log_dir)
        self.tf_profiler_start_time_in_micros = time.time() * CONVERT_TO_MICROSECS
        self.is_detailed_profiling = True

    def _stop_detailed_profiling(self, current_step):
        """Stop detailed profiling if the TF profiler is supported in the current TF version and detailed profiling
        was already started.
        """
        if not is_profiler_supported_for_tf_version() or not self.is_detailed_profiling:
            return

        self.logger.info(f"Disabling TF profiler on step: ={current_step}")
        stop_tf_profiler(
            tf_profiler=self.tf_profiler,
            log_dir=self._log_dir,
            start_time_us=self.tf_profiler_start_time_in_micros,
        )
        self.is_detailed_profiling = False

    def _start_or_stop_detailed_profiling(self, current_step):
        """Handle detailed profiling for the step.

        If detailed profiling should be enabled for the current step, then start detailed profiling. Otherwise, stop
        detailed profiling.
        """
        if self.profiler_config_parser.should_save_metrics(
            MetricsCategory.DETAILED_PROFILING, current_step
        ):
            self._start_detailed_profiling(current_step)
        else:
            self._stop_detailed_profiling(current_step)

    def _start_dataloader_profiling(self):
        """Start dataloader profiling if dataloader profiling wasn't already started. Write the TF dataloader start flag.

        Sets `is_dataloader_profiling` to True if the flag was written successfully.
        """
        if (
            not self.is_dataloader_profiling
            and self.profiler_config_parser.write_tf_dataloader_flag(
                TF_DATALOADER_START_FLAG_FILENAME
            )
        ):
            self.is_dataloader_profiling = True

    def _stop_dataloader_profiling(self):
        """Stop dataloader profiling if dataloader profiling was already started. Write the TF dataloader stop flag.

        Sets `is_dataloader_profiling` to False if the flag was written successfully.
        """
        if self.is_dataloader_profiling and self.profiler_config_parser.write_tf_dataloader_flag(
            TF_DATALOADER_END_FLAG_FILENAME
        ):
            self.is_dataloader_profiling = False

    def _start_or_stop_dataloader_profiling(self, current_step):
        """Handle dataloader profiling for the step.

        If dataloader profiling is enabled for the current step, then start dataloader profiling. Otherwise, stop
        dataloader profiling.
        """
        if self.profiler_config_parser.should_save_metrics(
            MetricsCategory.DATALOADER_PROFILING, current_step
        ):
            self._start_dataloader_profiling()
        else:
            self._stop_dataloader_profiling()

    @error_handling_agent.catch_smdebug_errors()
    def on_epoch_begin(self, batch, logs=None):
        pass

    @error_handling_agent.catch_smdebug_errors()
    def on_epoch_end(self, batch, logs=None):
        if self._is_not_supported():
            return
        self._save_metrics(batch=batch, logs=logs, force_save=True)
        self._close_writers()

    def _on_any_mode_begin(self, mode):
        if self._is_not_supported():
            return
        self.worker = self._get_worker_name()

        # Only the chief worker will read the Horovod timeline file
        # if HOROVOD_TIMELINE is a valid file and SM Profiler is enabled
        if not self.hvd_reader and self.worker == self.chief_worker:
            self.hvd_reader = HvdTraceFileRotation(self.profiler_config_parser)

        self.graph = tf.get_default_graph()
        self.set_mode(mode)

        if self.prepared_tf2_collections is False and is_tf_version_2_3_x():
            # Addresses ordering issues in TF 2.3.0
            # sets prepared_collections to True here
            self._prepare_collections_for_tf2()
            self._increment_step()
            self.step_incremented_in_on_train_begin = True

        # have to clear callable cache if we are not caching per mode
        self.callable_cache.change_mode()

    @error_handling_agent.catch_smdebug_errors()
    def on_train_begin(self, logs=None):
        self._on_any_mode_begin(ModeKeys.TRAIN)

    @error_handling_agent.catch_smdebug_errors()
    def on_test_begin(self, logs=None):
        self._on_any_mode_begin(ModeKeys.EVAL)

    def _on_any_mode_end(self, mode):
        self.profiler_config_parser.handle_step_end_python_profiling(mode, self.mode_steps[mode])
        self._stop_dataloader_profiling()

    @error_handling_agent.catch_smdebug_errors()
    def on_train_end(self, logs=None):
        self._on_any_mode_end(ModeKeys.TRAIN)
        self._stop_detailed_profiling(self.mode_steps[self.mode])

    @error_handling_agent.catch_smdebug_errors()
    # throws error in keras if this fn is absent
    def on_test_end(self, logs=None):
        self._on_any_mode_end(ModeKeys.EVAL)

    @error_handling_agent.catch_smdebug_errors()
    # throws error in keras if this fn is absent
    def on_predict_end(self, logs=None):
        self._on_any_mode_end(ModeKeys.PREDICT)

    @error_handling_agent.catch_smdebug_errors()
    def on_predict_begin(self, logs=None):
        self._on_any_mode_begin(ModeKeys.PREDICT)

    def _wrap_model_with_input_output_saver(self):
        if self.has_registered_model:
            return
        for layer in self.model.layers:
            layer._hooks = []
            layer.call = get_layer_call_fn(layer)
            layer.register_hook = lambda hook: layer._hooks.append(hook)
            saver = InputOutputSaver()
            layer.register_hook(saver)
            self.saved_layers[layer.name] = saver

    def _unwrap_model_with_input_output_saver(self):
        if self.has_registered_model is False:
            return
        for layer in self.model.layers:
            layer.call = layer.old_call

    def _on_any_batch_begin(self, batch, mode, logs=None):
        self.start = time.time()
        if self._is_not_supported():
            return

        # set mode for each batch as when users run model.fit() and pass validation data
        # through the optional argument, then mode_begin is not called for the training steps
        # after first evaluation during training
        self.set_mode(mode)

        # Write the gradients of the past step if the writer is still available.
        if self.writer is not None or len(self.writer_map):
            self._close_writers()

        # Addresses callback ordering bug in TF 2.3.0
        if self.step_incremented_in_on_train_begin is False:
            self._increment_step()
        else:
            self.step_incremented_in_on_train_begin = False

        self.profiler_config_parser.load_config()
        self.profiler_config_parser.handle_step_start_python_profiling(mode, self.mode_steps[mode])
        self._start_or_stop_dataloader_profiling(self.mode_steps[mode])

        if self.prepared_tf2_collections is False:
            # sets prepared_collections to True here
            self._prepare_collections_for_tf2()

        if self._prepared_tensors[mode] is False:
            if (is_tf_version_2x() and tf.executing_eagerly()) or self._validate_exec_function(
                self._get_exec_function(mode)
            ):
                self._prepare_layers(mode)
                self._prepare_non_layer_tensors()
                self._prepare_tensors_available_post_step()
                self._prepared_tensors[mode] = True
                # below should be after tensors are processed,
                # so we know that device map is populated
                self._set_chief_worker()
            # else:
            # this will delay the preparation of tensors as the
            # full graph is not built. Gradients are not available
            # at this stage for example

        if self._prepared_tensors[mode]:
            self._prepare_tensors_for_step(mode)
            if self.tensor_refs_to_save_this_step:
                # if saving metric, writer may not be initialized as a result
                self._initialize_writers()

            if not is_tf_version_2x() or (is_tf_version_2x() and not tf.executing_eagerly()):
                self._add_callbacks(mode)

    @error_handling_agent.catch_smdebug_errors()
    def on_train_batch_begin(self, batch, logs=None):
        self._on_any_batch_begin(batch, ModeKeys.TRAIN, logs=logs)
        self._start_or_stop_detailed_profiling(self.mode_steps[self.mode])

    @error_handling_agent.catch_smdebug_errors()
    def on_test_batch_begin(self, batch, logs=None):
        self._on_any_batch_begin(batch, ModeKeys.EVAL, logs=logs)

    @error_handling_agent.catch_smdebug_errors()
    def on_predict_batch_begin(self, batch, logs=None):
        self._on_any_batch_begin(batch, ModeKeys.PREDICT, logs=logs)

    def _save_layer_values(self, logs):
        if logs is None:
            return
        step_collections = self._get_collections_to_save_for_step()
        layer_collection = self.get_collection(CollectionKeys.LAYERS)
        collections_to_write = {layer_collection} if layer_collection in step_collections else set()
        for layer_name, layer_input, layer_output in logs:
            # Cast layer_name to str since it can also be of type bytes
            # when run with mirrored strategy
            if isinstance(layer_name, tf.Tensor):
                # Tensor.name is meaningless with eager execution
                layer_name = str(layer_name.numpy(), "utf-8")
            elif isinstance(layer_name, supported_tf_variables()):
                layer_name = layer_name.name
            elif isinstance(layer_name, bytes):
                layer_name = str(layer_name, "utf-8")
            layer_input_tensor_name = get_export_name_for_keras(str(layer_name), "input")
            if len(layer_input) == 1:
                # Layer Inputs are flattened and passed as a list into
                # the next layer. Unpacking it speeds up the _make_numpy fn.
                layer_input = layer_input[0]
                self._save_tensor_to_file(
                    layer_input_tensor_name, layer_input, collections_to_write
                )
            else:
                for idx, l_name in enumerate(layer_input):
                    layer_input_tensor_name_with_idx = f"{layer_input_tensor_name}_{idx}"
                    self._save_tensor_to_file(
                        layer_input_tensor_name_with_idx, l_name, collections_to_write
                    )
            layer_output_tensor_name = get_export_name_for_keras(str(layer_name), "output")
            if isinstance(layer_output, list):
                for idx, l_output in enumerate(layer_output):
                    layer_output_tensor_name_with_idx = f"{layer_output_tensor_name}_{idx}"
                    self._save_tensor_to_file(
                        layer_output_tensor_name_with_idx, l_output, collections_to_write
                    )
            else:
                self._save_tensor_to_file(
                    layer_output_tensor_name, layer_output, collections_to_write
                )

    def _write_optimizer_variables(self):
        optimizer_collections = self.collection_manager.get(CollectionKeys.OPTIMIZER_VARIABLES)
        collections_to_save = self._get_collections_to_save_for_step()
        for tensor_ref in optimizer_collections.get_tensors(mode=ModeKeys.TRAIN):
            tensor = tensor_ref.tf_obj
            collections_to_save = self._get_collections_with_tensor(tensor.name).intersection(
                collections_to_save
            )
            if len(collections_to_save):
                self._initialize_writers(only_initialize_if_missing=True)
                tensor = tensor_ref.tf_obj
                self._add_to_device_map(tensor)
                tf_names = get_tf_names(tensor)
                for name in tf_names:
                    self._save_for_tensor(
                        tensor_name=name, tensor_value=tensor.value(), check_before_write=False
                    )

    def _on_any_batch_end(self, batch, mode, logs=None):
        if self._is_not_supported():
            return

        self.record_trace_events(
            training_phase="Step:" + str(mode),
            op_name="Step:" + str(mode),
            phase="X",
            timestamp=self.start,  # this is start time for step
            duration=time.time() - self.start,
            pid=os.getpid(),
            step_num=str(self.mode_steps[mode]),
        )

        if not is_tf_version_2x() or (is_tf_version_2x() and not tf.executing_eagerly()):
            self._remove_fetches_and_callbacks(mode)

        self._save_tensors_post_step(batch, logs)
        if is_tf_version_2x() and tf.executing_eagerly():
            # Need to prepare non layer tensors again since
            # some tensors only become available on  batch end
            self._prepare_tensors_available_post_step()
            self._write_optimizer_variables()

        if self._prepared_tensors[mode]:
            if self._exported_collections is False:
                # in keras, these collections change when mode changes
                # but rest of the project isn't yet capable of handling this
                # this means that collections like outputs, or other collections with intermediate tensors
                # will only have tensor names from first mode

                # this means sometimes collections will be exported after 1 step
                self.export_collections()
                self._exported_collections = True

            if self._exported_model[self.mode] is False:
                # confirmed that keras has same graph for all modes
                # but we are writing it multiple times to keep behavior consistent with
                # estimator and to make it easier when seeing tensorboard
                self._export_model()
                self._exported_model[self.mode] = True

        self.profiler_config_parser.handle_step_end_python_profiling(mode, self.mode_steps[mode])

    @error_handling_agent.catch_smdebug_errors()
    def on_train_batch_end(self, batch, logs=None):
        self._on_any_batch_end(batch, ModeKeys.TRAIN, logs=logs)

    @error_handling_agent.catch_smdebug_errors()
    def on_test_batch_end(self, batch, logs=None):
        self._on_any_batch_end(batch, ModeKeys.EVAL, logs=logs)

    @error_handling_agent.catch_smdebug_errors()
    def on_predict_batch_end(self, batch, logs=None):
        self._on_any_batch_end(batch, ModeKeys.PREDICT, logs=logs)

    def wrap_optimizer(self, optimizer):
        """
        Wrapping your optimizer with this method enables finding gradient tensors and optimizer
        variables.

        :param optimizer: tf.train.Optimizer or tf.keras.optimizers.Optimizer
            the optimizer object used for training
        :return: Wrapped optimizer of same type as passed.
            This optimizer should be used for training
        """
        if isinstance(optimizer, tf.train.Optimizer):
            optimizer = self._wrap_apply_gradients(optimizer)
        elif isinstance(optimizer, tf.keras.optimizers.Optimizer) or is_keras_optimizer(optimizer):
            # either subclasse of optimizerV2 class in tf.keras
            # or keras.optimizers.Optimizer
            original_get_grads = optimizer.__class__.get_gradients

            def new_get_grads(opt, loss, params):
                grads = original_get_grads(opt, loss, params)
                self.set_gradients(gradients=grads)
                return grads

            optimizer.__class__.get_gradients = new_get_grads

            if isinstance(optimizer, tf.keras.optimizers.Optimizer):
                try:
                    original_add_weight = optimizer.__class__.add_weight

                    def new_add_weight(opt, *args, **kwargs):
                        var = original_add_weight(opt, *args, **kwargs)
                        self.set_optimizer_variables(var)
                        return var

                    optimizer.__class__.add_weight = new_add_weight
                except AttributeError:
                    # TF 1.13 Keras Optimizers have no add_weight attribute,
                    # so optimizer_variables is not supported
                    pass
        else:
            self._log_unsupported_optimizer(optimizer)
        # Optimizer is being saved to support additional features in the future.
        self.optimizer = optimizer
        return optimizer

    def _log_unsupported_tape(self, tape):
        self.logger.warning(
            f"Unsupported tape {tape} {tape.__class__}, cannot automatically find "
            "gradients, loss, weights, and biases."
        )

    def _unwrap_tape(self):
        """
        Unwrap the wrapped tape. Not doing so on hook cleanup or close,
        will lead to recursive wrapping when there are more tapes in the
        training script.
        """

        def _is_wrapper(f):
            return hasattr(f, "__wrapped__")

        def unwrap(func):
            while _is_wrapper(func):
                func = func.__wrapped__
            return func

        self.tape.__class__._push_tape = unwrap(self.tape.__class__._push_tape)
        self.tape.__class__._pop_tape = unwrap(self.tape.__class__._pop_tape)
        self.tape.__class__.gradient = unwrap(self.tape.__class__.gradient)

    @error_handling_agent.catch_smdebug_errors()
    def close(self):
        self._cleanup()
        self.profiler_config_parser.start_post_hook_close_python_profiling()

    def _cleanup(self):
        # Unwrap the tape before closing
        if self.tape:
            self._unwrap_tape()
        super()._cleanup()

    def _wrap_push_tape(self, function):
        """
        tape._push_tape is called at the beginning of the GradientTape block.
        Using this wrapper to prepare collections, initialize writers, and
        increment step.
        """

        def default_callback(*args, **kwargs):
            """
            Only call the original push tape if it isn't already recording.
            """
            if not self.tape._recording:
                return function(*args, **kwargs)

        @functools.wraps(function)
        @error_handling_agent.catch_smdebug_errors(default_return_val=default_callback)
        def run(*args, **kwargs):
            function(*args, **kwargs)
            if self._is_not_supported():
                return

            self.worker = self._get_worker_name()

            if self.writer is not None or len(self.writer_map):
                self._save_custom_tensors_post_step()
                self._close_writers()

            if not self.prepared_gradient_tape_collections:
                # at this point we need all collections to be ready
                # this may not be the case at creation of hook
                # as user's code after hook might add collections
                self.collection_manager.get(CollectionKeys.WEIGHTS).include(
                    "^weights/.*/((?!bias).)*$"
                )
                self.collection_manager.get(CollectionKeys.LOSSES).include(".*loss.*")
                self.collection_manager.get(CollectionKeys.GRADIENTS).include("^gradient")
                self._prepare_collections_for_tf2()
                self.prepared_gradient_tape_collections = True

            self._increment_step()

            if self._get_collections_to_save_for_step():
                self._initialize_writers()

            if self.last_saved_step is not None and self._exported_collections is False:
                # in keras, these collections change when mode changes
                # but rest of the project isn't yet capable of handling this
                # this means that collections like outputs, or other collections with intermediate tensors
                # will only have tensor names from first mode

                # this means sometimes collections will be exported after 1 step
                self.export_collections()
                self._exported_collections = True

        return run

    def _wrap_tape_gradient(self, function):
        """
        tape.gradient() is used to compute gradients from loss and model variables.
        Using this wrapper to get gradients, loss, weights, and bias values.
        """

        @functools.wraps(function)
        @error_handling_agent.catch_smdebug_errors(default_return_val=function)
        def run(*args, **kwargs):
            grads = function(*args, **kwargs)
            if self._is_not_supported():
                return grads
            loss = args[1]
            vars = args[2]
            if (
                (not grads or not vars)
                or (not isinstance(grads, list) or not isinstance(vars, list))
                or (
                    not (
                        (isinstance(vars[0], supported_tf_variables()))
                        and hasattr(vars[0], "numpy")
                    )
                )
                or (not ((isinstance(grads[0], tf.Tensor)) and hasattr(grads[0], "numpy")))
            ):
                return grads

            if self._get_collections_to_save_for_step():
                for (g, v) in zip(grads, vars):
                    layer = v.name.split(":")[0]
                    # Adding a check to make sure gradients are not None.
                    # gradients may be None if user tries to compute gradients for
                    # non-training variable when using model.variables instead of
                    # model.trainable_variables in tape.gradient().
                    # model.variables includes trainable and non-trainable
                    # variables.
                    if g is not None:
                        self._save_for_tensor(
                            tensor_name="gradients/" + layer + "Grad",
                            tensor_value=g,
                            check_before_write=True,
                        )
                    self._save_for_tensor(
                        tensor_name="weights/" + v.name,
                        tensor_value=v.value(),
                        check_before_write=True,
                    )

            self._write_optimizer_variables()
            self._save_layer_input_and_outputs()
            if not ((isinstance(loss, tf.Tensor)) and hasattr(loss, "numpy")):
                return grads
            self._add_metric(metric_name="loss", metric_value=loss)
            if self._is_collection_being_saved_for_step(CollectionKeys.LOSSES):
                self._initialize_writers(only_initialize_if_missing=True)
                self._save_for_tensor("loss", loss, check_before_write=False)

            return grads

        return run

    def _wrap_pop_tape(self, function):
        """
        tape._pop_tape() is called at the end of a GradientTape execution.
        Using this to export collections
        """

        def default_callback(*args, **kwargs):
            """
            Only call the original pop tape if it is already recording.
            """
            if self.tape._recording:
                return function(*args, **kwargs)

        @functools.wraps(function)
        @error_handling_agent.catch_smdebug_errors(default_return_val=default_callback)
        def run(*args, **kwargs):
            function(*args, **kwargs)
            if self._is_not_supported():
                return

            self.last_saved_step = self.step

        return run

    def _log_unsupported_tensors_in_non_eager_execution(self):
        if not self.has_logged_unsupported_tensors_in_non_eager_execution:
            self.logger.warning("cannot save model inputs and outputs in non-eager execution mode")
        self.has_logged_unsupported_tensors_in_non_eager_execution = True

    def save_tape_logs(self, model_inputs=None, outputs=None):
        """
        called by AWS TF to save model inputs and outputs
        :param model_inputs:
        :param outputs:
        :return:
        """
        logs = {ModelOutput.PREDICTIONS: outputs, ModelInput.INPUTS: model_inputs}
        if is_tf_version_2x() and tf.executing_eagerly():
            self.save_smdebug_logs(logs)
        else:
            self._log_unsupported_tensors_in_non_eager_execution()

    def wrap_tape(self, tape):
        @error_handling_agent.catch_smdebug_errors(default_return_val=tape)
        def _wrap_tape():
            """
            Wrapping your GradientTape with this method enables finding gradient tensors and optimizer
            variables.

            :param tape: tensorflow.python.eager.backprop.GradientTape
                the tape object used for training
            :return: Wrapped tape of same type as passed.
                This tape should be used for training
            """
            from tensorflow.python.eager.backprop import GradientTape

            self.set_mode(ModeKeys.TRAIN)

            if isinstance(tape, GradientTape):
                # unwrap tape before wrapping new tape to avoid recursive wrap tapes
                if self.tape:
                    self._unwrap_tape()

                self.tape = tape
                self.tape.__class__._push_tape = self._wrap_push_tape(tape.__class__._push_tape)
                self.tape.__class__.gradient = self._wrap_tape_gradient(tape.__class__.gradient)
                self.tape.__class__._pop_tape = self._wrap_pop_tape(tape.__class__._pop_tape)
            else:
                self._log_unsupported_tape(tape)
            return tape

        return _wrap_tape()

    def record_tensor_value(self, tensor_name, tensor_value):
        # To be used to save metrics of type EagerTensor
        if (
            not ((isinstance(tensor_value, tf.Tensor)) and hasattr(tensor_value, "numpy"))
        ) or self._is_not_supported():
            return

        self.logger.warning("This function has been deprecated. Please use the save_tensor API ")

        self._add_metric(metric_name=tensor_name, metric_value=tensor_value)
        if self._is_collection_being_saved_for_step(CollectionKeys.METRICS):
            self._initialize_writers(only_initialize_if_missing=True)
            self._save_for_tensor(tensor_name, tensor_value, check_before_write=False)

    def profiling_start_batch(self, mode=ModeKeys.TRAIN):
        """
        Enabling profiler at the start of train batch when native tf2 training is used.
        :param mode: ModeKeys.TRAIN ModeKeys.EVAL ModeKeys.PREDICT

        TODO: Add support for SMDDP profiling at the end of training.
        """
        self.start = time.time()

        if self._is_not_supported():
            return

        # When training with native TF2, `self.mode_steps[mode]` isn't incremented until the GradientTape callback
        # for `tape._push_tape` is called. So in this case, the current step is `self.mode_steps[mode] + 1`.
        current_step = self.mode_steps[mode]
        if self.is_profiler_enabled_for_native_training:
            current_step += 1

        self.profiler_config_parser.load_config()
        self._start_or_stop_detailed_profiling(current_step)
        self.profiler_config_parser.handle_step_start_python_profiling(mode, current_step)
        self._start_or_stop_dataloader_profiling(current_step)

    def profiling_end_batch(self, mode=ModeKeys.TRAIN):
        """
        Enabling profiler at the end of train batch for native Tf2 training.
        :param mode: ModeKeys.TRAIN ModeKeys.EVAL ModeKeys.PREDICT

        TODO: Add support for SMDDP profiling at the end of training.
        """
        if self._is_not_supported():
            return

        self.record_trace_events(
            training_phase="Step:" + str(mode),
            op_name="Step:" + str(mode),
            phase="X",
            timestamp=self.start,  # this is start time for step
            duration=time.time() - self.start,
            pid=os.getpid(),
            step_num=str(self.mode_steps[mode]),
        )

        self.profiler_config_parser.handle_step_end_python_profiling(mode, self.mode_steps[mode])

    def profiling_end(self):
        """
        Stop profiler at the end of training for native TF2.

        TODO: Add support for SMDDP profiling at the end of training.
        """
        # If the hook is closed twice, the process will hang.
        if self.is_hook_closed:
            return
        self.close()  # Unwrap the tape before closing
        self.is_hook_closed = True
        self._stop_detailed_profiling(self.mode_steps[self.mode])
        self.profiler_config_parser.stop_post_hook_close_python_profiling()
        self._stop_dataloader_profiling()
        self.is_profiler_enabled_for_native_training = False

    @contextlib.contextmanager
    def profiler(self, mode=ModeKeys.TRAIN):
        """
        Context manager to be inserted directly into the training script to enable profiling for native TF2 training.

        With respect to Python profiling, for a given step `n`, all code inside the context is profiled as during
        the step n. All code before the context is profiled as between steps n-1 and n. All code after the context
        is profiled as between steps n and n+1.

        Example:

        ```
        for epoch in range(n_epochs):
            for data, labels in dataset:
                setup()
                with hook.profiler():
                    labels = tf.one_hot(labels, depth=10)
                    with tf.GradientTape() as tape:
                        logits = train_step(data, labels)
                    grads = tape.gradient(logits, model.variables)
                    opt.apply_gradients(zip(grads, model.variables))
                cleanup()
        ```
        """
        _ = self.wrap_tape(
            tf.GradientTape()
        )  # GradientTape functions must be wrapped to correctly manage current step
        self.set_mode(mode)
        self.is_profiler_enabled_for_native_training = True
        self.profiling_start_batch(mode)
        yield
        self.profiling_end_batch(mode)
