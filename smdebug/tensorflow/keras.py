# Standard Library
import functools

# Third Party
from tensorflow.python.distribute import values

# First Party
from smdebug.core.modes import ModeKeys
from smdebug.core.utils import match_inc
from smdebug.tensorflow.callable_cache import CallableCache

# Local
from .base_hook import TensorflowBaseHook
from .collection import CollectionKeys
from .tensor_ref import TensorRef, get_tf_names
from .utils import (
    TFDistributionStrategy,
    get_export_name_for_keras,
    get_keras_layer_inputs,
    get_keras_layer_outputs,
    get_keras_mode,
    is_keras_optimizer,
)

try:
    # as most of the v1 API is deprecated from the main tf namespace from 1.14
    import tensorflow.compat.v1 as tf
except ImportError:
    # For TF 1.13
    import tensorflow as tf


class KerasHook(TensorflowBaseHook, tf.keras.callbacks.Callback):
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
        super().__init__(
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
        self.tensor_refs_to_save_this_step = set()
        self._fetches_added = set()
        self.callable_cache = CallableCache()

    def _is_not_supported(self):
        if self.distribution_strategy is None:
            self.distribution_strategy = self._get_distribution_strategy()
        if self._hook_supported is None:
            self._hook_supported = True
            if tf.executing_eagerly() or (
                hasattr(self.model, "run_eagerly") and self.model.run_eagerly
            ):
                self.logger.info("Disabling SMDebug as it does not support eager mode")
                self._hook_supported = False
            elif self.distribution_strategy == TFDistributionStrategy.MIRRORED:
                try:
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
            if current_coll.name in [CollectionKeys.WEIGHTS, CollectionKeys.BIASES]:
                # don't match regex for these as these are added specially above
                # we also don't want users to make mistakes configuring these collections
                continue

            if match_inc(ts_name, current_coll.include_regex):
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
                    if isinstance(tensor, tf.Variable):
                        tensor_refs.append(
                            coll.add_variable(tensor, export_name=export_name, mode=mode)
                        )
                    elif isinstance(tensor, tf.Tensor):
                        tensor_refs.append(coll.add_tensor(tensor, name=export_name, mode=mode))
                    elif isinstance(tensor, values.MirroredVariable):
                        tensor_refs.extend(
                            coll.add_mirrored_variable(tensor, export_name=export_name, mode=mode)
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

    def _is_input_layer(self, mode, layer_inputs):
        model_inputs = []
        if self.distribution_strategy == TFDistributionStrategy.MIRRORED:
            model = self._get_distributed_model(mode)
        else:
            model = self.model
        # when in mirrored strategy
        if hasattr(model, "values"):
            for per_replica_model in model.values:
                model_inputs.extend(per_replica_model.inputs)
        else:
            model_inputs.extend(model.inputs)
        return any([i in model_inputs for i in layer_inputs])

    def _is_output_layer(self, mode, layer_outputs):
        model_outputs = []
        if self.distribution_strategy == TFDistributionStrategy.MIRRORED:
            model = self._get_distributed_model(mode)
        else:
            model = self.model
        # when in mirrored strategy
        if hasattr(model, "values"):
            for per_replica_model in model.values:
                model_outputs.extend(per_replica_model.outputs)
        else:
            model_outputs.extend(model.outputs)
        return any([i in model_outputs for i in layer_outputs])

    def _prepare_layers(self, mode):
        # adds any layer tensor (input, output and weight) to appropriate collection
        for layer in self.model.layers:
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

            weights = layer.weights
            for w in weights:
                self._check_and_add_layer_tensor(mode, layer, "weight", w)

    def _prepare_non_layer_tensors(self):
        # for gradients, optimizer_variables
        for coll in self.collection_manager.get_collections().values():
            for tensor_ref in coll.get_tensors():
                if tensor_ref.name not in self.tensor_to_collections:
                    self.tensor_to_collections[tensor_ref.name] = {coll}
                elif coll not in self.tensor_to_collections[tensor_ref.name]:
                    self.tensor_to_collections[tensor_ref.name].add(coll)

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

    def _save_inputs(self, check_before_write=True):
        # TODO
        pass

    def _add_metric(self, metric_name):
        if metric_name in self.tensor_to_collections:
            return

        if metric_name in ["loss", "val_loss"]:
            coll_name = CollectionKeys.LOSSES
        else:
            coll_name = CollectionKeys.METRICS
        coll = self.collection_manager.get(coll_name)
        coll.set_tensor_ref(TensorRef.from_non_graph_var(metric_name))
        self.tensor_to_collections[metric_name] = {coll}

    def _save_metrics(self, batch, logs, force_save=False):
        # if force_save is True, doesn't check whether collection needs to be saved for steps
        if logs is None:
            return

        if force_save or self._is_collection_being_saved_for_step(CollectionKeys.METRICS):
            self._initialize_writers(only_initialize_if_missing=True)
            logs["batch"] = batch
            for key in logs:
                if key in ["loss", "val_loss", "outputs"]:
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

    def _save_tensors_post_step(self, batch, logs):
        # some tensors available as value from within hook are saved here
        # weights, metrics
        self._save_metrics(batch, logs)

        if self._is_collection_being_saved_for_step(CollectionKeys.INPUTS):
            self._save_inputs(check_before_write=False)

    def _get_exec_function(self, mode):
        if self.distribution_strategy in [
            TFDistributionStrategy.NONE,
            TFDistributionStrategy.HOROVOD,
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

    def on_epoch_begin(self, batch, logs=None):
        pass

    def on_epoch_end(self, batch, logs=None):
        if self._is_not_supported():
            return
        self._save_metrics(batch=batch, logs=logs, force_save=True)
        self._close_writers()

    def _on_any_mode_begin(self, mode):
        if self._is_not_supported():
            return
        self.worker = self._get_worker_name()
        self.graph = tf.get_default_graph()
        self.set_mode(mode)

        # have to clear callable cache if we are not caching per mode
        self.callable_cache.change_mode()

    def on_train_begin(self, logs=None):
        self._on_any_mode_begin(ModeKeys.TRAIN)

    def on_test_begin(self, logs=None):
        self._on_any_mode_begin(ModeKeys.EVAL)

    # throws error in keras if this fn is absent
    def on_test_end(self, logs=None):
        pass

    # throws error in keras if this fn is absent
    def on_predict_end(self, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        self._on_any_mode_begin(ModeKeys.PREDICT)

    def _on_any_batch_begin(self, batch, mode, logs=None):
        if self._is_not_supported():
            return

        # set mode for each batch as when users run model.fit() and pass validation data
        # through the optional argument, then mode_begin is not called for the training steps
        # after first evaluation during training
        self.set_mode(mode)

        self._close_writers()
        self._increment_step()

        if self.prepared_collections is False:
            # sets prepared_collections to True here
            self._prepare_collections()

        if self._prepared_tensors[mode] is False:
            if self._validate_exec_function(self._get_exec_function(mode)):
                self._prepare_layers(mode)
                self._prepare_non_layer_tensors()
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

            self._add_callbacks(mode)

    def on_train_batch_begin(self, batch, logs=None):
        self._on_any_batch_begin(batch, ModeKeys.TRAIN, logs=logs)

    def on_test_batch_begin(self, batch, logs=None):
        self._on_any_batch_begin(batch, ModeKeys.EVAL, logs=logs)

    def on_predict_batch_begin(self, batch, logs=None):
        self._on_any_batch_begin(batch, ModeKeys.PREDICT, logs=logs)

    def _on_any_batch_end(self, batch, mode, logs=None):
        if self._is_not_supported():
            return

        self._remove_fetches_and_callbacks(mode)
        self._save_tensors_post_step(batch, logs)

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

    def on_train_batch_end(self, batch, logs=None):
        self._on_any_batch_end(batch, ModeKeys.TRAIN, logs=logs)

    def on_test_batch_end(self, batch, logs=None):
        self._on_any_batch_end(batch, ModeKeys.EVAL, logs=logs)

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
