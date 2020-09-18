# Standard Library

# Third Party
import tensorflow.compat.v1 as tf
from tensorflow.python.keras.backend import is_placeholder

# First Party
from smdebug.core.collection import CollectionKeys
from smdebug.core.tfevent.proto.summary_pb2 import Summary
from smdebug.core.tfevent.util import make_numpy_array
from smdebug.core.utils import match_inc

# Local
from .base_hook import TensorflowBaseHook
from .tensor_ref import TensorType
from .utils import (
    TFDistributionStrategy,
    build_fetches_tuple,
    extract_graph_summary,
    tensor_can_be_saved,
)


class SessionHook(tf.train.SessionRunHook, TensorflowBaseHook):
    def __init__(
        self,
        out_dir=None,
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
        """
        A class used to represent the hook which gets attached to the
        training process. This takes the form appropriate for the framework
        such as tf.train.SessionRunHook for TF, Callback for keras...

        ...

        Attributes
        ----------
        out_dir : str
            represents a path into which outputs will be written to
        dry_run : bool
            when dry run is set, behavior is only described in the log file.
            tensors are not actually saved.
        save_config: SaveConfig object
            Takes save config object which is applied as default for all included tensors.
            A collection can optionally have its own saveconfig object
            which overrides this for its tensors.

        reduction_config: ReductionConfig object
            if passed, this reduction config object is used
            as default for all tensors included.
            A collection has its own saveconfig object
            which overrides this for its tensors. if this is not passed,
            tensor is saved in full.

        include_regex: list of str
            takes as input the list of string representing regular expressions. Tensors whose names match
            these regular expressions will be saved. These tensors will be available as part of the `default`
            collection.

        include_collections: list of (str or collection objects)
            takes as input the collections which should be saved.
            if this is empty, it defaults to including all collections from code

        save_all: bool
            a shortcut for saving all tensors in the model.
            they are all saved in the collection `all`
        """
        super().__init__(
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
        # this holds the map from a fetches tuple to a dictionary of all
        # nodes in the subgraph that can reach any of the fetches
        self._subgraph_nodes = {}

        # for (tf_obj, fetches, unfilledplaceholder) -> bool
        # whether tensor depends on any unfilled placeholder
        # such tensors can not be queried
        self._tensor_placeholder_dependence = {}
        self._placeholder_tensors = set()

        self.graph = None
        self.tensors_to_save_this_step = None

    def _merge_tensor_refs_across_collections(self, tensor):
        # merge tensor objects in all collections which has this tensor
        # this ensures that whichever collection you query for this tensorname
        # it returns the same internal Tensor object
        tensor_ref = None
        for coll in self.tensor_to_collections[tensor.name]:
            if tensor_ref is None:
                tensor_ref = coll.get_tensor(tensor.name)
            else:
                coll.set_tensor_ref(tensor_ref)
        return tensor_ref

    def _add_tensor_to_matching_collections(self, tensor):
        """
        Finds which collections to add this tensor to, and adds tensor to them
        """
        colls_with_tensor = set()
        for coll in sorted(self._get_all_collections_to_save(), key=lambda x: x.name):
            variable_collections_with_tensor, processed = self._process_tensor_from_variable_read_op(
                tensor
            )
            if processed:
                colls_with_tensor.update(variable_collections_with_tensor)
                # processed=True means this tensor was either a variable read tensor,
                # or a tensor with same name as variable
                # former will be added to collections such as weights, biases, opt_variables
                # latter will be skipped as they refer to the same tensor
            else:
                # some collections are added automatically, don't match regex for these
                if coll.name not in [
                    CollectionKeys.WEIGHTS,
                    CollectionKeys.BIASES,
                    CollectionKeys.OPTIMIZER_VARIABLES,
                    CollectionKeys.TENSORFLOW_SUMMARIES,
                ] and match_inc(tensor.name, coll.include_regex):
                    coll.add(tensor)

                if coll.has_tensor(tensor.name):
                    # it must have been added when collection was added to
                    # from user(custom_coll)/library(losses, weights, grads)
                    tensor_ref = coll.get_tensor(tensor.name)
                    tensor_ref.tf_obj = tensor
                    colls_with_tensor.add(coll)

        # create entry in hook's tensor_to_collections map for this tensor
        self._create_tensors_for_matching_collections(tensor, colls_with_tensor)

    def _process_tensor_from_variable_read_op(self, tensor):
        """
        Returns a tuple of (collections_tensor_belongs_to, whether_tensor_was_processed)
        whether_tensor_was_processed is True if this tensor is from a variable
        (either representing the variable, or its read op)

        If tensor represents a variable (tensor.name == variable.name), such as w1:0
        then returns an empty set
        :param tensor:
        :return:
        """
        colls_with_tensor = set()
        processed = False
        for cname in [
            CollectionKeys.WEIGHTS,
            CollectionKeys.BIASES,
            CollectionKeys.OPTIMIZER_VARIABLES,
        ]:
            coll = self.collection_manager.get(cname)
            # Will contain w1/read:0 and w1:0
            read_op_names = coll.get_tensors_dict().keys()
            # w1:0, these will be
            export_names = coll.get_export_names_of_tensors()
            if tensor.name in read_op_names:
                tensor_ref = coll.get_tensor(tensor.name)
                # reassign tf_obj with same name from current graph
                tensor_ref.tf_obj = tensor
                colls_with_tensor.add(coll)
                processed = True
            elif tensor.name in export_names:
                processed = True
        return colls_with_tensor, processed

    def _create_tensors_for_matching_collections(self, tensor, colls_with_tensor):
        if colls_with_tensor:
            # this is mapping from graph name
            self.tensor_to_collections[tensor.name] = colls_with_tensor
            self._merge_tensor_refs_across_collections(tensor)

    def _check_and_add_tensor(self, tensor):
        if tensor.dtype == tf.resource or tensor.dtype == tf.variant:
            return

        if not self.graph.is_fetchable(tensor.op):
            return

        self._add_to_device_map(tensor)
        self._add_tensor_to_matching_collections(tensor)

    def _add_tensors(self):
        # so collections have save configs and reduction configs
        self._prepare_collections()

        self._add_losses()
        self._add_weights_and_biases()
        self._add_summaries_tensors()

        for op in self.graph.get_operations():
            for tensor in op.outputs:
                if is_placeholder(tensor):
                    self._placeholder_tensors.add(tensor)
                self._check_and_add_tensor(tensor)

        self._prepared_tensors[self.mode] = True

    def _add_summaries_tensors(self):
        if CollectionKeys.TENSORFLOW_SUMMARIES in self.include_collections:
            c = self.collection_manager.get(CollectionKeys.TENSORFLOW_SUMMARIES).add(
                tf.get_collection(tf.GraphKeys.SUMMARIES)
            )
            for t in c.get_tensors(graph=self.graph):
                t.type = TensorType.SUMMARY
                t.original_tensor = t.op.inputs[1]

    def _add_weights_and_biases(self):
        wts = tf.trainable_variables()
        # variable such as <tf.Var w1:0>
        for w in wts:
            if match_inc(w.name, self.collection_manager.get(CollectionKeys.BIASES).include_regex):
                self.collection_manager.get(CollectionKeys.BIASES).add(w)
            else:
                # adds a tensor_ref with name `w1/read:0` and export_name `w1:0`
                self.collection_manager.get(CollectionKeys.WEIGHTS).add(w)

    def _add_losses(self):
        losses = tf.losses.get_losses()
        self.collection_manager.get(CollectionKeys.LOSSES).add(losses)

    def _is_not_supported(self):
        if self.distribution_strategy is None:
            self.distribution_strategy = self._get_distribution_strategy()
        if self._hook_supported is None:
            self._hook_supported = True
            if self.distribution_strategy == TFDistributionStrategy.MIRRORED:
                from packaging import version

                if version.parse(tf.__version__) < version.parse("1.14.0"):
                    self._hook_supported = False
                    # in tf 1.13, we can't support mirrored strategy as
                    # MirroredVariable does not have _values attribute
                    self.logger.info(
                        "Disabling SMDebug as it does not support mirrored strategy"
                        "with TensorFlow version <1.14"
                    )
            elif self.distribution_strategy == TFDistributionStrategy.UNSUPPORTED:
                self.logger.info(
                    f"Disabling SMDebug as it does not support " f"{tf.distribute.get_strategy()}"
                )
                self._hook_supported = False
        return not self._hook_supported

    def _clear_cached_state(self):
        self._subgraph_nodes = {}
        self._tensor_placeholder_dependence = {}
        self._placeholder_tensors = set()
        # assuming that graph changes when begin is called again (holds for estimator)
        # so when graph changes we need to update tensors
        # in gradients collection to have new graph tensors
        # setting this to False means that on next apply_gradients/get_grads gradients will be set again
        self._gradients_set = False

    def begin(self):
        if self._is_not_supported():
            return

        # clear all caches so we don't interfere with other modes
        self._clear_cached_state()

        # todo: use global step from TF instead of internal steps
        # todo: handle multiple graphs in the model
        self.worker = self._get_worker_name()
        self.graph = tf.get_default_graph()

        self._add_tensors()
        self._set_chief_worker()

        if self._exported_model[self.mode] is False:
            self._export_model()
            self._exported_model[self.mode] = True

        if self._exported_collections is False:
            self.export_collections()
            self._exported_collections = True

    def _get_tensors_to_save_this_step(self) -> set:
        tensors_to_save = set()
        for coll in self._get_collections_to_save_for_step():
            tensors_to_save.update(coll.get_tensors(graph=self.graph))
        return tensors_to_save

    def _get_subgraph_which_reach_fetches(self, fetches_ops_tuple):
        if fetches_ops_tuple in self._subgraph_nodes:
            subgraph_nodes = self._subgraph_nodes[fetches_ops_tuple]
        else:
            dest_names = [n.name for n in fetches_ops_tuple]
            subgraph = tf.graph_util.extract_sub_graph(self.graph.as_graph_def(), dest_names)
            _, subgraph_nodes, _ = extract_graph_summary(subgraph)
            self._subgraph_nodes[fetches_ops_tuple] = subgraph_nodes
        return subgraph_nodes

    def _is_tensor_dependent_on_unfilled_placeholder(
        self, tensor_obj, fetches_ops_tuple, unfilled_placeholders
    ):
        # making this a class method so we can cache this result
        # making set a tuple so we can hash it
        key = (
            tensor_obj,
            fetches_ops_tuple,
            tuple(sorted(list(unfilled_placeholders), key=lambda x: x.name)),
        )
        if key not in self._tensor_placeholder_dependence:
            subgraph_nodes = self._get_subgraph_which_reach_fetches(fetches_ops_tuple)
            self._tensor_placeholder_dependence[key] = tensor_can_be_saved(
                tensor_obj, subgraph_nodes, unfilled_placeholders
            )
        return self._tensor_placeholder_dependence[key]

    def _filter_to_be_saved(self, tensors_to_save: set, feeds, fetches) -> set:
        """
        :param tensors_to_save:
        :param feeds: a dictionary from tensor to value
        :param fetches: a nested list/tuple/dict of tensors/ops
        :return:
        """
        fetches_ops_tuple = build_fetches_tuple(fetches)
        unfilled_placeholders = set()
        for placeholder in self._placeholder_tensors:
            if feeds is None or placeholder not in feeds:
                unfilled_placeholders.add(placeholder)

        filtered = set()
        skipped = set()
        for tensor_ref in tensors_to_save:
            if self._is_tensor_dependent_on_unfilled_placeholder(
                tensor_ref.tf_obj, fetches_ops_tuple, unfilled_placeholders
            ):
                filtered.add(tensor_ref.tf_obj)
            else:
                skipped.add(tensor_ref.tf_obj)
        if len(skipped) > 0:
            self.logger.debug(f"Skipped {len(skipped)} unreachable tensors: {skipped}")
        self.logger.debug(f"Saving {len(filtered)} tensors: {filtered}")
        return filtered

    def before_run(self, run_context):
        if self._is_not_supported():
            return
        tensors_to_save = self._get_tensors_to_save_this_step()
        if tensors_to_save:
            if run_context:
                tensors_to_save = self._filter_to_be_saved(
                    tensors_to_save,
                    run_context.original_args.feed_dict,
                    run_context.original_args.fetches,
                )
            self.tensors_to_save_this_step = list(tensors_to_save)
            return tf.train.SessionRunArgs(self.tensors_to_save_this_step)
        else:
            self.tensors_to_save_this_step = list(tensors_to_save)
            return None

    def _write_tf_summary(self, tensor, value):
        try:
            # likely a summary
            self.logger.debug(f"Saving summary {tensor.name} with length {len(value)}")
            s = Summary.FromString(value)
            tb_writer = self._maybe_get_tb_writer()
            if tb_writer:
                tb_writer.write_summary(s, self.step)
        except Exception as e:
            # can it not be a summary?
            self.logger.debug(f"Ran into the exception when saving {tensor}: {e}")

    def _get_all_tensors_values(self, results):
        for (item, value) in zip(self.tensors_to_save_this_step, results):
            if not isinstance(value, list) or isinstance(value, tuple):
                assert not (isinstance(item, list) or isinstance(item, tuple))
                yield item, value
            elif isinstance(value, list) or isinstance(value, tuple):
                assert isinstance(item, list) or isinstance(item, tuple)
                for i in range(len(value)):
                    yield item[i], value[i]

    def after_run(self, run_context, run_values):
        if self._is_not_supported():
            return
        if self.tensors_to_save_this_step:
            self._initialize_writers()
            for (tensor, value) in self._get_all_tensors_values(run_values.results):
                if tensor.dtype == tf.string:
                    self._write_tf_summary(tensor, value)
                else:
                    self._save_for_tensor(tensor.name, value, check_before_write=False)
            self._close_writers()
        self._close_tb_writer()
        self._increment_step()

    @staticmethod
    def _make_numpy_array(tensor_value):
        """
        Convert the tensor value into a numpy array.
        Here it's already numpy array
        """
        return make_numpy_array(tensor_value)

    def end(self, sess):
        pass

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
        else:
            self._log_unsupported_optimizer(optimizer)
        self.optimizer = optimizer
        return optimizer


# to make it clear for estimator users
EstimatorHook = SessionHook
