# Standard Library
import os

# Third Party
import tensorflow as tf
from tensorflow.python.distribute import values

# First Party
from smdebug.core.collection import SUMMARIES_COLLECTIONS, CollectionKeys
from smdebug.core.reductions import get_reduction_tensor_name
from smdebug.core.tfevent.proto.summary_pb2 import Summary
from smdebug.core.tfevent.util import make_numpy_array
from smdebug.core.utils import match_inc

# Local
from .base_hook import TensorflowBaseHook
from .reductions import get_tensorflow_reduction
from .tensor_ref import TensorType
from .utils import TFDistributionStrategy, extract_graph_summary, get_original_fetch_ops, node_name


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
    ):
        """
        A class used to represent the hook which gets attached to the
        training process. This takes the form appropriate for the framework
        such as tf.train.SessionRunHook for TF, Callback for keras...

        ...

        Attributes
        ----------
        out_dir : str
            represents a path into which tornasole outputs will be written to
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
        )
        self.subgraph_nodes_cache = {}
        self.graph = None
        self.tensors_to_save_this_step = None

    def _add_reduction(self, tensor_ref, reduction_name, collection, on_absolute_values=False):
        if tensor_ref.tf_obj.dtype in [tf.bool, tf.string]:
            return

        tname = get_reduction_tensor_name(
            tensor_ref.export_name, reduction_name, on_absolute_values, remove_colon_index=False
        )
        red_tensor = get_tensorflow_reduction(
            reduction_name, tensor_ref.tf_obj, on_absolute_values=on_absolute_values
        )
        collection.add_reduction_tensor(
            tensor=red_tensor, original_tensor=tensor_ref.tf_obj, export_name=tname
        )
        # since tensor_to_collections is a mapping from graph name
        if red_tensor.name not in self.tensor_to_collections:
            self.tensor_to_collections[red_tensor.name] = {collection}
        else:
            self.tensor_to_collections[red_tensor.name].add(collection)

    def _add_reductions(self, tensor_ref, collection):
        reduction_config = collection.reduction_config
        for reduction_list in (reduction_config.reductions, reduction_config.norms):
            for reduction in reduction_list:
                self._add_reduction(tensor_ref, reduction, collection, False)
        for reduction_list in (reduction_config.abs_reductions, reduction_config.abs_norms):
            for reduction in reduction_list:
                self._add_reduction(tensor_ref, reduction, collection, True)

    def _merge_tensor_refs_across_collections(self, tensor):
        # merge tensor objects in all collections which has this tensor
        # this ensures that whichever collection you query for this tensorname
        # it returns the same Tornasole Tensor object
        tensor_ref = None
        for coll in self.tensor_to_collections[tensor.name]:
            if tensor_ref is None:
                tensor_ref = coll.get_tensor(tensor.name)
            else:
                coll.set_tensor_ref(tensor_ref)
        return tensor_ref

    def _get_matching_collections(self, tensor):
        colls_with_tensor = set()
        for coll in self._get_all_collections_to_save():
            # some collections are added automatically, don't match regex for these
            if coll.name not in [
                CollectionKeys.WEIGHTS,
                CollectionKeys.BIASES,
                CollectionKeys.TENSORFLOW_SUMMARIES,
            ] and match_inc(tensor.name, coll.include_regex):
                coll.add(tensor)

            if coll.has_tensor(tensor.name):
                # it must have been added when collection was added to
                # from user(custom_coll)/library(losses, weights, grads)
                tensor_ref = coll.get_tensor(tensor.name)
                tensor_ref.tf_obj = tensor
                colls_with_tensor.add(coll)
        return colls_with_tensor

    def _create_tensors_for_matching_collections(self, tensor, colls_with_tensor):
        if colls_with_tensor:
            # this is mapping from graph name
            self.tensor_to_collections[tensor.name] = colls_with_tensor
            tensor_ref = self._merge_tensor_refs_across_collections(tensor)
            for coll in colls_with_tensor:
                # this should be after we add tensor.name to tensor_to_collections
                self._add_reductions(tensor_ref, coll)

    def _check_and_add_tensor(self, tensor):
        if tensor.dtype == tf.resource or tensor.dtype == tf.variant:
            return False

        if not self.graph.is_fetchable(tensor.op):
            return False

        if isinstance(tensor, values.MirroredVariable):
            tensors = [t for t in tensor._values]
        else:
            tensors = [tensor]

        for t in tensors:
            self._add_to_device_map(t)
            colls_with_tensor = self._get_matching_collections(t)
            self._create_tensors_for_matching_collections(t, colls_with_tensor)

    def _add_tensors(self):
        # so collections have save configs and reduction configs
        self._prepare_collections()

        for op in self.graph.get_operations():
            for tensor in op.outputs:
                self._check_and_add_tensor(tensor)

        # need to do this for mirrored strategy
        for variable in tf.global_variables():
            self._check_and_add_tensor(variable)

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
        for w in wts:
            if match_inc(w.name, self.collection_manager.get(CollectionKeys.BIASES).include_regex):
                self.collection_manager.get(CollectionKeys.BIASES).add(w)
            else:
                self.collection_manager.get(CollectionKeys.WEIGHTS).add(w)

    def begin(self):
        # todo: should this be called first time a mode changes
        # todo: handle multiple graphs in the model
        self.worker = self.get_worker_name()
        self.distribution_strategy = self.get_distribution_strategy()
        self.graph = tf.get_default_graph()

        self._add_weights_and_biases()

        losses = tf.losses.get_losses()
        self.collection_manager.get(CollectionKeys.LOSSES).add(losses)

        # assuming that graph changes when begin is called again (holds for estimator)
        # so when graph changes we need to update tensors
        # in gradients collection to have new graph tensors
        # setting this to False means that on next apply_gradients/get_grads gradients will be set again
        self._gradients_set = False

        self._add_summaries_tensors()
        self._add_tensors()

        self._export_model()
        self.export_collections()

    def _get_tensors_to_save_this_step(self):
        tensors_to_save = set()
        for coll in self._get_collections_to_save_for_step():
            tensors_to_save.update(coll.get_tensors(graph=self.graph))
        return list(tensors_to_save)

    def _filter_to_be_saved(self, tensors_to_save, fetches):
        # todo: handle all types of complex fetches
        if (
            not isinstance(fetches, list)
            and not isinstance(fetches, tuple)
            and not isinstance(fetches, dict)
        ):
            fetches = [fetches]
        fetches_tuple = tuple(fetches)
        if fetches_tuple in self.subgraph_nodes_cache:
            subgraph_nodes = self.subgraph_nodes_cache[fetches_tuple]
        else:
            original_fetch_ops = get_original_fetch_ops(fetches)
            dest_names = [n.name for n in original_fetch_ops]
            subgraph = tf.graph_util.extract_sub_graph(self.graph.as_graph_def(), dest_names)
            _, subgraph_nodes, _ = extract_graph_summary(subgraph)
            self.subgraph_nodes_cache[fetches_tuple] = subgraph_nodes

        # this also allows us to skip all the assign, read, initial_value,
        # control_dependency nodes in the graph
        # check that this run includes the ops whose tensors are to be saved
        filtered = []
        skipped = []
        for tensor_ref in tensors_to_save:
            if tensor_ref.type == TensorType.REGULAR:
                if node_name(tensor_ref.tf_obj.name) in subgraph_nodes:
                    filtered.append(tensor_ref.tf_obj)
                else:
                    skipped.append(tensor_ref.tf_obj)
            else:
                if tensor_ref.type == TensorType.VARIABLE:
                    tf_obj = tensor_ref.tf_obj
                    original_tensor_name = tensor_ref.original_tensor.name
                else:
                    tf_obj = tensor_ref.tf_obj
                    original_tensor_name = tensor_ref.original_tensor.name
                if node_name(original_tensor_name) in subgraph_nodes:
                    filtered.append(tf_obj)
                else:
                    skipped.append(tf_obj)
        if len(skipped) > 0:
            self.logger.debug(f"Skipped {len(skipped)} unreachable tensors: {skipped}")
        return filtered

    def before_run(self, run_context):
        tensors_to_save = self._get_tensors_to_save_this_step()
        if tensors_to_save:
            if run_context:
                tensors_to_save = self._filter_to_be_saved(
                    tensors_to_save, run_context.original_args.fetches
                )
            self.tensors_to_save_this_step = tensors_to_save
            return tf.train.SessionRunArgs(tensors_to_save)
        else:
            self.tensors_to_save_this_step = tensors_to_save
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
            self.logger.error(f"Ran into the exception when saving {tensor}: {e}")

    def _write_reductions(self, tensor_name, tensor_value, save_collections, **kwargs):
        # skip writing reductions as TF Session/Estimator handles them in the graph itself
        pass

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
        if self.tensors_to_save_this_step:
            self._initialize_writer()
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
        Wrapping your optimizer with this method allows Tornasole to
        find gradient tensors and optimizer variables.

        :param optimizer: tf.train.Optimizer or tf.keras.optimizers.Optimizer
            the optimizer object used for training
        :return: Tornasole aware optimizer of same type as passed.
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
