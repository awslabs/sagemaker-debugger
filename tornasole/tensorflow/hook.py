import tensorflow as tf
from .utils import *
from .reductions import get_tensorflow_reduction
from .collection import get_collection_manager
from tornasole.core.tfevent.proto.summary_pb2 import Summary
from tornasole.core.utils import match_inc
from tornasole.core.collection import CollectionKeys, SUMMARIES_COLLECTIONS
from tornasole.core.hook import BaseHook
from tornasole.core.reductions import get_reduction_tensor_name
from tornasole.core.json_config import (
    TORNASOLE_CONFIG_DEFAULT_WORKER_NAME,
    create_hook_from_json_config,
)
from tornasole.tensorflow.singleton_utils import set_hook


DEFAULT_INCLUDE_COLLECTIONS = [
    CollectionKeys.WEIGHTS,
    CollectionKeys.GRADIENTS,
    CollectionKeys.DEFAULT,
    CollectionKeys.LOSSES,
    CollectionKeys.SCALARS,
]


class TornasoleHook(tf.train.SessionRunHook, BaseHook):
    def __init__(
        self,
        out_dir=None,
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
        worker: string
            name of worker in a multi process training job
            outputs and tensors are organized by this name during retrieval.

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
            collection_manager=get_collection_manager(),
            default_include_collections=DEFAULT_INCLUDE_COLLECTIONS,
            out_dir=out_dir,
            dry_run=dry_run,
            reduction_config=reduction_config,
            save_config=save_config,
            include_regex=include_regex,
            include_collections=include_collections,
            save_all=save_all,
        )
        self.reduction_original_tensors = {}
        self.subgraph_nodes_cache = {}
        self.summaries_original_tensors = {}
        self.graph = None
        self.tensors_to_save_this_step = None

        set_hook(self)

    def get_worker_name(self):
        try:
            import horovod.tensorflow as hvd

            if hvd.size():
                return f"worker_{hvd.rank()}"
        except (ModuleNotFoundError, ValueError, ImportError):
            return TORNASOLE_CONFIG_DEFAULT_WORKER_NAME

    def get_num_workers(self):
        try:
            import horovod.tensorflow as hvd

            if hvd.size():
                return hvd.size()
        except (ModuleNotFoundError, ValueError, ImportError):
            return 1

    @classmethod
    def hook_from_config(cls, json_config_path=None):
        return create_hook_from_json_config(
            cls, get_collection_manager(), json_config_path=json_config_path
        )

    def _prepare_tensors(self):
        """
        If collections have already been populated with tensors,
        then we can add them to the tensor_to_collections map
        """
        for c in self._get_all_collections_to_save():
            for t_list in (c.tensors, c.reduction_tensors_added):
                for t in t_list:
                    if t.name not in self.tensor_to_collections:
                        self.tensor_to_collections[t.name] = {c}
                    else:
                        self.tensor_to_collections[t.name].add(c)

    def _process_matched_tensor(self, tensor, collection):
        reduction_config = collection.reduction_config
        if reduction_config:
            for reduction_list in (reduction_config.reductions, reduction_config.norms):
                for reduction in reduction_list:
                    self._add_reduction(tensor, reduction, collection, False)
            for reduction_list in (reduction_config.abs_reductions, reduction_config.abs_norms):
                for reduction in reduction_list:
                    self._add_reduction(tensor, reduction, collection, True)

        if reduction_config.save_raw_tensor is False:
            # this is for export so that trial knows that this tensor
            # belongs to this collection
            collection.add_tensor_name(tensor.name)

        case1 = collection.name in [CollectionKeys.SCALARS, CollectionKeys.LOSSES]
        case2 = collection.name in self.include_collections and collection.save_histogram is True
        case3 = reduction_config.save_raw_tensor is True
        if case1 or case2 or case3:
            collection.add(tensor)

    def _check_and_add_tensor(self, t):
        if t.dtype == tf.resource or t.dtype == tf.variant:
            return False

        if not self.graph.is_fetchable(t.op):
            return False

        added = False
        for coll in self._get_all_collections_to_save():
            if coll.name in SUMMARIES_COLLECTIONS:
                # these summaries directly does not take any regex patterns
                # or tensor names. it just takes them from tf.summaries
                # or added when other collections have write_histograms True
                # this also ensures that we don't look at reductions for
                # these collections
                # note that for scalars collection we do
                # look at regex patterns
                continue

            if match_inc(t.name, coll.include_regex) or t.name in coll.tensor_names:
                self._process_matched_tensor(t, coll)
                added = True
        return added

    def _add_reduction(self, tensor, reduction_name, collection, abs=False):
        if tensor.dtype in [tf.bool, tf.string]:
            return
        tname = get_reduction_tensor_name(tensor.name, reduction_name, abs)
        red_tensor = get_tensorflow_reduction(reduction_name, tensor, tname, abs=abs)
        self.reduction_original_tensors[red_tensor.name] = tensor
        collection.add_reduction_tensor(red_tensor, original_tensor=tensor)

    def _add_tensors(self):
        # gradients and optimizer_variables added
        # in user code or TornasoleOptimizer
        total_tensor_count = 0
        # todo: do we ever need inputs of the op
        for op in self.graph.get_operations():
            for tensor in op.outputs:
                self._check_and_add_tensor(tensor)
                total_tensor_count += 1
        return total_tensor_count

    def _add_summaries_tensors(self):
        if CollectionKeys.TENSORFLOW_SUMMARIES in self.include_collections:
            c = self.collection_manager.get(CollectionKeys.TENSORFLOW_SUMMARIES)
            for t in tf.get_collection(tf.GraphKeys.SUMMARIES):
                self.summaries_original_tensors[t.name] = t.op.inputs[1]
                c.add(t)

    def _clear_tensor_objects_from_collections(self):
        for coll_name, coll in self.collection_manager.get_collections().items():
            # to make multiple graphs work with the same tensor names
            # this can happen when we use same hook for training and evaluation
            # what is going on here is that we clear the tensors and reduction tensors
            # but we use the tensor names field in collection to readd tensors
            # from the new graph to the collection so we can them right
            coll.tensors = []
            coll.reduction_tensors_added = []

    def begin(self):
        # todo: should this be called first time a mode changes
        # todo: handle multiple graphs in the model
        self.graph = tf.get_default_graph()

        wts = tf.trainable_variables()
        self.collection_manager.get(CollectionKeys.WEIGHTS).add(wts)

        losses = tf.losses.get_losses()
        self.collection_manager.get(CollectionKeys.LOSSES).add(losses)

        # so that new graph does not cause conflicts for the graph
        # property of tensors added
        self._clear_tensor_objects_from_collections()

        # needs this so save_config for collection can be used in add_tensors
        self._prepare_collections()

        self._add_summaries_tensors()
        self._add_tensors()

        # after all tensors are added
        self._prepare_tensors()

        for coll in self._get_all_collections_to_save():
            self.logger.info(
                f"Saving the collection {coll.name} with {len(coll.tensors)} tensors "
                f"and {len(coll.reduction_tensors_added)} reductions"
            )
            self.logger.debug(f"  Collection {coll.name} has tensors: {coll.tensors}")
            self.logger.debug(
                f"  Collection {coll.name} has reductions: {coll.reduction_tensors_added}"
            )

        self._export_model()
        self.export_collections()

    def _export_model(self):
        self.logger.info("Writing graph")
        self._get_tb_writer().write_graph(self.graph.as_graph_def(add_shapes=True))
        # self._close_tb_writer()

    def _get_tensors_to_save_this_step(self):
        tensors_to_save = {"watched": [], "added": []}
        for coll in self._get_collections_to_save_for_step():
            tensors_to_save["watched"].extend(coll.tensors)
            tensors_to_save["added"].extend(coll.reduction_tensors_added)
        # dedup watched and added
        tensors_to_save["watched"] = list(set(tensors_to_save["watched"]))
        tensors_to_save["added"] = list(set(tensors_to_save["added"]))
        return tensors_to_save

    def _filter_to_be_saved(self, dict_to_save, fetches):
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
            subgraph = tf.graph_util.extract_sub_graph(
                tf.get_default_graph().as_graph_def(), dest_names
            )
            _, subgraph_nodes, _ = extract_graph_summary(subgraph)
            self.subgraph_nodes_cache[fetches_tuple] = subgraph_nodes

        # this also allows us to skip all the assign, read, initial_value,
        # control_dependency nodes in the graph
        # check that this run includes the ops whose tensors are to be saved
        filtered = []
        skipped = []
        for tensor in dict_to_save["watched"]:
            if node_name(tensor.name) in subgraph_nodes:
                filtered.append(tensor)
            elif tensor.name in self.summaries_original_tensors.keys():
                ot = self.summaries_original_tensors[tensor.name]
                if node_name(ot.name) in subgraph_nodes:
                    # summary should only be included
                    # if the original tensor is included for safety
                    filtered.append(tensor)
                else:
                    skipped.append(tensor)
            else:
                skipped.append(tensor)
        for tensor in dict_to_save["added"]:
            assert isinstance(tensor, tf.Tensor)
            original_tensor = self.reduction_original_tensors[tensor.name]
            if node_name(original_tensor.name) in subgraph_nodes:
                filtered.append(tensor)
            else:
                skipped.append(tensor)

        if len(skipped) > 0:
            self.logger.debug(f"Skipped {len(skipped)} unreachable tensors: {skipped}")

        # todo(huilgolr) can we filter tensors with (0)shape here. do we want to?
        return filtered

    def before_run(self, run_context):
        tensors_to_save = self._get_tensors_to_save_this_step()
        if len(tensors_to_save["watched"]) + len(tensors_to_save["added"]) > 0:
            if run_context:
                list_to_save = self._filter_to_be_saved(
                    tensors_to_save, run_context.original_args.fetches
                )
            else:
                list_to_save = tensors_to_save["watched"] + tensors_to_save["added"]
        else:
            list_to_save = []
        self.tensors_to_save_this_step = list_to_save
        return tf.train.SessionRunArgs(list_to_save) if list_to_save else None

    def _write_tf_summary(self, tensor, value):
        try:
            # likely a summary
            self.logger.debug(f"Saving summary {tensor.name} with length {len(value)}")
            s = Summary.FromString(value)
            self._get_tb_writer().write_summary(s, self.step)
        except Exception as e:
            # can it not be a summary?
            self.logger.error(f"Ran into the exception when saving {tensor}: {e}")

    def _write_for_tensor(self, tensor_name, tensor_value, save_collections):
        # if reduction tensor
        if tensor_name in self.reduction_original_tensors:
            self._write_raw_tensor_simple(tensor_name, tensor_value)
        else:
            self._write_histogram_summary(tensor_name, tensor_value, save_collections)
            # skip writing reductions as TF handles them in the graph itself

            # save raw tensor if reduction config calls for that
            self._write_raw_tensor(tensor_name, tensor_value, save_collections)
            self._write_scalar_summary(tensor_name, tensor_value, save_collections)

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
            self._close_writer()
        self._close_tb_writer()
        self._increment_step()

    @staticmethod
    def _make_numpy_array(tensor_value):
        """
        Convert the tensor value into a numpy array.
        Here it's already numpy array
        """
        return tensor_value

    def end(self, sess):
        pass
