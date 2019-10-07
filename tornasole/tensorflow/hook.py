import numpy as np

from .utils import *
from .reductions import get_tensorflow_reduction
from .collection import *

from tornasole.core.hook import BaseHook
from tornasole.core.utils import match_inc
from tornasole.core.reductions import get_reduction_tensor_name
from tornasole.core.json_config import TORNASOLE_CONFIG_DEFAULT_WORKER_NAME, create_hook_from_json_config

DEFAULT_INCLUDE_COLLECTIONS = [
    CollectionKeys.WEIGHTS,
    CollectionKeys.GRADIENTS,
    CollectionKeys.DEFAULT,
    CollectionKeys.LOSSES
]


class TornasoleHook(tf.train.SessionRunHook, BaseHook):
    def __init__(self, out_dir=None,
                 dry_run=False,
                 reduction_config=None,
                 save_config=None,
                 include_regex=None,
                 include_collections=None,
                 save_all=False):
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
        super().__init__(collection_manager=get_collection_manager(),
                         default_include_collections=DEFAULT_INCLUDE_COLLECTIONS,
                         out_dir=out_dir,
                         dry_run=dry_run,
                         reduction_config=reduction_config,
                         save_config=save_config,
                         include_regex=include_regex,
                         include_collections=include_collections,
                         save_all=save_all)
        self.reduction_original_tensors = {}
        self.subgraph_nodes_cache = {}

    def get_worker_name(self):
        try:
            import horovod.tensorflow as hvd
            if hvd.size():
                return f'worker_{hvd.rank()}'
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
    def hook_from_config(cls):
        return create_hook_from_json_config(cls, get_collection_manager())


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
        reduction_config = collection.get_reduction_config()
        if reduction_config:
            for reduction in reduction_config.reductions + reduction_config.norms:
                self._add_reduction(tensor, reduction, collection, False)
            for reduction in reduction_config.abs_reductions + reduction_config.abs_norms:
                self._add_reduction(tensor, reduction, collection, True)

            # here if reduction config was set,
            # but tensors were added to collection,
            # they will be removed and added to reduction_tensors
            try:
                collection.remove_tensor(tensor)
            except IndexError:
                # was not in the list
                pass
            # so this is available in this collection for reader
            # hook will read from tensors and reduction_tensors_added lists
            collection.add_tensor_name(tensor.name)
        else:
            collection.add(tensor)

    def _check_and_add_tensor(self, t):
        if t.dtype == tf.resource or t.dtype == tf.variant:
            return False

        if not self.graph.is_fetchable(t.op):
            return False

        added = False
        for coll in self._get_all_collections_to_save():
            if match_inc(t.name, coll.get_include_regex()) \
                    or t.name in coll.tensor_names:
                self._process_matched_tensor(t, coll)
                # only matches with one collection
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
        # gradients and optimizer_variables added in user code or TornasoleOptimizer

        total_tensor_count = 0
        # todo: do we ever need inputs of the op
        for op in self.graph.get_operations():
            for tensor in op.outputs:
                self._check_and_add_tensor(tensor)
                total_tensor_count += 1
        # all variables we are interested in are part of the graph tensors
        # for variable in tf.global_variables():
        #     self._check_and_add_tensor(variable)
        #     total_tensor_count += 1
        return total_tensor_count

    def begin(self):
        # todo: handle multiple graphs in the model
        self.graph = tf.get_default_graph()

        for coll_name, coll in self.collection_manager.get_collections().items():
            # hack to make multiple graphs work with the same tensor names
            # this can happen when we use same hook for training and evaluation
            # what is going on here is that we clear the tensors and reduction tensors
            # but we use the tensor names field in collection to readd tensors
            # from the new graph to the collection so we can them right
            coll.tensors = []
            coll.reduction_tensors = []

        wts = tf.trainable_variables()
        self.collection_manager.get(CollectionKeys.WEIGHTS).add(wts)

        losses = tf.losses.get_losses()
        self.collection_manager.get(CollectionKeys.LOSSES).add(losses)

        # at this point we need all collections to be ready
        # this may not be the case at creation of hook
        # as user's code after hook might add collections
        self._prepare_collections()

        # adds all tensors in graph based on regexes in collections default and other custom ones
        self._add_tensors()
        self._prepare_tensors()

        for coll in self._get_all_collections_to_save():
            self.logger.info(f'Saving the collection {coll.name} with {len(coll.tensor_names)} tensors ' \
                             f'and {len(coll.reduction_tensors_added)} reductions')
            self.logger.debug(f'  Collection {coll.name} has tensors: {coll.tensors}')
            self.logger.debug(f'  Collection {coll.name} has reductions: {coll.reduction_tensors_added}')

        self.export_collections()
        self._export_model()

    def _export_model(self):
        # todo save model
        pass

    def _get_tensors_to_save_this_step(self):
        colls_to_save = self._get_collections_to_save_for_step(
                self.mode, self.mode_steps[self.mode])
        tensors_to_save = {'watched': [], 'added': []}
        for coll in colls_to_save:
            tensors_to_save['watched'].extend(coll.tensors)
            tensors_to_save['added'].extend(coll.reduction_tensors_added)
        # dedup watched and added
        tensors_to_save['watched'] = list(set(tensors_to_save['watched']))
        tensors_to_save['added'] = list(set(tensors_to_save['added']))
        return tensors_to_save

    def _filter_to_be_saved(self, dict_to_save, fetches):
        if not isinstance(fetches, list) and not isinstance(fetches, tuple) \
                and not isinstance(fetches, dict):
            fetches = [fetches]
        fetches_tuple = tuple(fetches)
        if fetches_tuple in self.subgraph_nodes_cache:
            subgraph_nodes = self.subgraph_nodes_cache[fetches_tuple]
        else:
            original_fetch_ops = get_original_fetch_ops(fetches)
            dest_names = [n.name for n in original_fetch_ops]
            subgraph = tf.graph_util.extract_sub_graph(
                tf.get_default_graph().as_graph_def(), dest_names)
            _, subgraph_nodes, _ = extract_graph_summary(subgraph)
            self.subgraph_nodes_cache[fetches_tuple] = subgraph_nodes

        # this also allows us to skip all the assign, read, initial_value,
        # control_dependency nodes in the graph
        # check that this run includes the ops whose tensors are to be saved
        filtered = []
        skipped = []
        for tensor in dict_to_save['watched']:
            if node_name(tensor.name) in subgraph_nodes:
                filtered.append(tensor)
            else:
                skipped.append(tensor)
        for tensor in dict_to_save['added']:
            assert isinstance(tensor, tf.Tensor)
            original_tensor = self.reduction_original_tensors[tensor.name]
            if node_name(original_tensor.name) in subgraph_nodes:
                filtered.append(tensor)
            else:
                skipped.append(tensor)
        self.logger.debug(f'Skipped {len(skipped)} unreachable tensors: {skipped}')

        # todo(huilgolr) can we filter tensors with (0)shape here
        return filtered

    def before_run(self, run_context):
        tensors_to_save = self._get_tensors_to_save_this_step()
        if len(tensors_to_save['watched']) + len(tensors_to_save['added']) > 0:
            if run_context:
                list_to_save = self._filter_to_be_saved(
                        tensors_to_save, run_context.original_args.fetches)
            else:
                list_to_save = tensors_to_save['watched'] + \
                               tensors_to_save['added']
        else:
            list_to_save = []
        self.prev_to_be_saved = list_to_save
        return tf.train.SessionRunArgs(list_to_save) if list_to_save else None

    def _save_tensor(self, tensor, value):
        if tensor.dtype == np.float16:
            # todo: save as fp16 itself.
            #  measure perf as proto doesn't handle that well
            value = np.float32(value)
        size_saved = value.nbytes
        this_size, this_shape = size_and_shape(value)
        if this_size > 0:
            self.logger.debug(f'    Saving {tensor.name}, type={tensor.dtype}, shape={this_shape},' +
                            f'size={this_size}')
            if not self.dry_run:
                self.writer.write_tensor(tdata=value, tname=tensor.name,
                                         mode=self.mode,
                                         mode_step=self.mode_steps[self.mode])
        else:
            self.logger.debug(f'    Not saving {tensor.name}, type={tensor.dtype}, shape={this_shape},' +
                              f'size={this_size}')
        return size_saved

    def _get_all_tensors_values(self, results):
        for (item, value) in zip(self.prev_to_be_saved, results):
            if not isinstance(value, list) or isinstance(value, tuple):
                assert not (isinstance(item, list) or isinstance(item, tuple))
                yield item, value
            elif isinstance(value, list) or isinstance(value, tuple):
                assert (isinstance(item, list) or isinstance(item, tuple))
                for i in range(len(value)):
                    yield item[i], value[i]

    def after_run(self, run_context, run_values):
        if self.prev_to_be_saved:
            self._initialize_writer()
            running_size = 0
            for (item, value) in self._get_all_tensors_values(run_values.results):
                running_size += self._save_tensor(item, value)
            self.logger.info(f'Saved {running_size} bytes for '
                             f'{len(self.prev_to_be_saved)} objects '
                             f'at step {self.step}')
            self._flush_and_close_writer()
        self._increment_step()

    def end(self, sess):
        pass
