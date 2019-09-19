import os
import socket
import atexit
import numpy as np
from .utils import *
from .reductions import get_tensorflow_reduction
from .collection import *
from tornasole.core.writer import FileWriter
from tornasole.core.utils import flatten, match_inc
from tornasole.core.logger import get_logger
from tornasole.core.hook_utils import verify_and_get_out_dir
from tornasole.core.reductions import get_reduction_tensor_name
from tornasole.core.json_config import TORNASOLE_CONFIG_DEFAULT_WORKER_NAME, create_hook_from_json_config
from tornasole.core.modes import ModeKeys, ALLOWED_MODES
from tornasole.core.save_config import SaveConfig
from tornasole.core.access_layer.utils import training_has_ended
from .save_manager import TFSaveManager

DEFAULT_INCLUDE_COLLECTIONS = ['weights', 'gradients', 'default', 'losses']

class TornasoleHook(tf.train.SessionRunHook):
    def __init__(self, out_dir=None,
                 dry_run=False,
                 worker=TORNASOLE_CONFIG_DEFAULT_WORKER_NAME,
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
        self.out_dir = verify_and_get_out_dir(out_dir)

        self.dry_run = dry_run
        self.worker = worker if worker is not None else socket.gethostname()
        if include_collections is None:
            include_collections = DEFAULT_INCLUDE_COLLECTIONS
        self.include_collections = flatten(include_collections)
        if include_regex is not None:
            get_collection('default').include(include_regex)
            if 'default' not in self.include_collections:
                self.include_collections.append('default')

        self.save_all = save_all
        if self.save_all:
            get_collection('all').include('.*')
            if 'all' not in self.include_collections:
                self.include_collections.append('all')

        if 'default' not in self.include_collections and get_collection('default').get_include_regex():
            self.logger.warn('The `default` collection was not passed to include_collections.' \
                             'So it is not being saved')
        if save_config is None:
            save_config = SaveConfig()

        self.save_manager = TFSaveManager(collection_manager=get_collection_manager(),
                                        include_collections_names=self.include_collections,
                                        default_save_config=save_config,
                                        default_reduction_config=reduction_config)

        self.step = 0
        self.mode = ModeKeys.GLOBAL
        self.mode_steps = {ModeKeys.GLOBAL: 0}
        self.logger = get_logger()
        self.writer = None
        self.reduction_original_tensors = {}
        self.subgraph_nodes_cache = {}
        self.logger.info('Saving to {}'.format(self.out_dir))
        atexit.register(self.cleanup)
    
    @classmethod
    def hook_from_config(cls):
        return create_hook_from_json_config(cls, get_collection_manager(), DEFAULT_INCLUDE_COLLECTIONS)

    def cleanup(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        #creates file  "trial_prefix/END_OF_JOB.ts" at the end of training job.
        # Trial prefix can be s3/local.
        training_has_ended(self.out_dir)

    def set_mode(self, mode):
        # train
        if mode in ALLOWED_MODES:
            self.mode = mode
        else:
            raise ValueError('Invalid mode {}. Valid modes are {}.'
                             .format(mode, ','.join(ALLOWED_MODES)))

        if mode not in self.mode_steps:
            self.mode_steps[mode] = 0

    def _process_matched_tensor(self, tensor, collection):
        reduction_config = self.save_manager.get_reduction_config(collection)
        # if reduction config and saveconfig.when_nan are set, the when_nan tensors will be reduced
        # todo think about this
        if reduction_config:
            for reduction in reduction_config.reductions + reduction_config.norms:
                self._add_reduction(tensor, reduction, collection, False)
            for reduction in reduction_config.abs_reductions + reduction_config.abs_norms:
                self._add_reduction(tensor, reduction, collection, True)
            # here if reduction config was set, but tensors were added to collection,
            # they will be removed and added to reduction_tensors
            try:
                collection.remove_tensor(tensor)
            except IndexError:
                # was not in the list
                pass
        else:
            collection.add(tensor)

    def _check_and_add_tensor(self, t):
        if t.dtype == tf.resource or t.dtype == tf.variant:
            return False

        if not self.graph.is_fetchable(t.op):
            return False

        added = False
        for coll in self.save_manager.get_all_collections_to_save():
            if match_inc(t.name, coll.get_include_regex()) \
                    or t.name in coll.tensor_names:
                    # or t.name in coll.reduction_tensor_names:
                self._process_matched_tensor(t, coll)
                # only matches with one collection
                added = True
            sc = self.save_manager.get_save_config(coll, self.mode)
            if sc and match_inc(t.name, sc.when_nan):
                # add when_nan tensors to watched, so they are returned
                # matches for all collections
                # self._process_matched_tensor(t, coll)
                self.save_manager.add_when_nan_tensor(coll, t)
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
        for variable in tf.global_variables():
            self._check_and_add_tensor(variable)
            total_tensor_count += 1
        return total_tensor_count

    def begin(self):
        # todo: handle multiple graphs in the model
        self.graph = tf.get_default_graph()

        for coll_name, coll in get_collections().items():
            # hack to make multiple graphs work with the same tensor names
            # this can happen when we use same hook for training and evaluation
            # what is going on here is that we clear the tensors and reduction tensors
            # but we use the tensor names field in collection to readd tensors
            # from the new graph to the collection so we can them right
            coll.tensors = []
            coll.reduction_tensors = []

        wts = tf.trainable_variables()
        add_to_collection('weights', wts)

        losses = tf.losses.get_losses()
        add_to_collection('losses', losses)

        # todo: fix this coll.save_config.when_nan_tensors = []

        # at this point we need all collections to be ready
        # this may not be the case at creation of hook
        # as user's code after hook might add collections
        self.save_manager.prepare()

        # adds all tensors in graph based on regexes in collections default and other custom ones
        self._add_tensors()
        self.save_manager.prepare_tensors()

        for coll in self.save_manager.get_all_collections_to_save():
            self.logger.info(f'Saving the collection {coll.name} with {len(coll.tensor_names)} tensors ' \
                                 f'and {len(coll.reduction_tensors_added)} reductions for {len(coll.reduction_tensor_names)} tensors.')
            self.logger.debug(f'  Collection {coll.name} has tensors: {coll.tensors}')
            self.logger.debug(f'  Collection {coll.name} has reductions: {coll.reduction_tensors_added}')

        export_collections(os.path.join(self.out_dir, 'collections.ts'))
        self._export_model()

    def _export_model(self):
        # todo save model
        pass

    def _save_this_step(self):
        coll_save_state = self.save_manager.collections_to_save(self.mode, self.mode_steps[self.mode])
        tensors_to_save = {'watched': [], 'added': []}
        for coll_name, save_state in coll_save_state.items():
            coll = get_collection(coll_name)
            if save_state['step'] or save_state['when_nan']:
                tensors_to_save['watched'].extend(coll.tensors)
                tensors_to_save['added'].extend(coll.reduction_tensors_added)
            if save_state['when_nan']:
                tensors_to_save['watched'].extend(
                    self.save_manager.get_save_config(coll, self.mode).when_nan_tensors)
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
            subgraph = tf.compat.v1.graph_util.extract_sub_graph(
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
        tensors_to_save = self._save_this_step()
        if len(tensors_to_save['watched'] + tensors_to_save['added']):
            if run_context:
                list_to_save = self._filter_to_be_saved(tensors_to_save,
                                                        run_context.original_args.fetches)
            else:
                list_to_save = tensors_to_save['watched'] + tensors_to_save['added']
        else:
            list_to_save = []
            # self.logger.info('Skipping step %s' % str(self.step))

        self.prev_to_be_saved = list_to_save
        return tf.train.SessionRunArgs(list_to_save) if list_to_save else None

    def _save_tensor(self, tensor, value, running_size):
        running_size += value.nbytes
        if tensor.dtype == np.float16:
            value = np.float32(value)
            running_size += value.nbytes
        this_size, this_shape = size_and_shape(value)
        if this_size > 0:
            self.logger.debug(f'    Saving {tensor.name}, type={tensor.dtype}, shape={this_shape},' +
                            f'size={this_size}, running_size={running_size}')
            if not self.dry_run:
                self.writer.write_tensor(tdata=value, tname=tensor.name,
                                         mode=self.mode,
                                         mode_step=self.mode_steps[self.mode])
        else:
            self.logger.debug(f'    Not saving {tensor.name}, type={tensor.dtype}, shape={this_shape},' +
                              f'size={this_size}, running_size={running_size}')
        return running_size

    def _check_when_nan_tensors(self, values):
        tensors = self.prev_to_be_saved
        is_nan_for_colls = set()
        assert len(tensors) == len(values)
        for i in range(len(tensors)):
            tensor = tensors[i]
            value = values[i]
            if self.save_manager.is_when_nan_tensor(tensor.name):
                is_nan = np.isnan(np.sum(value)) or np.isinf(np.sum(value))
                if is_nan:
                    is_nan_for_colls.update([x.name for x in self.save_manager.when_nan_collections(tensor.name)])
                if len(is_nan_for_colls) == len(self.save_manager.get_all_collections_to_save()):
                    # all collections are nan already, don't check other tensors
                    break
        return is_nan_for_colls

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
            self.writer = FileWriter(trial_dir=self.out_dir,
                                     step=self.step,
                                     worker=self.worker)
            running_size = 0
            is_nan_for_collections = self._check_when_nan_tensors(run_values.results)
            for (item, value) in self._get_all_tensors_values(run_values.results):
                save_state = self.save_manager.should_save_tensor(item.name, self.mode,
                                                                  self.mode_steps[self.mode])
                from_colls = set([x.name for x in self.save_manager.from_collections(item.name)])
                if save_state['step'] or \
                    (save_state['when_nan'] and from_colls.intersection(is_nan_for_collections)):
                    running_size = self._save_tensor(item, value, running_size)
                else:
                    self.logger.debug(f'Not saving {item} as no nan seen')
            self.logger.info(f'Saved {running_size} bytes for {len(self.prev_to_be_saved)} objects at step {self.step}')
            self.writer.close()
            self.writer = None
        self.step += 1
        self.mode_steps[self.mode] += 1

    def end(self, sess):
        pass
        # self.logger.info('End of run')
