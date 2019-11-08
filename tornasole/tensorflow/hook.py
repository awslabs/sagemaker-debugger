import os
import tensorflow as tf
from .utils import (
    node_name,
    extract_graph_summary,
    is_parameter_server_strategy,
    get_original_fetch_ops,
    get_num_workers_from_tf_config,
    get_worker_id_from_tf_config,
    TFDistributionStrategy,
)
from .reductions import get_tensorflow_reduction
from .collection import get_collection_manager, Tensor, TensorType
from tornasole.core.tfevent.proto.summary_pb2 import Summary
from tornasole.core.utils import match_inc, serialize_tf_device
from tornasole.core.tfevent.util import make_numpy_array
from tornasole.core.collection import CollectionKeys, SUMMARIES_COLLECTIONS
from tornasole.core.hook import BaseHook
from tornasole.core.writer import FileWriter
from tornasole.core.reductions import get_reduction_tensor_name
from tornasole.core.json_config import (
    TORNASOLE_CONFIG_DEFAULT_WORKER_NAME,
    create_hook_from_json_config,
)
from tornasole.tensorflow.singleton_utils import set_hook
from typing import Optional, List, Union, Tuple, Dict, Set


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
            collection_manager=get_collection_manager(),
            default_include_collections=DEFAULT_INCLUDE_COLLECTIONS,
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
        self.tensor_cache = {}
        """self.device_map is a mapping between a tf device string to a serialized (filename-friendly) device string
        Example -> /job:worker/replica:0/task:1/device:GPU:0 : _job-worker_replica-0_task-1_device-GPU-0"""
        self.device_map = {}
        self.writer_map = {}
        self.distribution_strategy = None
        set_hook(self)

    @staticmethod
    def get_distribution_strategy() -> TFDistributionStrategy:
        try:
            import horovod.tensorflow as hvd

            if hvd.size():
                return TFDistributionStrategy.HOROVOD
        except (ModuleNotFoundError, ValueError, ImportError):
            pass
        tf_config = os.getenv("TF_CONFIG")
        if tf_config and is_parameter_server_strategy(tf_config):
            return TFDistributionStrategy.PARAMETER_SERVER_STRATEGY
        if tf.distribute.get_strategy().num_replicas_in_sync > 1:
            return TFDistributionStrategy.MIRRORED_STRATEGY
        return TFDistributionStrategy.NONE

    def get_worker_name(self) -> str:
        """
        This function returns the name of the worker based on
        the distribution strategy.

        We do not use this function for MirroredStrategy.
        Device names are used as worker names for this MirroredStrategy.
        The names of the workers are managed by device_map in the case of this strategy.

        It is safe to return the TORNASOLE_CONFIG_DEFAULT_WORKER_NAME in this case.
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
        return TORNASOLE_CONFIG_DEFAULT_WORKER_NAME

    def get_num_workers(self):
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

    def export_collections(self):
        self.collection_manager.set_num_workers(self.get_num_workers())
        if self.distribution_strategy == TFDistributionStrategy.MIRRORED_STRATEGY:
            for device, serialized_device in self.device_map.items():
                collection_file_name = f"{serialized_device}_collections.json"
                self.collection_manager.export(self.out_dir, collection_file_name)
        else:
            collection_file_name = f"{self.worker}_collections.json"
            self.collection_manager.export(self.out_dir, collection_file_name)

    def _initialize_writer(self) -> None:
        if self.dry_run:
            return
        if self.distribution_strategy == TFDistributionStrategy.MIRRORED_STRATEGY:
            """
                Initialize one writer per device string
            """
            for device, device_string in self.device_map.items():
                self.writer_map[device_string] = FileWriter(
                    trial_dir=self.out_dir, step=self.step, worker=device_string
                )
            return
        self.writer = FileWriter(trial_dir=self.out_dir, step=self.step, worker=self.worker)

    def _close_writer(self) -> None:
        if self.dry_run:
            return

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

    @classmethod
    def hook_from_config(cls, json_config_path=None):
        return create_hook_from_json_config(
            cls, get_collection_manager(), json_config_path=json_config_path
        )

    def _add_reduction(self, tensor, reduction_name, collection, on_absolute_values=False):
        if tensor.dtype in [tf.bool, tf.string]:
            return
        ts_tensor = self._get_ts_tensor(tensor.name)
        tname = get_reduction_tensor_name(
            ts_tensor.tornasole_name, reduction_name, on_absolute_values, remove_colon_index=False
        )
        red_tensor = get_tensorflow_reduction(reduction_name, tensor, abs=on_absolute_values)
        collection.add_reduction_tensor(
            tensor=red_tensor, original_tensor=tensor, tornasole_name=tname
        )
        # since tensor_to_collections is a mapping from graph name
        if red_tensor.name not in self.tensor_to_collections:
            self.tensor_to_collections[red_tensor.name] = {collection}
        else:
            self.tensor_to_collections[red_tensor.name].add(collection)

    def _add_reductions(self, tensor, collection):
        reduction_config = collection.reduction_config
        for reduction_list in (reduction_config.reductions, reduction_config.norms):
            for reduction in reduction_list:
                self._add_reduction(tensor, reduction, collection, False)
        for reduction_list in (reduction_config.abs_reductions, reduction_config.abs_norms):
            for reduction in reduction_list:
                self._add_reduction(tensor, reduction, collection, True)

    def _merge_ts_tensor_objects_across_collections(self, tensor):
        # merge tensor objects in all collections which has this tensor
        # this ensures that whichever collection you query for this tensorname
        # it returns the same Tornasole Tensor object
        ts_tensor = None
        for coll in self.tensor_to_collections[tensor.name]:
            if ts_tensor is None:
                ts_tensor = coll.get_tensor(tensor.name)
            else:
                coll.set_tensor(ts_tensor)

    def _check_and_add_tensor(self, tensor):
        if tensor.dtype == tf.resource or tensor.dtype == tf.variant:
            return False

        if not self.graph.is_fetchable(tensor.op):
            return False

        colls_with_tensor = set()
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

            if match_inc(tensor.name, coll.include_regex):
                coll.add(tensor)

            if coll.has_tensor(tensor.name):
                ts_tensor = coll.get_tensor(tensor.name)
                ts_tensor.obj_in_graph = tensor
                colls_with_tensor.add(coll)

                if (
                    self.distribution_strategy == TFDistributionStrategy.MIRRORED_STRATEGY
                    and len(tensor.device)
                    and "CPU" not in tensor.device
                    and tensor.device not in self.device_map
                ):
                    self.device_map[tensor.device] = serialize_tf_device(tensor.device)

        if colls_with_tensor:
            # this is mapping from graph name
            self.tensor_to_collections[tensor.name] = colls_with_tensor
            self._merge_ts_tensor_objects_across_collections(tensor)
            for coll in colls_with_tensor:
                # this should be after we add tensor.name to tensor_to_collections
                self._add_reductions(tensor, coll)

    def _add_tensors(self):
        # so collections have save configs and reduction configs
        self._prepare_collections()

        # todo: do we ever need inputs of the op
        for op in self.graph.get_operations():
            for tensor in op.outputs:
                self._check_and_add_tensor(tensor)

    def _add_summaries_tensors(self):
        if CollectionKeys.TENSORFLOW_SUMMARIES in self.include_collections:
            c = self.collection_manager.get(CollectionKeys.TENSORFLOW_SUMMARIES).add(
                tf.get_collection(tf.GraphKeys.SUMMARIES)
            )
            for t in c.get_tensors():
                t.type = TensorType.SUMMARY
                t.original_tensor = t.op.inputs[1]

    def begin(self):
        # todo: should this be called first time a mode changes
        # todo: handle multiple graphs in the model
        self.distribution_strategy = TornasoleHook.get_distribution_strategy()
        self.worker = self.get_worker_name()
        self.graph = tf.get_default_graph()

        wts = tf.trainable_variables()
        self.collection_manager.get(CollectionKeys.WEIGHTS).add(wts)

        losses = tf.losses.get_losses()
        self.collection_manager.get(CollectionKeys.LOSSES).add(losses)

        self._add_summaries_tensors()
        self._add_tensors()

        self._export_model()
        self.export_collections()

    def _export_model(self):
        self.logger.info("Writing graph")
        tb_writer = self._maybe_get_tb_writer()
        if tb_writer:
            tb_writer.write_graph(self.graph.as_graph_def(add_shapes=True))
        else:
            self.logger.debug("Graph not exported because `hook.tensorboard_dir` is None")
        # don't close writer as it might be needed in the step that follows
        # else we will have to open the file again

    def _get_tensors_to_save_this_step(self):
        tensors_to_save = set()
        for coll in self._get_collections_to_save_for_step():
            for ts_tensor in coll.get_tensors():
                tensors_to_save.add(ts_tensor.obj_in_graph)
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
        for tensor in tensors_to_save:
            ts_tensor = self._get_ts_tensor(tensor.name)
            if ts_tensor.type == TensorType.REGULAR:
                if node_name(tensor.name) in subgraph_nodes:
                    filtered.append(tensor)
            else:
                if node_name(ts_tensor.original_tensor.name) in subgraph_nodes:
                    filtered.append(tensor)
                else:
                    skipped.append(tensor)
        if len(skipped) > 0:
            self.logger.debug(f"Skipped {len(skipped)} unreachable tensors: {skipped}")

        # todo(huilgolr) can we filter tensors with (0) size here. do we want to?
        return filtered

    def _get_collections_with_tensor(self, tensor_name) -> Set["Collection"]:
        self._assert_prep()
        return self.tensor_to_collections[tensor_name]

    def _get_ts_tensor(self, tf_tensor_name, save_collections=None):
        if save_collections is None:
            save_collections = self._get_collections_with_tensor(tf_tensor_name)
        if save_collections:
            return next(iter(save_collections)).get_tensor(tf_tensor_name)
        else:
            raise RuntimeError(
                f"Hook attempted to save unknown tensor {tf_tensor_name}."
                f"This does not belong to any collection"
            )

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

    def _write_for_tensor(self, tensor_name, tensor_value, save_collections):
        # this tensor_name is tf tensor name, need to convert to tornasole_name
        ts_tensor = self._get_ts_tensor(tensor_name, save_collections=save_collections)
        name = ts_tensor.tornasole_name
        self.logger.debug(f"Saving {name} for global step {self.step}")
        if ts_tensor.type == TensorType.REDUCTION:
            self._write_raw_tensor_simple(name, tensor_value)
        else:
            self._write_histogram_summary(name, tensor_value, save_collections)

            # skip writing reductions as TF handles them in the graph itself

            # save raw tensor if reduction config calls for that
            self._write_raw_tensor(name, tensor_value, save_collections)
            self._write_scalar_summary(name, tensor_value, save_collections)

    def _get_all_tensors_values(self, results):
        for (item, value) in zip(self.tensors_to_save_this_step, results):
            if not isinstance(value, list) or isinstance(value, tuple):
                assert not (isinstance(item, list) or isinstance(item, tuple))
                yield item, value
            elif isinstance(value, list) or isinstance(value, tuple):
                assert isinstance(item, list) or isinstance(item, tuple)
                for i in range(len(value)):
                    yield item[i], value[i]

    def get_writers(self, tensor_name) -> List[FileWriter]:
        """
        For tensors generated during distributed tf jobs, we map the tensor to a writer
        with its device attribute.
        If the device attribute is CPU, we map it to all the writers.
        For all other frameworks and single worker jobs we return a list with a single worker.
        :param tensor_name:
        :return: List[FileWriter]
        """
        if self.distribution_strategy == TFDistributionStrategy.MIRRORED_STRATEGY:
            worker = self.tensor_cache.get(tensor_name).device
            if not bool(worker) or "CPU" in worker:
                return list(self.writer_map.values())
            worker = self.device_map[worker]
            return [self.writer_map[worker]]
        return [self.writer]

    def after_run(self, run_context, run_values):
        if self.tensors_to_save_this_step:
            self._initialize_writer()
            for (tensor, value) in self._get_all_tensors_values(run_values.results):
                if tensor.dtype == tf.string:
                    self._write_tf_summary(tensor, value)
                else:
                    # todo: need to use ts_tensor for this and remove tensor_cache
                    self.tensor_cache[tensor.name] = tensor
                    self._save_for_tensor(tensor.name, value, check_before_write=False)
                    self.tensor_cache.clear()  # cleanup to remove memory footprint
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
