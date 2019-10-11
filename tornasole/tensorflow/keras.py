import keras
from .collection import get_collection_manager, CollectionKeys
from tornasole.core.hook import BaseHook
from tornasole.core.save_config import SaveConfig


DEFAULT_INCLUDE_COLLECTIONS=[
    CollectionKeys.WEIGHTS,
    CollectionKeys.GRADIENTS,
    'metrics'
]


class TornasoleHook(keras.callbacks.Callback, BaseHook):
    def __init__(self, out_dir,
                 dry_run=False,
                 reduction_config=None,
                 save_config=SaveConfig(),
                 include_regex=None,
                 include_collections=None,
                 save_all=False):
        if include_regex is not None:
            msg = "'include_regex' is not yet supported and will be ignored."
            self.logger.warning(msg)
        if save_all is not None:
            msg = "'include_regex' is not yet supported and will be ignored."
            self.logger.warning(msg)
        super().__init__(collection_manager=get_collection_manager(),
                         default_include_collections=DEFAULT_INCLUDE_COLLECTIONS,
                         out_dir=out_dir,
                         dry_run=dry_run,
                         reduction_config=reduction_config,
                         save_config=save_config,
                         include_regex=None,
                         include_collections=include_collections,
                         save_all=False)
        self.exported_collections = False

    def _export_collections( self, logs):
        if self.exported_collections :
            return

        for k in logs:
            self.collection_manager.get("metrics").add_tensor_name(k)

        for layer in self.model.layers:
            ws = layer.get_weights()
            if len(ws) == 0:
                continue
            cfg = layer.get_config()
            multi = len(ws) > 1
            for i in range(len(ws)):
                tensor_name = cfg['name']
                if multi:
                    tensor_name += "_" + str(i)
                self.collection_manager.get(CollectionKeys.WEIGHTS).add_tensor_name(tensor_name)

        self.collection_manager.get(CollectionKeys.GRADIENTS).add([])

        # at this point we need all collections to be ready
        # this may not be the case at creation of hook
        # as user's code after hook might add collections
        self._prepare_collections()

        self.export_collections()
        self.exported_collections = True

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.save_metrics(logs=logs, force=True)
        self._close_writer()

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self._export_collections(logs)
        self.save_metrics(logs=logs, force=False)
        self.save_layer_data()
        self._close_writer()
        self._increment_step()

    def save_metrics(self, logs, force):
        for k in logs:
            if self._should_save_tensor_for_step(k) or force:
                val = logs[k]
                self._initialize_writer()
                self.writer.write_tensor(tname=k, tdata=val,
                                         mode=self.mode,
                                         mode_step=self.mode_steps[self.mode])

    def save_layer_data(self):
        assert len(self.model.layers) > 0
        for layer in self.model.layers:
            ws = layer.get_weights()
            if len(ws) == 0:
                continue
            cfg = layer.get_config()

            multi = len(ws) > 1
            for i, tensor_value in enumerate(ws):
                tensor_name = cfg['name']
                if multi:
                    tensor_name += "_" + str(i)
                if self._should_save_tensor_for_step(tensor_name):
                    self._initialize_writer()
                    self.writer.write_tensor(
                            tdata=tensor_value,
                            tname=tensor_name,
                            mode=self.mode,
                            mode_step=self.mode_steps[self.mode])
