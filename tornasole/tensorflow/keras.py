import keras
import os
import socket

from .collection import *
from tornasole.core.writer import FileWriter
from tornasole.core.utils import flatten
from tornasole.core.logger import get_logger
from tornasole.core.hook_utils import verify_and_get_out_dir
from tornasole.core.modes import ModeKeys
from tornasole.core.save_config import SaveConfig
from tornasole.core.save_manager import SaveManager


class TornasoleHook(keras.callbacks.Callback):
    def __init__(self, out_dir,
                 dry_run=False,
                 worker='worker0',
                 reduction_config=None,
                 save_config=SaveConfig(),
                 # TODO: support include_regex
                 # include_regex=None,
                 include_collections=['weights', 'gradients', 'metrics', 'default'],
                 save_all=False):
        self.out_dir = verify_and_get_out_dir(out_dir)

        self.dry_run = dry_run
        self.worker = worker if worker is not None else socket.gethostname()
        if include_collections is None:
            include_collections = []
        self.include_collections = flatten(include_collections)
        # TODO: support include_regex
        # if include_regex is not None:
        #    get_collection('default').include(include_regex)
        #    if 'default' not in self.include_collections:
        #        self.include_collections.append('default')

        self.save_all = save_all
        if self.save_all:
            get_collection('all').include('.*')
            if 'all' not in self.include_collections:
                self.include_collections.append('all')

        self.logger = get_logger()
        if 'default' not in self.include_collections and get_collection('default').get_include_regex():
            self.logger.warn('The `default` collection was not passed to include_collections.' \
                             'So it is not being saved')

        self.save_manager = SaveManager(collection_manager=get_collection_manager(),
                                        include_collections_names=self.include_collections,
                                        default_save_config=save_config,
                                        default_reduction_config=reduction_config)

        self.step = 0
        self.mode = ModeKeys.GLOBAL
        self.mode_steps = {ModeKeys.GLOBAL: 0}
        self.writer = None
        self.logger.info('Saving to {}'.format(self.out_dir))
        self._collection_created = False

        super().__init__()

    def _export_collections( self, logs):
        if self._collection_created:
            return

        for k in logs:
            get_collection("metrics").add_tensor_name(k)

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
                get_collection("weights").add_tensor_name(tensor_name)

        add_to_collection("gradients", [])

        export_collections(os.path.join(self.out_dir, 'collections.ts'))
        # at this point we need all collections to be ready
        # this may not be the case at creation of hook
        # as user's code after hook might add collections
        self.save_manager.prepare()
        self._collection_created = True

    def on_epoch_end(self, epoch, logs={}):
        self.save_metrics(logs=logs, force=True)
        self._delete_writer()

    def on_batch_end(self, batch, logs={}):
        self._export_collections(logs)
        self.save_metrics(logs=logs, force=False)
        self.save_layer_data()
        self._delete_writer()
        self.step += 1
        self.mode_steps[self.mode] += 1
        #print( "Writer=", self.writer)


    def _create_writer(self):
        if self.writer is None:
            self.writer = FileWriter(trial_dir=self.out_dir,
                                     step=self.step,
                                     worker=self.worker)
        return self.writer

    def _delete_writer(self):
        if self.writer:
            self.writer.close()
            self.writer = None

    def save_metrics(self, logs, force):
        for k in logs:
            save_state = self.save_manager.should_save_tensor(k, self.mode,
                                                              self.mode_steps[self.mode])
            if save_state['step'] or force:
                val = logs[k]
                self._create_writer()
                self.writer.write_tensor(tname=k, tdata=val)

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
                save_state = self.save_manager.should_save_tensor(tensor_name, self.mode,
                                                        self.mode_steps[self.mode])
                if save_state['step']:
                    self._create_writer()
                    self.writer.write_tensor(tdata=tensor_value, tname=tensor_name)
