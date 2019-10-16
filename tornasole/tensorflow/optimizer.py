import tensorflow as tf
from .collection import *


class TornasoleOptimizer(tf.train.Optimizer):
    def __init__(self, optimizer, use_locking=False, name="Tornasole"):
        super(TornasoleOptimizer, self).__init__(use_locking, name)
        self.optimizer = optimizer
        add_to_collection("optimizer_variables", optimizer.variables())

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        add_to_collection("gradients", [g for (g, v) in grads_and_vars])
        return self.optimizer.apply_gradients(grads_and_vars, global_step, name)
