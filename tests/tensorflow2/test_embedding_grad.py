import numpy as np
import tensorflow as tf
from packaging import version

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from tensorflow.python.eager import backprop
from tensorflow.python.util import nest
from tensorflow.python.keras.engine import data_adapter

if version.parse(tf.__version__) >= version.parse("2.11.0") or "rc" in tf.__version__:
    from tensorflow.keras.optimizers.legacy import Adam
else:
    from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Concatenate, Dense
from sklearn.preprocessing import LabelEncoder

import smdebug.tensorflow as smd


class ModelSubClassing(tf.keras.Model):
    def __init__(self, hook=None):
        super(ModelSubClassing, self).__init__()
        self.embedding_encoded_layer = tf.keras.layers.Embedding(
            input_dim=20,
            output_dim=10,
        )
        self.dense1 = Dense(10, input_shape=(10,), activation='relu')
        self.dense = Dense(1, activation='sigmoid')
        self.hook = hook

    def call(self, x, training=False):
        x = self.embedding_encoded_layer(x)
        x = self.dense1(x)
        return self.dense(x)

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with self.hook.wrap_tape(backprop.GradientTape()) as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics


def train(out_dir):
    hook = smd.KerasHook(out_dir=out_dir, save_all=["true"],
                         include_collections=["weights", "biases", "default", "gradients"],
                         save_config=smd.SaveConfig(save_interval=5), )

    x_train = np.random.randint(1, 20, size=(1000, 20))
    y_train = np.random.choice([0, 1], size=(1000, 1), p=[1. / 3, 2. / 3])
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    sub_classing_model = ModelSubClassing(hook)

    hook.register_model(sub_classing_model)

    optimizer = Adam()
    optimizer = hook.wrap_optimizer(optimizer)

    sub_classing_model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(),
                               run_eagerly=True)

    sub_classing_model.fit(x_train, encoded_Y, batch_size=128, epochs=1, callbacks=[hook])
    print(sub_classing_model.summary())


def test_embedding_grad(out_dir):
    train(out_dir)
    trial = smd.create_trial(path=out_dir)
    assert len(trial.tensor_names(collection="gradients")) == 5
