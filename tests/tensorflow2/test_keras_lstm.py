# Standard Library

# Third Party
import numpy as np
import pytest
import tensorflow.compat.v2 as tf
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Embedding, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# First Party
from smdebug import SaveConfig
from smdebug.core.collection import CollectionKeys
from smdebug.tensorflow import KerasHook
from smdebug.trials import create_trial


class KerasBatchGenerator(object):
    def __init__(self, num_steps, batch_size, skip_step=5):
        self.data = np.random.randint(low=0, high=1000, size=10000).tolist()
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = 1000
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx : self.current_idx + self.num_steps]
                temp_y = self.data[self.current_idx + 1 : self.current_idx + self.num_steps + 1]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
                self.current_idx += self.skip_step
            yield x, y


def train(num_epochs, batch_size, model, num_steps, hook):
    train_data_generator = KerasBatchGenerator(num_steps, batch_size, skip_step=num_steps)
    valid_data_generator = KerasBatchGenerator(num_steps, batch_size, skip_step=num_steps)
    callbacks = []
    if hook:
        callbacks.append(hook)

    model.fit_generator(
        train_data_generator.generate(),
        len(train_data_generator.data) // (batch_size * num_steps),
        num_epochs,
        validation_data=valid_data_generator.generate(),
        validation_steps=len(valid_data_generator.data) // (batch_size * num_steps),
        verbose=0,
        callbacks=callbacks,
    )


@pytest.mark.slow
def test_lstm_and_generator(out_dir, tf_eager_mode):
    # init hook
    hook = KerasHook(
        out_dir,
        include_collections=[
            CollectionKeys.WEIGHTS,
            CollectionKeys.LOSSES,
            CollectionKeys.GRADIENTS,
        ],
        save_config=SaveConfig(save_steps=[0, 1, 2, 3]),
    )

    if not tf_eager_mode:
        tf.compat.v1.disable_eager_execution()

    # init model
    num_steps = 100
    hidden_size = 100
    vocabulary = 1000
    model = Sequential()
    model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(vocabulary)))
    model.add(Activation("softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=hook.wrap_optimizer(Adam()),
        metrics=["categorical_accuracy"],
    )

    train(3, 32, model, num_steps, hook)

    tr = create_trial(out_dir)
    assert len(tr.tensor_names(collection=CollectionKeys.LOSSES)) > 0
    assert len(tr.tensor_names(collection=CollectionKeys.WEIGHTS)) > 0
    # can't get gradients with TF 2.x yet
    # assert len(tr.tensor_names(collection=CollectionKeys.GRADIENTS)) > 0
