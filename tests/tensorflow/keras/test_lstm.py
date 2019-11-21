# Standard Library
import collections
import os
import time

# Third Party
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Embedding, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# First Party
from smdebug import SaveConfig
from smdebug.core.collection import CollectionKeys
from smdebug.tensorflow import KerasHook
from smdebug.trials import create_trial


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def build_vocab(filename):
    data = read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def load_data(data_path):
    # get the data paths
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    # build the complete vocabulary, then convert text data to list of integers
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)

    return train_data, valid_data, test_data, vocabulary


class KerasBatchGenerator(object):
    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=5):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
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


def train(num_epochs, batch_size, model, train_data, valid_data, vocabulary, num_steps, hook):
    train_data_generator = KerasBatchGenerator(
        train_data, num_steps, batch_size, vocabulary, skip_step=num_steps
    )
    valid_data_generator = KerasBatchGenerator(
        valid_data, num_steps, batch_size, vocabulary, skip_step=num_steps
    )
    time_callback = TimeHistory()
    callbacks = [time_callback]
    if hook:
        callbacks.append(hook)

    model.fit_generator(
        train_data_generator.generate(),
        len(train_data) // (batch_size * num_steps),
        num_epochs,
        validation_data=valid_data_generator.generate(),
        validation_steps=len(valid_data) // (batch_size * num_steps),
        verbose=0,
        callbacks=callbacks,
    )

    p50 = np.percentile(time_callback.times, 50)
    return p50


def create_tornasole_hook(output_uri, save_more):
    # With the following SaveConfig, we will save tensors for steps 1, 2 and 3
    # (indexing starts with 0).
    save_config = SaveConfig(start_step=1, save_interval=101)

    collections = [CollectionKeys.LOSSES]
    if save_more:
        collections = [CollectionKeys.LOSSES, CollectionKeys.WEIGHTS, CollectionKeys.GRADIENTS]

    # Create a hook that logs weights, biases and gradients while training the model.
    hook = KerasHook(out_dir=output_uri, save_config=save_config, include_collections=collections)
    return hook


def test_lstm(out_dir):

    # init data set
    train_data, valid_data, test_data, vocabulary = load_data(
        "/Users/huilgolr/projects/smdebug-benchmark/data/nlp/"
    )

    # init hook
    hook = None
    hook = create_tornasole_hook(out_dir, True)

    # init model
    num_steps = 100
    hidden_size = 100
    model = Sequential()
    model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(vocabulary)))
    model.add(Activation("softmax"))

    if hook:
        model.compile(
            loss="categorical_crossentropy",
            optimizer=hook.wrap_optimizer(Adam()),
            metrics=["categorical_accuracy"],
        )
    else:
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"]
        )

    # start the training.
    median_time = train(3, 32, model, train_data, valid_data, vocabulary, num_steps, hook)

    # verify output dir
    if hook:
        tr = create_trial(out_dir)
        assert len(tr.tensors(collection=CollectionKeys.LOSSES)) > 0
        assert len(tr.tensors(collection=CollectionKeys.WEIGHTS)) > 0
        assert len(tr.tensors(collection=CollectionKeys.GRADIENTS)) > 0
        print("output directory verified")

    print("Median training time for each epoch is %.1f sec" % median_time)
