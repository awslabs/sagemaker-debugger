import argparse
import numpy as np
import os

import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D

from tornasole.tensorflow.keras import TornasoleHook
from tornasole import SaveConfig


max_features = 20000
maxlen = 400
embedding_dims = 300
filters = 250
kernel_size = 3
hidden_dims = 250


def parse_args():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--tornasole_outdir', type=str, required=True)
    parser.add_argument('--tornasole_save_interval', type=int, default=10)

    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    return parser.parse_known_args()


def get_train_data(train_dir):

    x_train = np.load(os.path.join(train_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(train_dir, 'y_train.npy'))
    print('x train', x_train.shape,'y train', y_train.shape)
    print( x_train[:10])
    print( y_train[:10])

    return x_train, y_train


def get_test_data(test_dir):

    x_test = np.load(os.path.join(test_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(test_dir, 'y_test.npy'))
    print('x test', x_test.shape,'y test', y_test.shape)

    return x_test, y_test


def get_model():

    embedding_layer = keras.layers.Embedding(max_features,
                                                embedding_dims,
                                                input_length=maxlen)

    sequence_input = keras.Input(shape=(maxlen,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = keras.layers.Dropout(0.2)(embedded_sequences)
    x = keras.layers.Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)(x)
    x = keras.layers.MaxPooling1D()(x)
    x = keras.layers.GlobalMaxPooling1D()(x)
    x = keras.layers.Dense(hidden_dims, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    preds = keras.layers.Dense(1, activation='sigmoid')(x)

    return keras.Model(sequence_input, preds)


if __name__ == "__main__":

    args, _ = parse_args()

    hook = TornasoleHook(out_dir=args.tornasole_outdir,
                         save_config=SaveConfig(save_interval=args.tornasole_save_interval))

    x_train, y_train = get_train_data(args.train)
    x_test, y_test = get_test_data(args.test)

    model = get_model()

    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', 'mean_squared_error'])

    model.fit(x_train, y_train,
              batch_size=args.batch_size,
              epochs=args.epochs,
              validation_data=(x_test, y_test),
              callbacks=[hook]
              )

    model.save(os.path.join(args.model_dir,'sentiment_keras.h5'))
