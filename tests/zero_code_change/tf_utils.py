# Standard Library
import typing as Tuple

# Third Party
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
from tensorflow.python.client import device_lib

tfds.disable_progress_bar()


class LarcOptimizer(tf.train.Optimizer):
    """ LARC implementation
        -------------------
        Parameters:
          - optimizer:     initial optimizer that you wanna apply
                           example: tf.train.MomentumOptimizer
          - learning_rate: initial learning_rate from initial optimizer
          - clip:          if True apply LARC otherwise LARS
          - epsilon:       default value is weights or grads are 0.
          - name
          - use_locking
    """

    def __init__(
        self,
        optimizer,
        learning_rate,
        eta,
        clip=True,
        epsilon=1.0,
        name="LarcOptimizer",
        use_locking=False,
    ):
        super(LarcOptimizer, self).__init__(name=name, use_locking=use_locking)
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._eta = float(eta)
        self._clip = clip
        self._epsilon = float(epsilon)

    def compute_gradients(self, *args, **kwargs):
        return self._optimizer.compute_gradients(*args, **kwargs)

    def apply_gradients(self, gradvars, *args, **kwargs):
        v_list = [tf.norm(tensor=v, ord=2) for _, v in gradvars]
        g_list = [tf.norm(tensor=g, ord=2) if g is not None else 0.0 for g, _ in gradvars]
        v_norms = tf.stack(v_list)
        g_norms = tf.stack(g_list)
        zeds = tf.zeros_like(v_norms)
        # assign epsilon if weights or grads = 0, to avoid division by zero
        # also prevent biases to get stuck at initialization (0.)
        cond = tf.logical_and(tf.not_equal(v_norms, zeds), tf.not_equal(g_norms, zeds))
        true_vals = tf.scalar_mul(self._eta, tf.div(v_norms, g_norms))
        false_vals = tf.fill(tf.shape(v_norms), self._epsilon)
        larc_local_lr = tf.where(cond, true_vals, false_vals)
        if self._clip:
            ones = tf.ones_like(v_norms)
            lr = tf.fill(tf.shape(v_norms), self._learning_rate)
            # We need gradients to compute local learning rate,
            # so compute_gradients from initial optimizer have to called
            # for which learning rate is already fixed
            # We then have to scale the gradients instead of the learning rate.
            larc_local_lr = tf.minimum(tf.div(larc_local_lr, lr), ones)
        gradvars = [
            (tf.multiply(larc_local_lr[i], g), v) if g is not None else (None, v)
            for i, (g, v) in enumerate(gradvars)
        ]
        return self._optimizer.apply_gradients(gradvars, *args, **kwargs)


def get_estimator(nested_optimizer=False, mirrored=False) -> tf.estimator.Estimator:
    """ Return an estimator object ready for testing. """
    if mirrored:
        distribution = tf.distribute.MirroredStrategy()
        config = tf.estimator.RunConfig(train_distribute=distribution, model_dir="/tmp/mnist_model")
    else:
        config = None

    return tf.estimator.Estimator(
        model_fn=_cnn_model_fn,
        model_dir="/tmp/mnist_model",
        config=config,
        params={"nested_optimizer": nested_optimizer},
    )


def get_input_fns(batch_size=32) -> Tuple:
    # Load training and eval data
    ((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

    train_data = train_data / np.float32(255)
    train_labels = train_labels.astype(np.int32)  # not required

    eval_data = eval_data / np.float32(255)
    eval_labels = eval_labels.astype(np.int32)  # not required

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data}, y=train_labels, batch_size=batch_size, num_epochs=5, shuffle=True
    )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False
    )

    return train_input_fn, eval_input_fn


def _cnn_model_fn(features, labels, mode, params):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu
    )

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the`logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    )
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    tf.summary.scalar("loss", loss)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        if params["nested_optimizer"]:
            optimizer = LarcOptimizer(optimizer, 0.01, 0.0005)

        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


### Used for tf.Session ###


def get_train_op_and_placeholders():
    # Parameters
    learning_rate = 0.1
    num_steps = 200  # 500
    batch_size = 128
    display_step = 100

    # Network Parameters
    n_hidden_1 = 256  # 1st layer number of neurons
    n_hidden_2 = 256  # 2nd layer number of neurons
    num_input = 784  # MNIST data input (img shape: 28*28)
    num_classes = 10  # MNIST total classes (0-9 digits)

    # tf Graph input
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    # Store layers weight & bias
    weights = {
        "h1": tf.Variable(tf.random.normal([num_input, n_hidden_1]), name="h1"),
        "h2": tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2]), name="h2"),
        "out": tf.Variable(tf.random.normal([n_hidden_2, num_classes]), name="h_out"),
    }
    biases = {
        "b1": tf.Variable(tf.random.normal([n_hidden_1]), name="b1"),
        "b2": tf.Variable(tf.random.normal([n_hidden_2]), name="b2"),
        "out": tf.Variable(tf.random.normal([num_classes]), name="b_out"),
    }

    # Create model
    def neural_net(x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(x, weights["h1"]), biases["b1"])
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, weights["h2"]), biases["b2"])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, weights["out"]) + biases["out"]
        return out_layer

    # Construct model
    logits = neural_net(X)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=logits)
    # Using a functional loss will fail because TF optimizes away the mean.
    # See https://stackoverflow.com/questions/58532324/tf-gradients-dont-flow-through-tf-reduce-mean
    # loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    return train_op, X, Y


def get_data() -> "tf.contrib.learn.python.learn.datasets.base.Datasets":
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    return mnist


### Used for tf.keras


def get_keras_data(n_examples=32) -> Tuple:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255
    return (x_train[:n_examples], y_train[:n_examples]), (x_test[:n_examples], y_test[:n_examples])


def get_keras_model_v1():
    import tensorflow.compat.v1.keras as keras

    inputs = keras.Input(shape=(784,), name="img")
    x = keras.layers.Dense(64, activation="relu")(inputs)
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
    return model


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == "GPU"]
