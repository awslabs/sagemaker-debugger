# Third Party
import tensorflow as tf

# First Party
import smdebug.tensorflow as smd
from smdebug.core.collection import CollectionKeys


def create_hook(trial_dir):
    hook = smd.KerasHook(trial_dir, save_all=True)
    return hook


def create_model():
    model = tf.keras.models.Sequential(
        [
            # WA for TF issue https://github.com/tensorflow/tensorflow/issues/36279
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return model


def test_gradtape_tf_function(out_dir):
    def get_grads(images, labels):
        # with tf.GradientTape() as tape:
        return model(images, training=True)

    @tf.function
    def train_step(images, labels):
        return tf.reduce_mean(get_grads(images, labels))

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), _ = mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_train[..., tf.newaxis] / 255, tf.float32), tf.cast(y_train, tf.int64))
    )
    dataset = dataset.shuffle(1000).batch(64)
    model = create_model()
    hook = create_hook(out_dir)
    opt = tf.keras.optimizers.Adam()
    hook.wrap_optimizer(opt)

    n_epochs = 1
    for epoch in range(n_epochs):
        for data, labels in dataset:
            dataset_labels = labels
            labels = tf.one_hot(labels, depth=10)
            with hook.wrap_tape(tf.GradientTape()) as tape:
                logits = train_step(data, labels)
            grads = tape.gradient(logits, model.variables)
            opt.apply_gradients(zip(grads, model.variables))
            hook.save_tensor("inputs", data, CollectionKeys.INPUTS)
            hook.save_tensor("logits", logits, CollectionKeys.OUTPUTS)
            hook.save_tensor("labels", labels, CollectionKeys.OUTPUTS)

    model.save(out_dir, save_format="tf")
    hook.close()

    trial = smd.create_trial(out_dir)
    assert trial.tensor_names(collection=CollectionKeys.LOSSES) == ["loss"]
    assert trial.tensor_names(collection=CollectionKeys.WEIGHTS) == [
        "weights/dense/kernel:0",
        "weights/dense_1/kernel:0",
    ]
    assert trial.tensor_names(collection=CollectionKeys.BIASES) == [
        "weights/dense/bias:0",
        "weights/dense_1/bias:0",
    ]
    assert trial.tensor_names(collection=CollectionKeys.OPTIMIZER_VARIABLES) == [
        "Adam/beta_1:0",
        "Adam/beta_2:0",
        "Adam/decay:0",
        "Adam/iter:0",
        "Adam/learning_rate:0",
    ]
    assert trial.tensor_names(collection=CollectionKeys.INPUTS) == ["inputs"]
    assert trial.tensor_names(collection=CollectionKeys.OUTPUTS) == ["labels", "logits"]
