# Third Party
import tensorflow as tf
import tensorflow_datasets as tfds
from tests.constants import TEST_DATASET_S3_PATH

# First Party
from smdebug.tensorflow import EstimatorHook, modes


def test_keras_to_estimator(out_dir):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(16, activation="relu", input_shape=(4,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    def input_fn():
        split = tfds.Split.TRAIN
        dataset = tfds.load("iris", data_dir=TEST_DATASET_S3_PATH, split=split, as_supervised=True)
        dataset = dataset.map(lambda features, labels: ({"dense_input": features}, labels))
        dataset = dataset.batch(32).repeat()
        return dataset

    model.compile(loss="categorical_crossentropy", optimizer="adam")
    model.summary()

    keras_estimator = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=out_dir)

    hook = EstimatorHook(out_dir)

    hook.set_mode(modes.TRAIN)
    keras_estimator.train(input_fn=input_fn, steps=25, hooks=[hook])

    hook.set_mode(modes.EVAL)
    eval_result = keras_estimator.evaluate(input_fn=input_fn, steps=10, hooks=[hook])

    from smdebug.trials import create_trial

    tr = create_trial(out_dir)
    assert len(tr.tensor_names()) == 1
    assert len(tr.steps()) == 2
    assert len(tr.steps(modes.TRAIN)) == 1
    assert len(tr.steps(modes.EVAL)) == 1
