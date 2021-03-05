# Third Party
import tensorflow as tf
import tensorflow_datasets as tfds
from tests.utils import SagemakerSimulator
from transformers import (
    BertTokenizer,
    TFBertForSequenceClassification,
    glue_convert_examples_to_features,
)

# First Party
import smdebug.tensorflow as smd
from smdebug.core.collection import CollectionKeys


def test_bert_simple():
    # Test bert with the default smdebug configuration
    smd.del_hook()
    with SagemakerSimulator(enable_tb=False) as sim:
        epochs = 1
        model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        data = tfds.load("glue/mrpc")
        train_dataset = glue_convert_examples_to_features(
            data["train"], tokenizer, max_length=128, task="mrpc"
        )
        train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=optimizer, loss=loss)
        model.fit(train_dataset, epochs=epochs, steps_per_epoch=10)

    hook = smd.get_hook()
    assert hook.has_default_hook_configuration()
    hook.close()
    # Check that hook created and tensors saved
    trial = smd.create_trial(path=sim.out_dir)
    assert len(trial.steps()) > 0, "Nothing saved at any step."
    assert len(trial.tensor_names()) > 0, "Tensors were not saved."

    # DEFAULT TENSORS SAVED
    assert len(trial.tensor_names(collection=CollectionKeys.LOSSES)) > 0, "No Losses Saved"
    assert len(trial.tensor_names(collection=CollectionKeys.METRICS)) > 0, "No Metrics Saved"
    assert (
        len(trial.tensor_names(collection=CollectionKeys.WEIGHTS)) == 0
    ), "Weights were not expected to be saved by default"
    assert (
        len(trial.tensor_names(collection=CollectionKeys.BIASES)) == 0
    ), "Biases were not expected to be saved by default"
