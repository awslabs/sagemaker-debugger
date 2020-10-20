# Third Party
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.models import Model

# First Party
import smdebug.tensorflow as smd
from smdebug.trials import create_trial


class CustomModel(Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.dense = Dense(10, activation="relu")

    def call(self, x):
        x = self.dense(x)
        x = self.dense(x)
        return self.dense(x)


def test_layer_reusability(out_dir):
    model = CustomModel()
    hook = smd.KerasHook(
        out_dir,
        save_all=True,
        save_config=smd.SaveConfig(save_steps=[0], save_interval=1),
        reduction_config=smd.ReductionConfig(save_shape=True, save_raw_tensor=True),
    )

    hook.register_model(model)
    x_train = np.random.random((1000, 10))
    y_train = np.random.random((1000, 1))
    model.compile(optimizer="Adam", loss="mse", run_eagerly=True)
    model.fit(x_train, y_train, epochs=1, steps_per_epoch=1, callbacks=[hook])

    trial = create_trial(path=out_dir, name="training_run")
    tensor_names = trial.tensor_names(collection=smd.CollectionKeys.LAYERS)
    assert len(tensor_names) == 6
    for name in tensor_names:
        shape = trial.tensor(name).shape(step_num=0)
        assert shape == (1000, 10)
