# Third Party
import numpy as np
from tensorflow.keras.layers import Concatenate, Dense
from tensorflow.python.keras.models import Model

# First Party
import smdebug.tensorflow as smd
from smdebug.trials import create_trial


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.con = Concatenate()
        self.dense = Dense(10, activation="relu")

    def call(self, x):
        x = self.con([x, x])
        return self.dense(x)


def test_multiple_inputs(out_dir):
    my_model = MyModel()
    hook = smd.KerasHook(
        out_dir, save_all=True, save_config=smd.SaveConfig(save_steps=[0], save_interval=1)
    )

    hook.register_model(my_model)
    x_train = np.random.random((1000, 20))
    y_train = np.random.random((1000, 1))
    my_model.compile(optimizer="Adam", loss="mse", run_eagerly=True)
    my_model.fit(x_train, y_train, epochs=1, steps_per_epoch=1, callbacks=[hook])

    trial = create_trial(path=out_dir)
    tnames = sorted(trial.tensor_names(collection=smd.CollectionKeys.LAYERS))
    assert "concatenate" in tnames[0]
    assert len(trial.tensor(tnames[0]).value(0)) == 2
    assert trial.tensor(tnames[0]).shape(0) == (2, 1000, 20)
