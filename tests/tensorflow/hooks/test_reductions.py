import os
from datetime import datetime
from tornasole.core.reduction_config import ALLOWED_REDUCTIONS, ALLOWED_NORMS
from tornasole.exceptions import *

def simple_model(hook, steps=10, lr=0.4):
  import tensorflow as tf
  from tornasole.tensorflow import TornasoleOptimizer
  import numpy as np

  # Network definition
  with tf.name_scope('foobar'):
    x = tf.placeholder(shape=(None, 2), dtype=tf.float32)
    w = tf.Variable(initial_value=[[10.], [10.]], name='weight1')
  with tf.name_scope('foobaz'):
    w0 = [[1], [1.]]
    y = tf.matmul(x, w0)
  loss = tf.reduce_mean((tf.matmul(x, w) - y) ** 2, name="loss")

  global_step = tf.Variable(17, name="global_step", trainable=False)
  increment_global_step_op = tf.assign(global_step, global_step + 1)

  optimizer = tf.train.AdamOptimizer(lr)
  optimizer = TornasoleOptimizer(optimizer)
  optimizer_op = optimizer.minimize(loss, global_step=increment_global_step_op)

  sess = tf.train.MonitoredSession(hooks=[hook])

  for i in range(steps):
    x_ = np.random.random((10, 2)) * 0.1
    _loss, opt, gstep = sess.run([loss, optimizer_op, increment_global_step_op], {x: x_})
    print(f'Step={i}, Loss={_loss}')

  sess.close()

def get_dirs_files(path):
  entries = os.listdir(path)
  onlyfiles = [f for f in entries if os.path.isfile(os.path.join(path, f))]
  subdirs = [x for x in entries if x not in onlyfiles]
  return subdirs, onlyfiles


def test_reductions():
  from tornasole.tensorflow import TornasoleHook, \
    get_collections, ReductionConfig, SaveConfig, reset_collections
  import tensorflow as tf

  run_id = 'trial_' + datetime.now().strftime('%Y%m%d-%H%M%S%f')
  trial_dir = os.path.join('/tmp/tornasole_rules_tests/', run_id)

  tf.reset_default_graph()
  reset_collections()

  rdnc = ReductionConfig(reductions=ALLOWED_REDUCTIONS,
                         abs_reductions=ALLOWED_REDUCTIONS,
                         norms=ALLOWED_NORMS,
                         abs_norms=ALLOWED_NORMS)
  hook = TornasoleHook(out_dir=trial_dir,
                       save_config=SaveConfig(save_interval=1),
                       reduction_config=rdnc)

  simple_model(hook)
  _, files = get_dirs_files(trial_dir)
  coll = get_collections()
  from tornasole.trials import create_trial

  tr = create_trial(trial_dir)
  assert len(tr.tensors()) == 2
  for tname in tr.tensors():
    t = tr.tensor(tname)
    try:
      t.value(0)
      assert False
    except TensorUnavailableForStep:
      pass
    assert len(t.reduction_values(0)) == 18
    for r in ALLOWED_REDUCTIONS + ALLOWED_NORMS:
      for b in [False, True]:
        assert t.reduction_value(0, reduction_name=r, abs=b) is not None