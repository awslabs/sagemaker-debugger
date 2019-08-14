from tornasole.core.writer import FileWriter
from tornasole.core.reader import FileReader
import numpy as np
from tornasole.core.modes import ModeKeys
from datetime import datetime
import glob
import shutil

def test_mode_writing():
  run_id = 'trial_' + datetime.now().strftime('%Y%m%d-%H%M%S%f')
  for s in range(0, 10):

    fw = FileWriter(logdir='ts_outputs', trial=run_id, step=s)
    if s % 2 == 0:
      fw.write_tensor(tdata=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                      tname='arr', mode=ModeKeys.TRAIN, mode_step=s//2)
    else:
      fw.write_tensor(tdata=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                      tname='arr', mode=ModeKeys.EVAL, mode_step=s // 2)
  fw.close()
  files = glob.glob('ts_outputs/' + run_id + '/**/*.tfevents',
                    recursive=True)
  for f in files:
    fr = FileReader(fname=f)
    for tu in fr.read_tensors():
      tensor_name, step, tensor_data, mode, mode_step = tu
      if step % 2 == 0:
        assert mode == ModeKeys.TRAIN
      else:
        assert mode == ModeKeys.EVAL
      assert mode_step == step // 2
  shutil.rmtree('ts_outputs/' + run_id)
