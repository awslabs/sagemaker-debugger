# Example Scripts
This folder has some example scripts which invoke rules that we have written.
The way to run the scripts is straight forward.

### Vanishing Gradient
```
python check_grads.py --trial-dir ~/ts_outputs/grads/
```

### Similar across runs
This scripts you how to check which tensors have different values across two runs.
You pass both trials as `trial-dir`.
```
python similar_across_runs.py \
    --trial-dir ~/ts_outputs/trial1 \
    --trial-dir ~/ts_outputs/trial2 
```

## Weight Update Ratio
This script lets you monitor the ratio of weights to updates.
You can configure the thresholds by passing them.

```
python weight_update_ratio.py \
    --trial-dir ~/ts_outputs/trial \
    --large-threshold 10 \
    --small-threshold 0.00000001
```
