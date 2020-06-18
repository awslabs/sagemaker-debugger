# Overview
SageMaker Debugger Profiler provides better insights for training jobs. Customer can use SystemMetricsReader to monitor
system metrics (CPU, GPU, etc.) and find issues during training.

# Examples

The below code snippet shows how to use SystemMetricsReader in local mode and s3 mode.

## Examples for reading system profiler metrics locally
```
from smdebug.profiler.SystemMetricsReader import SystemLocalMetricsReader
lt = SystemLocalMetricsReader('/localpath/profiler-output')
events = lt.get_events(1591100000, 1692300000, unit=TimeUnits.SECONDS)
```

## Example for reading system profiler metrics from s3
```
from smdebug.profiler.SystemMetricsReader import SystemS3MetricsReader
s3Path = "s3://bucket/prefix/trainingjob_name/profiler-output"
tt = SystemS3MetricsReader(s3Path)
events = tt.get_events(1591100000, 1692300000, unit=TimeUnits.SECONDS)
```
