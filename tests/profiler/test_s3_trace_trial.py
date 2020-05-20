# First Party
from smdebug.profiler.TraceTrial import S3TraceTrial


def test_trace_trial():
    bucket_name = "tornasole-dev"
    tt = S3TraceTrial(bucket_name)
    events = tt.get_events(1589930980, 1589930995)
    print(f"Number of events {len(events)}")
