# First Party
from smdebug.profiler.analysis.utils.profiler_data_to_pandas import PandasFrame


def test_pandas_frames(metricfolder="./tests/profiler/resources/test_traces"):
    pf = PandasFrame(metricfolder)
    system_metrics_df, framework_metrics_df = pf.get_latest_data()

    print(f"Number of rows in system metrics dataframe = {system_metrics_df.shape[0]}")
    assert system_metrics_df.shape[0] == 46

    print(f"Number of columns in system metrics dataframe = {system_metrics_df.shape[1]}")
    assert system_metrics_df.shape[1] == 4

    print(f"Number of rows in framework metrics dataframe = {framework_metrics_df.shape[0]}")
    assert framework_metrics_df.shape[0] == 3

    print(f"Number of columns in framework metrics dataframe = {framework_metrics_df.shape[1]}")
    assert framework_metrics_df.shape[1] == 6
