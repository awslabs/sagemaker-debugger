# Standard Library
import re

# Third Party
import numpy as np
from bokeh.io import output_notebook, push_notebook, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show

output_notebook(hide_banner=True)


class MetricsHistogram:
    def __init__(self, metrics_reader):

        self.metrics_reader = metrics_reader

        # get timestamp of latest files
        self.last_timestamp = self.metrics_reader.get_timestamp_of_latest_available_file()
        self.seen_system_metric_list = set()
        self.select_metrics = []
        self.sources = {}
        self.target = None

    """
    @param starttime is starttime_since_epoch_in_micros. Default value 0, which means start
    @param endtime is endtime_since_epoch_in_micros. Default value is  MetricsHistogram.last_timestamp , i.e., last_timestamp seen by system_metrics_reader
    @param select_metrics is array of metrics to be selected, Default ["cpu", "gpu"]
    """

    def plot(self, starttime=0, endtime=None, select_metrics=[".*"]):
        if endtime == None:
            endtime = self.metrics_reader.get_timestamp_of_latest_available_file()
        all_events = self.metrics_reader.get_events(starttime, endtime)
        print(
            f"Found {len(all_events)} system metrics events from timestamp_in_us:{starttime} to timestamp_in_us:{endtime}"
        )
        self.last_timestamp = endtime
        self.select_metrics = select_metrics
        self.system_metrics = self.preprocess_system_metrics(all_events=all_events)
        self.create_plot(self.system_metrics)

    def clear():
        self.system_metrics = {}
        self.sources = {}

    def preprocess_system_metrics(self, all_events=[], system_metrics={}):
        cpu_name = None
        # read all available system metric events and store them in dict
        for event in all_events:
            if event.name not in system_metrics:
                system_metrics[event.name] = []
                if cpu_name is None and event.dimension == "CPUUtilization":
                    cpu_name = event.name
                    print(cpu_name)
            system_metrics[event.name].append(event.value)

        # total cpu utilization is not recorded in SM
        if cpu_name is not None:
            self.cores = 0.0
            cpu_total = np.zeros(len(system_metrics[cpu_name]))
            for metric in system_metrics:
                # TODO should we do similar for gpu too
                if "cpu" in metric and metric:
                    if metric not in self.seen_system_metric_list:
                        self.cores += 1
                        self.seen_system_metric_list.add(metric)

                    cpu_total += system_metrics[metric]

            system_metrics["cpu_total"] = cpu_total / self.cores

        # add user defined metrics to the list
        filtered_metrics = []
        available_metrics = list(system_metrics.keys())
        print(f"select metrics:{self.select_metrics}")

        for metric in self.select_metrics:
            r = re.compile(r".*" + metric + r".*")
            filtered_metrics.extend(list(filter(r.search, available_metrics)))
        print(f"filtered_metrics:{filtered_metrics}")

        # delete the keys which needs to be filtered out
        for key in available_metrics:
            if key not in filtered_metrics and "total" not in key:
                del system_metrics[key]
        return system_metrics

    def _get_probs_binedges(self, values):
        # create histogram bins
        bins = np.arange(0, 100, 2)
        probs, binedges = np.histogram(values, bins=bins)
        bincenters = 0.5 * (binedges[1:] + binedges[:-1])
        return probs, binedges

    def create_plot(self, system_metrics={}):
        metrics = list(system_metrics.keys())
        figures = []

        # create a histogram per metric
        for index, metric in enumerate(metrics):
            p = figure(plot_height=250, plot_width=250)
            probs, binedges = self._get_probs_binedges(system_metrics[metric])
            # set data
            source = ColumnDataSource(data=dict(top=probs, left=binedges[:-1], right=binedges[1:]))
            self.sources[metric] = source
            p.quad(
                top="top",
                bottom=0,
                left="left",
                right="right",
                source=source,
                fill_color="navy",
                line_color="white",
                fill_alpha=0.5,
            )

            # set plot
            p.y_range.start = 0
            p.xaxis.axis_label = metric + " util"
            p.yaxis.axis_label = "Occurences"
            p.grid.grid_line_color = "white"
            figures.append(p)

        p = gridplot(figures, ncols=4)
        self.target = show(p, notebook_handle=True)

    def update_data(self, current_timestamp):
        # get all events from last to current timestamp
        events = self.metrics_reader.get_events(self.last_timestamp, current_timestamp)
        self.last_timestamp = current_timestamp

        self.system_metrics = self.preprocess_system_metrics(events, self.system_metrics)

        # update histograms
        for index, metric in enumerate(self.system_metrics):
            values = self.system_metrics[metric]

            # create new histogram bins
            probs, binedges = self._get_probs_binedges(self.system_metrics[metric])
            # update data
            self.sources[metric].data["top"] = probs
            self.sources[metric].data["left"] = binedges[:-1]
            self.sources[metric].data["right"] = binedges[1:]
        push_notebook()
