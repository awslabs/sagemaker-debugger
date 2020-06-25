# Standard Library
import re

# Third Party
import numpy as np
from bokeh.io import output_notebook, push_notebook, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show

output_notebook()


class MetricsHistogram:
    def __init__(self, metrics_reader, select_metrics=None):

        self.metrics_reader = metrics_reader

        # get timestamp of latest files
        self.last_timestamp = self.metrics_reader.get_timestamp_of_latest_available_file()
        self.all_events = self.metrics_reader.get_events(0, self.last_timestamp)

        # define the list of metrics to plot: per default cpu and gpu
        self.select_metrics = ["cpu", "gpu"]
        if select_metrics is not None:
            self.select_metrics.extend(select_metrics)

        self.preprocess_system_metrics()

        self.create_plot()

    def preprocess_system_metrics(self):

        # read all available system metric events and store them in dict
        self.system_metrics = {}
        for event in self.all_events:
            if (
                event.name not in self.system_metrics
                and event.dimension is not "GPUMemoryUtilization"
            ):
                self.system_metrics[event.name] = []
            self.system_metrics[event.name].append(event.value)

        # total cpu utilization is not recorded in SM
        self.cores = 0.0
        cpu_total = np.zeros(len(self.system_metrics["cpu0"]))
        for metric in self.system_metrics:
            if "cpu" in metric:
                self.cores += 1
                cpu_total += self.system_metrics[metric]

        self.system_metrics["cpu_total"] = cpu_total / self.cores

        # number of datapoints
        self.width = self.system_metrics["cpu_total"].shape[0]

        # add user defined metrics to the list
        self.metrics = []
        available_metrics = list(self.system_metrics.keys())

        for metric in self.select_metrics:
            r = re.compile(".*" + metric)
            self.metrics.extend(list(filter(r.match, available_metrics)))

    def create_plot(self):

        figures = []
        self.sources = {}

        # create a histogram per metric
        for index, metric in enumerate(self.metrics):
            p = figure(plot_height=250, plot_width=250)
            values = self.system_metrics[metric]

            # create histogram bins
            bins = np.arange(0, 100, 2)
            probs, binedges = np.histogram(values, bins=bins)
            bincenters = 0.5 * (binedges[1:] + binedges[:-1])

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
            p.xaxis.axis_label = metric
            p.grid.grid_line_color = "white"
            figures.append(p)

        p = gridplot(figures, ncols=4)
        self.target = show(p, notebook_handle=True)

    def update_data(self, current_timestamp):

        # get all events from last to current timestamp
        events = self.metrics_reader.get_events(self.last_timestamp, current_timestamp)
        self.last_timestamp = current_timestamp

        if len(events) > 0:
            for event in events:
                if event.name != None:
                    self.system_metrics[event.name].append(event.value)

            cpu_total = np.zeros(len(self.system_metrics["cpu0"]))

            # iterate over available metrics
            for metric in self.system_metrics:

                # compute total cpu utilization
                if "cpu" in metric and metric != "cpu_total":
                    cpu_total += self.system_metrics[metric]

            self.system_metrics["cpu_total"] = cpu_total / self.cores

            # update histograms
            for index, metric in enumerate(self.metrics):
                values = self.system_metrics[metric]

                # create new histogram bins
                bins = np.arange(0, 100, 2)
                probs, binedges = np.histogram(values, bins=bins)
                bincenters = 0.5 * (binedges[1:] + binedges[:-1])

                # update data
                self.sources[metric].data["top"] = probs
                self.sources[metric].data["left"] = binedges[:-1]
                self.sources[metric].data["right"] = binedges[1:]

            push_notebook()
