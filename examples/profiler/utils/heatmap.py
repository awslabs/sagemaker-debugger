# Standard Library
import re

# Third Party
import bokeh
import numpy as np
from bokeh.io import output_notebook, push_notebook, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models.glyphs import Image
from bokeh.plotting import figure, show

output_notebook()


class Heatmap:
    def __init__(self, metrics_reader, select_metrics=[]):

        self.metrics_reader = metrics_reader

        # get timestamp of latest file and events
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
            if event.dimension == "GPUMemoryUtilization":
                continue
            if event.name not in self.system_metrics:
                self.system_metrics[event.name] = []
            self.system_metrics[event.name].append([event.timestamp, event.value])

        # total cpu utilization is not recorded in SM
        cpu_total = np.zeros((len(self.system_metrics["cpu0"]), 2))

        # first timestamp
        self.start = self.system_metrics["cpu0"][0][0]

        self.cores = 0.0
        for metric in self.system_metrics:

            # convert to numpy
            self.system_metrics[metric] = np.array(self.system_metrics[metric])

            # subtract first timestamp
            self.system_metrics[metric][:, 0] = self.system_metrics[metric][:, 0] - self.start

            # compute total cpu utilization
            if "cpu" in metric:
                self.cores += 1
                cpu_total[:, 0] = self.system_metrics[metric][:, 0]
                cpu_total[:, 1] += self.system_metrics[metric][:, 1]

        self.system_metrics["cpu_total"] = cpu_total
        self.system_metrics["cpu_total"][:, 1] = cpu_total[:, 1] / self.cores

        # number of datapoints
        self.width = self.system_metrics["cpu_total"].shape[0]

        # add user defined metrics to the list
        self.metrics = []
        available_metrics = list(self.system_metrics.keys())

        for metric in self.select_metrics:
            r = re.compile(".*" + metric)
            self.metrics.extend(list(filter(r.match, available_metrics)))

    def create_plot(self):

        # define list of metric names (needed for tooltip)
        tmp = []
        metric_names = []
        yaxis = {}
        for index, metric in enumerate(self.metrics):
            values = self.system_metrics[metric][: self.width, 1]
            tmp.append(values)
            metric_names.append(metric),
            timestamps = self.system_metrics[metric][: self.width, 0]
            yaxis[index] = metric
        yaxis[index + 1] = ""

        # define figure
        start = 0
        if self.width > 1000:
            start = self.width - 1000
        self.plot = figure(
            plot_height=150,
            x_range=(start, self.width),
            y_range=(0, index + 1),
            plot_width=1000,
            tools="crosshair,reset,xwheel_zoom, box_edit",
        )
        self.plot.xaxis.axis_label = "Indices"
        # tooltip
        hover = HoverTool(
            tooltips=[("usage", "@image"), ("metric", "@metric"), ("index", "$x{10}")]
        )

        color_mapper = bokeh.models.LinearColorMapper(bokeh.palettes.viridis(100))
        color_mapper.high = 100
        color_mapper.low = 0

        tmp = np.array(tmp)
        self.source = ColumnDataSource(
            data=dict(
                image=[np.array(tmp[i]).reshape(1, -1) for i in range(len(tmp))],
                x=[0] * (index + 1),
                y=[i for i in range(index + 1)],
                dw=[self.width] * (index + 1),
                dh=[1.3] * (index + 1),
                metric=[i for i in metric_names],
            )
        )

        images = Image(image="image", x="x", y="y", dw="dw", dh="dh", color_mapper=color_mapper)

        # plot
        self.plot.add_glyph(self.source, images)
        self.plot.add_tools(hover)
        self.plot.xgrid.visible = False
        self.plot.ygrid.visible = False
        self.plot.xaxis.major_tick_line_color = None
        self.plot.xaxis.minor_tick_line_color = None
        self.plot.yaxis.major_tick_line_color = None
        self.plot.yaxis.minor_tick_line_color = None
        self.plot.yaxis.major_label_overrides = yaxis

        self.target = show(self.plot, notebook_handle=True)

    def update_data(self, current_timestamp):

        # get all events from last to current timestamp
        events = self.metrics_reader.get_events(self.last_timestamp, current_timestamp)
        self.last_timestamp = current_timestamp

        if len(events) > 0:

            new_system_metrics = {}
            for event in events:
                # get_events may return different types of events
                if event.name not in new_system_metrics:
                    new_system_metrics[event.name] = []
                new_system_metrics[event.name].append([event.timestamp, event.value])

            cpu_total = np.zeros((len(new_system_metrics["cpu0"]), 2))

            # iterate over available metrics
            for metric in new_system_metrics:

                # convert to numpy
                new_system_metrics[metric] = np.array((new_system_metrics[metric]))

                # subtract first timestamp
                new_system_metrics[metric][:, 0] = new_system_metrics[metric][:, 0] - self.start

                # compute total cpu utilization
                if "cpu" in metric:
                    cpu_total[:, 0] = new_system_metrics[metric][:, 0]
                    cpu_total[:, 1] += new_system_metrics[metric][:, 1]

            new_system_metrics["cpu_total"] = cpu_total
            new_system_metrics["cpu_total"][:, 1] = (
                new_system_metrics["cpu_total"][:, 1] / self.cores
            )

            # append numpy arrays to previous numpy arrays
            for metric in self.system_metrics:
                self.system_metrics[metric] = np.vstack(
                    [self.system_metrics[metric], new_system_metrics[metric]]
                )

            self.width = self.system_metrics["cpu_total"].shape[0]

            tmp = []
            metric_names = []
            for index, metric in enumerate(self.metrics):
                values = self.system_metrics[metric][:, 1]
                tmp.append(values)
                metric_names.append(metric)

            # update heatmap
            images = [np.array(tmp[i]).reshape(1, -1) for i in range(len(tmp))]
            self.source.data["image"] = images
            self.source.data["dw"] = [self.width] * (index + 1)
            self.source.data["metric"] = metric_names

            if self.width > 1000:
                self.plot.x_range.start = self.width - 1000
            self.plot.x_range.end = self.width
            push_notebook()
