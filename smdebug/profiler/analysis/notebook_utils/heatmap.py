# Standard Library
import re

# Third Party
import bokeh
import numpy as np
from bokeh.io import output_notebook, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models.glyphs import Image
from bokeh.models.tickers import FixedTicker
from bokeh.plotting import figure, show

output_notebook(hide_banner=True)


class Heatmap:
    def __init__(
        self,
        metrics_reader,
        select_metrics=[],
        starttime=0,
        endtime=None,
        select_dimensions=[".*CPU", ".*GPU", ".*Memory"],
        select_events=[".*"],
        plot_height=350,
        show_workers=True,
    ):

        self.select_dimensions = select_dimensions
        self.select_events = select_events
        self.show_workers = show_workers
        self.metrics_reader = metrics_reader
        self.available_dimensions = []
        self.available_events = []
        self.start = 0  # replace with system_metrics_reader.get_first_available_timestamp()/1000000

        if endtime == None:
            # get timestamp of latest file and events
            self.last_timestamp_system_metrics = (
                self.metrics_reader.get_timestamp_of_latest_available_file()
            )
        else:
            self.last_timestamp_system_metrics = endtime
        events = self.metrics_reader.get_events(starttime, self.last_timestamp_system_metrics)

        self.plot_height = plot_height

        # get timestamp of latest file and events
        self.last_timestamp = self.metrics_reader.get_timestamp_of_latest_available_file()
        events = self.metrics_reader.get_events(0, self.last_timestamp)

        self.system_metrics = self.preprocess_system_metrics(events, system_metrics={})
        self.create_plot()

    def preprocess_system_metrics(self, events, system_metrics):

        # read all available system metric events and store them in dict
        for event in events:
            if event.node_id not in system_metrics:
                system_metrics[event.node_id] = {}
            if event.dimension not in system_metrics[event.node_id]:
                system_metrics[event.node_id][event.dimension] = {}
            if event.name not in system_metrics[event.node_id][event.dimension]:
                system_metrics[event.node_id][event.dimension][event.name] = []
            system_metrics[event.node_id][event.dimension][event.name].append(
                [event.timestamp, event.value]
            )

        for node in system_metrics:
            for dimension in system_metrics[node]:
                if dimension not in self.available_dimensions:
                    self.available_dimensions.append(dimension)
                for event in system_metrics[node][dimension]:
                    if event not in self.available_events:
                        self.available_events.append(event)
                    # convert to numpy
                    system_metrics[node][dimension][event] = np.array(
                        system_metrics[node][dimension][event]
                    )

                    # Convert network and I/O metrics into percentage (max color value in heatmap is 100)
                    if dimension in ["Algorithm", "Platform", ""]:
                        max_value = np.max(system_metrics[node][dimension][event])
                        system_metrics[node][dimension][event] = (
                            system_metrics[node][dimension][event] / max_value
                        )
                        system_metrics[node][dimension][event] = (
                            system_metrics[node][dimension][event] * 100
                        )

                    # subtract first timestamp
                    system_metrics[node][dimension][event][:, 0] = (
                        system_metrics[node][dimension][event][:, 0] - self.start
                    )

        # compute total utilization per event dimension
        for node in system_metrics:
            for dimension in system_metrics[node]:
                n = len(system_metrics[node][dimension])
                total = [sum(x) for x in zip(*system_metrics[node][dimension].values())]
                system_metrics[node][dimension]["total"] = np.array(total) / n
                self.available_events.append("total")

        # compute total utilization per node
        nodes = list(system_metrics.keys())
        system_metrics["node_total"] = {}
        for dimension in system_metrics[nodes[0]]:
            system_metrics["node_total"][dimension] = {}
            node_total = []
            for node in nodes:
                len2 = len(node_total)
                if len2 > 0:
                    len1 = system_metrics[node][dimension]["total"].shape[0]
                    if len1 < len2:
                        node_total[:len1] = (
                            node_total[:len1] + system_metrics[node][dimension]["total"]
                        )
                    else:
                        node_total = node_total + system_metrics[node][dimension]["total"][:len2]
                else:
                    node_total = system_metrics[node][dimension]["total"]

            system_metrics["node_total"][dimension]["total"] = node_total / (len(nodes))

        self.filtered_events = []
        print(f"select events:{self.select_events}")
        self.filtered_dimensions = []
        print(f"select dimensions:{self.select_dimensions}")
        for metric in self.select_events:
            r = re.compile(r".*" + metric)
            self.filtered_events.extend(list(filter(r.search, self.available_events)))
        self.filtered_events = set(self.filtered_events)
        print(f"filtered_events:{self.filtered_events}")
        for metric in self.select_dimensions:
            r = re.compile(metric)  # + r".*")
            self.filtered_dimensions.extend(list(filter(r.search, self.available_dimensions)))
        self.filtered_dimensions = set(self.filtered_dimensions)
        print(f"filtered_dimensions:{self.filtered_dimensions}")

        return system_metrics

    def create_plot(self):

        # define list of metric names (needed for tooltip)
        tmp = []
        metric_names = []
        yaxis = {}

        # number of datapoints
        max_width = 0
        for key in self.system_metrics.keys():
            if key.startswith("CPUUtilization"):
                width = self.system_metrics[key]["total"].shape[0]
                if width >= max_width:
                    max_width = width
        self.width = max_width

        for node in self.system_metrics:
            for dimension in self.filtered_dimensions:
                for event in self.system_metrics[node][dimension]:
                    if event in self.filtered_events:
                        values = self.system_metrics[node][dimension][event][: self.width, 1]
                        tmp.append(values)
                        metric_names.insert(0, dimension + "_" + event + "_" + node),
                        timestamps = self.system_metrics[node][dimension][event][: self.width, 0]

        for index, metric in enumerate(metric_names):
            yaxis[len(tmp) - index] = metric

        ymax = len(tmp) + 1
        yaxis[ymax] = ""

        # define figure
        start = 0
        if self.width > 1000:
            start = self.width - 1000
        self.plot = figure(
            plot_height=self.plot_height,
            x_range=(start, self.width),
            y_range=(0, ymax),
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
                x=[0] * ymax,
                y=[i for i in range(ymax)],
                dw=[self.width] * (ymax),
                dh=[1.3] * (ymax),
                metric=[i for i in metric_names],
            )
        )

        images = Image(image="image", x="x", y="y", dw="dw", dh="dh", color_mapper=color_mapper)

        # plot
        self.plot.add_glyph(self.source, images)
        self.plot.add_tools(hover)
        self.plot.xgrid.visible = False
        self.plot.ygrid.visible = False
        self.plot.yaxis.ticker = FixedTicker(ticks=np.arange(0, ymax).tolist())
        self.plot.yaxis.major_label_text_font_size = "5pt"
        self.plot.yaxis.major_label_overrides = yaxis

        self.target = show(self.plot, notebook_handle=True)
