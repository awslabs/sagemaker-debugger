# Standard Library
import re

# Third Party
import numpy as np
from bokeh.io import output_notebook, push_notebook, show
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CustomJS, Div, HoverTool
from bokeh.models.glyphs import Circle, Line
from bokeh.plotting import figure, show

# First Party
from smdebug.profiler.utils import TimeUnits

output_notebook()


class TimelineCharts:
    def __init__(self, system_metrics_reader, framework_metrics_reader, select_metrics=[], x=1000):

        # placeholder
        self.system_metrics_sources = {}

        # read and preprocess system metrics: per default we only show overall cpu and gpu usage
        self.select_metrics = ["cpu", "gpu"]
        if select_metrics is not None:
            self.select_metrics.extend(select_metrics)

        self.system_metrics_reader = system_metrics_reader
        self.framework_metrics_reader = framework_metrics_reader

        # get timestamp of latest file and events
        self.last_timestamp_system_metrics = (
            self.system_metrics_reader.get_timestamp_of_latest_available_file()
        )
        self.system_metrics_events = self.system_metrics_reader.get_events(
            0, self.last_timestamp_system_metrics
        )

        self.preprocess_system_metrics()

        if x < self.system_metrics["cpu_total"].shape[0]:
            self.width = x
        else:
            self.width = self.system_metrics["cpu_total"].shape[0] - 1

        # create plot
        self.create_plot()

    def preprocess_system_metrics(self):

        # read all available system metric events and store them in dict
        self.system_metrics = {}
        for event in self.system_metrics_events:
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

        # add user defined metrics to the list
        self.metrics = []
        available_metrics = list(self.system_metrics.keys())

        for metric in self.select_metrics:
            r = re.compile(".*" + metric)
            self.metrics.extend(list(filter(r.match, available_metrics)))

    def plot_system_metrics(self):

        self.figures = []

        # iterate over metrics e.g. cpu usage, gpu usage, i/o reads and writes etc
        for index, metric in enumerate(self.metrics):

            values = self.system_metrics[metric]
            values = values[values[:, 0].argsort()]
            # set y ranges for cpu and gpu which are measured in percent
            if "gpu" in metric or "cpu" in metric:
                y_range = (0, 102)
            else:
                y_range = (np.min(values[-self.width :, 1]), np.max(values[-self.width :, 1]))

            # create figure: each system metric has its own figure
            if index == 0:
                plot = figure(
                    plot_height=200,
                    plot_width=1000,
                    x_range=(values[-self.width, 0], values[-1, 0]),
                    y_range=y_range,
                    tools="crosshair,xbox_select,pan,reset,save,xwheel_zoom",
                )
                x_range = plot.x_range
            else:
                plot = figure(
                    plot_height=200,
                    plot_width=1000,
                    x_range=x_range,
                    y_range=y_range,
                    tools="crosshair,xbox_select,pan,reset,save,xwheel_zoom",
                )

            plot.xgrid.visible = False
            plot.ygrid.visible = False

            # create line chart for system metric
            source = ColumnDataSource(
                data=dict(
                    x=values[:, 0],
                    y=values[:, 1],
                    # gpu0=self.system_metrics["gpu0"][:,1],
                    cpu=self.system_metrics["cpu_total"],
                    # memory=self.system_metrics["memory"][-self.width:,1]
                )
            )

            callback = CustomJS(
                args=dict(s1=source, div=self.div),
                code="""
                    console.log('Running CustomJS callback now.');
                    var inds = s1.selected.indices;
                    console.log(inds);
                    var line = "<span style=float:left;clear:left;font_size=13px><b> Selected index range: [" + Math.min.apply(Math,inds) + "," + Math.max.apply(Math,inds) + "]</b></span>\\n";
                    console.log(line)
                    var text = div.text.concat(line);
                    var lines = text.split("\\n")
                    if (lines.length > 35)
                        lines.shift();
                    div.text = lines.join("\\n");""",
            )

            plot.js_on_event("selectiongeometry", callback)

            line = Line(x="x", y="y", line_color="blue")
            circle = Circle(x="x", y="y", fill_alpha=0, line_width=0)
            p = plot.add_glyph(source, line)
            p = plot.add_glyph(source, circle)

            # create tooltip for hover tool
            hover = HoverTool(
                renderers=[p],
                tooltips=[
                    ("index", "$index"),
                    ("(x,y)", "($x, $y)"),  # ("gpu0", "@gpu0"),
                    ("cpu", "@cpu"),  # ("memory", "@memory")
                ],
            )

            plot.xaxis.axis_label = "Time in ms"
            plot.yaxis.axis_label = metric
            plot.add_tools(hover)

            # store figure and datasource
            self.figures.append(plot)
            self.system_metrics_sources[metric] = source

        return self.figures

    def create_plot(self):

        self.div = Div(width=250, height=100, height_policy="fixed")
        figures = self.plot_system_metrics()
        p = column(figures)
        self.target = show(row(p, self.div), notebook_handle=True)

    def find_time_annotations(self, indexes):

        if len(indexes) > 0:
            begin_timestamp = self.system_metrics["cpu_total"][np.min(indexes), 0]
            end_timestamp = self.system_metrics["cpu_total"][np.max(indexes), 0]
            total_time = end_timestamp - begin_timestamp
            print(
                f"Selected timerange: {begin_timestamp + self.start} to {end_timestamp + self.start}"
            )

            cumulative_time = {}
            events = self.framework_metrics_reader.get_events(
                begin_timestamp + self.start, end_timestamp + self.start, unit=TimeUnits.SECONDS
            )
            for event in events:
                if event.event_args is not None and "step_num" in event.event_args:
                    key = event.event_name + "_" + event.event_args["step_num"]
                else:
                    key = event.event_name
                if key not in cumulative_time:
                    cumulative_time[key] = 0

                cumulative_time[key] += event.duration / 1000000000.0

            for event_name in cumulative_time:
                print(f"Spent {cumulative_time[event_name]} ms (cumulative time) in {event_name}")
        else:
            print("No selection made")

    def update_data(self, current_timestamp):

        # get all events from last to current timestamp
        events = self.system_metrics_reader.get_events(
            self.last_timestamp_system_metrics, current_timestamp
        )

        if len(events) > 0:
            new_system_metrics = {}
            for event in events:
                if event.dimension == "GPUMemoryUtilization":
                    continue

                if event.name not in new_system_metrics:
                    new_system_metrics[event.name] = []
                new_system_metrics[event.name].append([event.timestamp, event.value])
            self.last_timestamp_system_metrics = current_timestamp
            cpu_total = np.zeros((len(new_system_metrics["cpu0"]), 2))

            # iterate over available metrics
            for metric in new_system_metrics:

                # convert to numpy
                new_system_metrics[metric] = np.array((new_system_metrics[metric]))

                # subtract first timestamp
                new_system_metrics[metric][:, 0] = new_system_metrics[metric][:, 0] - self.start

                # compute total cpu utilization
                if metric is not None and "cpu" in metric:
                    cpu_total[:, 0] = new_system_metrics[metric][:, 0]
                    cpu_total[:, 1] += new_system_metrics[metric][:, 1]

            new_system_metrics["cpu_total"] = cpu_total
            new_system_metrics["cpu_total"][:, 1] = (
                new_system_metrics["cpu_total"][:, 1] / self.cores
            )

            # append numpy arrays to previous numpy arrays
            for metric in self.system_metrics:
                new_system_metrics[metric] = new_system_metrics[metric][
                    new_system_metrics[metric][:, 0].argsort()
                ]
                self.system_metrics[metric] = np.vstack(
                    [self.system_metrics[metric], new_system_metrics[metric]]
                )
                self.system_metrics[metric] = self.system_metrics[metric][
                    self.system_metrics[metric][:, 0].argsort()
                ]
            self.width = self.system_metrics["cpu_total"].shape[0] - 1

            if self.width > 1000:
                min_value = self.system_metrics["cpu_total"][-1000, 0]
            else:
                min_value = self.system_metrics["cpu_total"][-self.width, 0]
            max_value = self.system_metrics["cpu_total"][-1, 0]

            for figure in self.figures:
                figure.x_range.start = int(min_value)
                figure.x_range.end = int(max_value)

            # update line charts with system metrics

            for metric in self.metrics:
                values = np.array(self.system_metrics[metric])
                self.system_metrics_sources[metric].data["x"] = values[:, 0]
                self.system_metrics_sources[metric].data["y"] = values[:, 1]
                # self.system_metrics_sources[metric].data['gpu0'] = self.system_metrics['gpu0'][:,1]
                self.system_metrics_sources[metric].data["cpu"] = self.system_metrics["cpu_total"][
                    :, 1
                ]
                # self.system_metrics_sources[metric].data['memory'] = self.system_metrics['memory'][:,1]

            push_notebook()
