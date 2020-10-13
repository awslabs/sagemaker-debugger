# Standard Library
import re

# Third Party
import numpy as np
from bokeh.io import output_notebook, push_notebook, show
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CustomJS, Div, HoverTool
from bokeh.models.glyphs import Circle, Line, Quad
from bokeh.plotting import figure, show

# First Party
from smdebug.profiler.utils import TimeUnits

output_notebook(hide_banner=True)


class TimelineCharts:
    def __init__(
        self,
        system_metrics_reader,
        framework_metrics_reader,
        starttime=0,
        endtime=None,
        select_dimensions=[".*"],
        select_events=[".*"],
        x=1000,
        show_workers=True,
    ):

        self.select_dimensions = select_dimensions
        self.select_events = select_events
        self.show_workers = show_workers

        # placeholder
        self.sources = {}
        self.available_dimensions = []
        self.available_events = []

        self.system_metrics_reader = system_metrics_reader
        self.framework_metrics_reader = framework_metrics_reader

        if endtime == None:
            # get timestamp of latest file and events
            self.last_timestamp_system_metrics = (
                self.system_metrics_reader.get_timestamp_of_latest_available_file()
            )
        else:
            self.last_timestamp_system_metrics = endtime
        events = self.system_metrics_reader.get_events(
            starttime, self.last_timestamp_system_metrics
        )
        # first timestamp
        self.start = 0  # replace with system_metrics_reader.get_first_available_timestamp()/1000000
        self.system_metrics = self.preprocess_system_metrics(events, system_metrics={})

        min_width = float("inf")
        for key in self.system_metrics.keys():
            if key.startswith("CPUUtilization"):
                width = self.system_metrics[key]["total"].shape[0]
                if width <= min_width:
                    min_width = width
        if x < min_width:
            self.width = x
        else:
            self.width = min_width - 1

        # create plot
        self.create_plot()

    def preprocess_system_metrics(self, events, system_metrics):

        # read all available system metric events and store them in dict
        for event in events:
            if self.show_workers is True:
                event_unique_id = f"{event.dimension}-nodeid:{str(event.node_id)}"
            else:
                event_unique_id = event.dimension
            if event_unique_id not in system_metrics:
                system_metrics[event_unique_id] = {}
                self.available_dimensions.append(event_unique_id)
            if event.name not in system_metrics[event_unique_id]:
                system_metrics[event_unique_id][event.name] = []
                self.available_events.append(event.name)
            system_metrics[event_unique_id][event.name].append([event.timestamp, event.value])

        for dimension in system_metrics:
            for event in system_metrics[dimension]:
                # convert to numpy
                system_metrics[dimension][event] = np.array(system_metrics[dimension][event])

                # subtract first timestamp
                system_metrics[dimension][event][:, 0] = (
                    system_metrics[dimension][event][:, 0] - self.start
                )

        # compute total utilization per event dimension
        for event_dimension in system_metrics:
            n = len(system_metrics[event_dimension])
            total = [sum(x) for x in zip(*system_metrics[event_dimension].values())]
            system_metrics[event_dimension]["total"] = np.array(total) / n
            self.available_events.append("total")

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

    def plot_system_metrics(self):

        self.figures = []
        x_range = None
        # iterate over metrics e.g. cpu usage, gpu usage, i/o reads and writes etc
        # create a histogram per dimension and event
        for dimension in self.filtered_dimensions:
            self.sources[dimension] = {}
            for event in self.filtered_events:
                if event in self.system_metrics[dimension]:

                    values = self.system_metrics[dimension][event]
                    values = values[values[:, 0].argsort()]
                    # set y ranges for cpu and gpu which are measured in percent
                    if "Utilization" in dimension or "Memory" in dimension:
                        y_range = (0, 102)
                    else:
                        y_range = (
                            np.min(values[-self.width :, 1]),
                            np.max(values[-self.width :, 1]),
                        )

                    # create figure: each system metric has its own figure

                    if x_range == None:
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
                    source = ColumnDataSource(data=dict(x=values[:, 0], y=values[:, 1]))

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
                        renderers=[p], tooltips=[("index", "$index"), ("(x,y)", "($x, $y)")]
                    )

                    plot.xaxis.axis_label = "Time in ms"
                    plot.yaxis.axis_label = dimension + "_" + event
                    plot.add_tools(hover)

                    # store figure and datasource
                    self.figures.append(plot)
                    self.sources[dimension][event] = source

        return self.figures

    def create_plot(self):

        self.div = Div(width=250, height=100, height_policy="fixed")
        figures = self.plot_system_metrics()
        p = column(figures)
        self.target = show(row(p, self.div), notebook_handle=True)

    def find_time_annotations(self, indexes):

        if len(indexes) > 0:
            cpu_util = None
            for key in self.system_metrics.keys():
                if key.startswith("CPUUtilization"):
                    width = self.system_metrics[key]["total"].shape[0]
                    if cpu_util is None or np.min(indexes) <= width <= np.max(indexes):
                        cpu_util = self.system_metrics[key]

            begin_timestamp = cpu_util["total"][np.min(indexes), 0]
            end_timestamp = cpu_util["total"][np.max(indexes), 0]
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
                    key = event.event_name + "_" + str(event.event_args["step_num"])
                else:
                    key = event.event_name
                if key not in cumulative_time:
                    cumulative_time[key] = 0

                cumulative_time[key] += event.duration / 1000000000.0

            for event_name in cumulative_time:
                print(f"Spent {cumulative_time[event_name]} ms (cumulative time) in {event_name}")
        else:
            print("No selection made")

    def plot_framework_events(self, events, begin_timestamp, end_timestamp):
        framework_events = {}
        yaxis = {}
        counter = 0
        for index, event in enumerate(events):
            if event.event_args is not None:
                if "bytes_fetched" in event.event_args or "worker_id" in event.event_args:
                    continue

            if event.event_phase not in framework_events:
                framework_events[event.event_phase] = []
                yaxis[event.event_phase] = counter
                counter += 1

            framework_events[event.event_phase].append(
                [
                    int(event.start_time / 1000.0),
                    int(event.end_time / 1000.0),
                    yaxis[event.event_phase],
                ]
            )
            if index > 500:
                print(
                    """Reached more than 500 datapoints.
                      Will only plot first 500 datapoints for the given timerange"""
                )
                break
        return framework_events

    def plot_dataloaders(self, events, begin_timestamp, end_timestamp):
        # read all available system metric events and store them in dict
        dataloaders = {}
        tids = {}

        for index, event in enumerate(events):
            if event.event_args is None:
                continue
            if "bytes_fetched" in event.event_args:
                if event.event_name not in dataloaders:
                    dataloaders[event.event_name] = []
                if event.tid not in tids:
                    tids[event.tid] = len(tids.keys())
                dataloaders[event.event_name].append(
                    [int(event.start_time / 1000.0), int(event.end_time / 1000.0), tids[event.tid]]
                )
            elif "worker_id" in event.event_args:
                if event.event_name not in dataloaders:
                    dataloaders[event.event_name] = []
                worker_id = event.event_args["worker_id"]
                if worker_id not in tids:
                    tids[worker_id] = len(tids.keys())
                dataloaders[event.event_name].append(
                    [
                        int(event.start_time / 1000.0),
                        int(event.end_time / 1000.0),
                        tids[event.event_args["worker_id"]],
                    ]
                )

                if index > 500:
                    print("Reached more than 500 datapoints. Will stop plotting.")
                    break

        return dataloaders

    def plot_detailed_profiler_data(self, indexes):

        if len(indexes) > 0:
            cpu_util = None
            for key in self.system_metrics.keys():
                if key.startswith("CPUUtilization"):
                    width = self.system_metrics[key]["cpu0"].shape[0]
                    if cpu_util is None or np.min(indexes) <= width <= np.max(indexes):
                        cpu_util = self.system_metrics[key]

            begin_timestamp = cpu_util["cpu0"][np.min(indexes), 0]
            end_timestamp = cpu_util["cpu0"][np.max(indexes), 0]
            print(
                f"Selected timerange: {begin_timestamp + self.start} to {end_timestamp + self.start}"
            )
            events = self.framework_metrics_reader.get_events(
                begin_timestamp + self.start, end_timestamp + self.start, unit=TimeUnits.SECONDS
            )

            dataloaders = self.plot_dataloaders(
                events, begin_timestamp + self.start, end_timestamp + self.start
            )
            framework_events = self.plot_framework_events(
                events, begin_timestamp + self.start, end_timestamp + self.start
            )

            # define figure
            plot_dataloaders = figure(
                plot_height=250,
                plot_width=1000,
                tools="crosshair,xbox_select,pan,reset,save,xwheel_zoom",
            )
            plot_dataloaders.xaxis.axis_label = "Time in ms"
            plot_dataloaders.yaxis.axis_label = "Thread ID"
            # tooltip
            hover = HoverTool(tooltips=[("metric", "@metric"), ("index", "$x{10}")])

            for event in dataloaders.keys():
                for entry in range(len(dataloaders[event])):
                    # create source that contains time annotations
                    source = ColumnDataSource(
                        data=dict(
                            top=[dataloaders[event][entry][2]],
                            bottom=[dataloaders[event][entry][2] - 1],
                            left=[dataloaders[event][entry][0]],
                            right=[dataloaders[event][entry][1]],
                            metric=[event],
                        )
                    )

                    # vertical bars
                    quad = Quad(
                        top="top",
                        bottom="bottom",
                        left="left",
                        right="right",
                        fill_color="black",
                        line_color=None,
                        fill_alpha=0.2,
                    )

                    # plot
                    plot_dataloaders.add_glyph(source, quad)
            plot_dataloaders.add_tools(hover)

            plot_framework_events = figure(
                plot_height=250,
                plot_width=1000,
                tools="crosshair,xbox_select,pan,reset,save,xwheel_zoom",
            )
            plot_framework_events.xaxis.axis_label = "Time in ms"
            plot_framework_events.yaxis.axis_label = "Framework metric"
            # tooltip
            hover = HoverTool(tooltips=[("metric", "@metric"), ("index", "$x{10}")])

            for event in framework_events.keys():
                for entry in range(len(framework_events[event])):
                    # create source that contains time annotations
                    source = ColumnDataSource(
                        data=dict(
                            top=[framework_events[event][entry][2]],
                            bottom=[framework_events[event][entry][2] - 1],
                            left=[framework_events[event][entry][0]],
                            right=[framework_events[event][entry][1]],
                            metric=[event],
                        )
                    )

                    # vertical bars
                    quad = Quad(
                        top="top",
                        bottom="bottom",
                        left="left",
                        right="right",
                        fill_color="blue",
                        fill_alpha=0.2,
                        line_color=None,
                    )

                    # plot
                    plot_framework_events.add_glyph(source, quad)
            plot_framework_events.add_tools(hover)

            p = column([plot_dataloaders, plot_framework_events])
            self.target = show(p, notebook_handle=True)
        else:
            print("No selection made")

    def update_data(self, current_timestamp):

        # get all events from last to current timestamp
        events = self.system_metrics_reader.get_events(
            self.last_timestamp_system_metrics, current_timestamp
        )
        print(
            f"Found {len(events)} new system metrics events from timestamp_in_us:{self.last_timestamp_system_metrics} to timestamp_in_us:{current_timestamp}"
        )
        if len(events) > 0:
            new_system_metrics = self.preprocess_system_metrics(events, system_metrics={})

            # append numpy arrays to previous numpy arrays
            for dimension in self.filtered_dimensions:
                for event in self.filtered_events:
                    if event in self.system_metrics[dimension]:
                        new_system_metrics[dimension][event] = new_system_metrics[dimension][event][
                            new_system_metrics[dimension][event][:, 0].argsort()
                        ]
                        self.system_metrics[dimension][event] = np.vstack(
                            [
                                self.system_metrics[dimension][event],
                                new_system_metrics[dimension][event],
                            ]
                        )
                        self.system_metrics[dimension][event] = self.system_metrics[dimension][
                            event
                        ][self.system_metrics[dimension][event][:, 0].argsort()]

            max_width = 0
            cpu_util = None
            for key in self.system_metrics.keys():
                if key.startswith("CPUUtilization"):
                    width = self.system_metrics[key]["total"].shape[0]
                    if cpu_util is None or width >= max_width:
                        max_width = width
                        cpu_util = self.system_metrics[key]

            self.width = max_width - 1

            if self.width > 1000:
                min_value = cpu_util["total"][-1000, 0]
            else:
                min_value = cpu_util["total"][-self.width, 0]
            max_value = cpu_util["total"][-1, 0]

            for figure in self.figures:
                figure.x_range.start = int(min_value)
                figure.x_range.end = int(max_value)

            # update line charts with system metrics
            for dimension in self.filtered_dimensions:
                for event in self.filtered_events:
                    if event in self.system_metrics[dimension]:
                        values = np.array(self.system_metrics[dimension][event])
                        self.sources[dimension][event].data["x"] = values[:, 0]
                        self.sources[dimension][event].data["y"] = values[:, 1]

            self.last_timestamp_system_metrics = current_timestamp
            push_notebook()
