# Third Party
import numpy as np
from bokeh.io import push_notebook, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models.glyphs import Line
from bokeh.plotting import figure, show


class StepTimelineChart:
    def __init__(self, metrics_reader=None, width=1000):

        self.metrics_reader = metrics_reader

        # get last timestamp
        self.last_timestamp = self.metrics_reader.get_timestamp_of_latest_available_file()

        # read timeline
        self.metrics_reader.get_events(0, self.last_timestamp)

        self.steps = []
        self.metric_names = []
        # read events for given timerange
        events = self.metrics_reader.get_events(0, self.last_timestamp)

        # process events
        for event in events:
            if event.event_name.startswith("Step"):
                self.metric_names.append(event.event_name)
                self.steps.append(
                    [
                        event.start_time / 1000000.0,
                        event.duration / 1000000.0,
                        int(event.event_args["step_num"]),
                    ]
                )
        # convert to numpy array
        self.steps_np = np.array(self.steps)
        self.steps_np = self.steps_np[1:, :]
        self.start = self.steps_np[0, 0]
        self.steps_np = self.steps_np[self.steps_np[:, 0].argsort()]

        # number of datapoints to plot
        self.width = width

        # create plot
        self.create_plot()

    def create_plot(self):

        # plot either last 500 or last x datapoints
        if self.steps_np.shape[0] >= self.width:
            start_index = self.width
        else:
            start_index = self.steps_np.shape[0] - 1

        # create figure: set the xrange to only show last start_index datapoints
        self.plot = figure(
            plot_height=250,
            plot_width=1000,
            x_range=(
                self.steps_np[-start_index, 0] - self.start,
                self.steps_np[-1, 0] - self.start,
            ),
            tools="crosshair,pan,reset,save,wheel_zoom",
        )

        # create line chart for step duration
        self.source = ColumnDataSource(
            data=dict(
                x=self.steps_np[:, 0] - self.start,
                y=self.steps_np[:, 1],
                step=self.steps_np[:, 2],
                metric=self.metric_names,
            )
        )
        line = Line(x="x", y="y", line_color="blue")

        # tooltip
        hover = HoverTool(
            tooltips=[
                ("index", "$index"),
                ("step", "@step"),
                ("metric", "@metric"),
                ("(x,y)", "($x, $y)"),
            ]
        )

        # create plot
        self.plot.add_tools(hover)
        self.plot.add_glyph(self.source, line)
        self.plot.xaxis.axis_label = "Time in ms"
        self.plot.yaxis.axis_label = "Step duration in ms"
        self.plot.xgrid.visible = False
        self.plot.ygrid.visible = False

        # show
        self.target = show(self.plot, notebook_handle=True)

    def update_data(self, current_timestamp):

        # get new events
        events = self.metrics_reader.get_events(self.last_timestamp, current_timestamp)
        self.last_timestamp = current_timestamp

        if len(events) > 0:
            # process new events and append to list
            for event in events:
                if event.event_name.startswith("Step"):
                    self.metric_names.append(event.event_name)
                    self.steps.append(
                        [
                            event.start_time / 1000000.0,
                            event.duration / 1000000.0,
                            int(event.event_args["step_num"]),
                        ]
                    )
            # convert to numpy
            self.steps_np = np.array(self.steps)
            self.steps_np = self.steps_np[self.steps_np[:, 0].argsort()]

            # check how many datapoints can be plotted
            if self.steps_np.shape[0] > self.width:
                start_index = self.width
                self.plot.x_range.start = self.steps_np[-self.width, 0] - self.start
                self.plot.y_range.end = np.max(self.steps_np[-self.width, 1])
            else:
                start_index = self.steps_np.shape[0] - 1
                self.plot.x_range.start = self.steps_np[-start_index, 0] - self.start
                self.plot.y_range.end = np.max(self.steps_np[-start_index:, 1])

            # set new datarange
            self.plot.x_range.end = self.steps_np[-1, 0] - self.start

            # update line chart
            self.source.data["x"] = self.steps_np[:, 0] - self.start
            self.source.data["y"] = self.steps_np[:, 1]
            self.source.data["step"] = self.steps_np[:, 2]
            self.source.data["metric"] = self.metric_names
            push_notebook()
