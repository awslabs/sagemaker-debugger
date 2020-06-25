# Third Party
import numpy as np
from bokeh.io import output_notebook, push_notebook, show
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show

output_notebook()


class StepHistogram:
    def __init__(self, metrics_reader, width=500):

        # read timeline
        self.metrics_reader = metrics_reader

        # get timestamp of latest file and events
        self.last_timestamp = self.metrics_reader.get_timestamp_of_latest_available_file()
        events = self.metrics_reader.get_events(0, self.last_timestamp)

        self.steps = []

        # process events
        for event in events:
            if event.event_name == "Step:ModeKeys.TRAIN":
                self.steps.append(event.duration / 1000000.0)

        # convert to numpy array
        self.steps_np = np.array(self.steps)
        self.steps_np = self.steps_np[1:]
        self.start = self.steps_np[0]

        # number of datapoints to plot
        self.width = width
        self.create_plot()

    def create_plot(self):

        p = figure(plot_height=350, plot_width=450)

        # get min and max values
        values = self.steps_np[:-1]
        min_value = np.min(values)
        max_value = np.median(values) + 2 * np.std(values)

        # define histogram bins
        bins = np.arange(min_value, max_value, (max_value - min_value) / 50.0)
        probs, binedges = np.histogram(self.steps_np[:-1], bins=bins)
        bincenters = 0.5 * (binedges[1:] + binedges[:-1])

        # creates bins
        self.source = ColumnDataSource(data=dict(top=probs, left=binedges[:-1], right=binedges[1:]))

        p.quad(
            top="top",
            bottom=0,
            left="left",
            right="right",
            source=self.source,
            fill_color="navy",
            line_color="white",
            alpha=0.5,
        )

        # create plot
        p.y_range.start = 0
        p.xaxis.axis_label = "Step duration in ms"
        p.yaxis.axis_label = "Occurences"
        p.grid.grid_line_color = "white"

        # show
        self.target = show(p, notebook_handle=True)

    def update_data(self, current_timestamp):

        # get new events
        events = self.metrics_reader.get_events(self.last_timestamp, current_timestamp)
        self.last_timestamp = current_timestamp

        if len(events) > 0:
            # process new events and append to list
            for event in events:
                if event.event_name == "Step:ModeKeys.TRAIN":
                    self.steps.append(event.duration / 1000000.0)

            # convert to numpy
            self.steps_np = np.array(self.steps)
            self.steps_np = self.steps_np[1:]

            # get new min and max values
            values = self.steps_np[:-1]
            min_value = np.min(values)
            # max_value = np.max(values)
            max_value = np.median(values) + 2 * np.std(values)

            # define new bins
            bins = np.arange(min_value, max_value, (max_value - min_value) / 50.0)
            probs, binedges = np.histogram(values, bins=bins)
            bincenters = 0.5 * (binedges[1:] + binedges[:-1])

            # update plot
            self.source.data["top"] = probs
            self.source.data["left"] = binedges[:-1]
            self.source.data["right"] = binedges[1:]

            push_notebook()
