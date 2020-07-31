# Third Party
import numpy as np
from bokeh.io import output_notebook, push_notebook, show
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show

output_notebook(hide_banner=True)

MICROS = 1000000.0
class StepHistogram:
    def __init__(self, metrics_reader, width=500):

        self.metrics_reader = metrics_reader

        # get timestamp of latest file and events
        self.last_timestamp = self.metrics_reader.get_timestamp_of_latest_available_file()

        self.step_metrics = {}
        self.sources = {}
        # number of datapoints to plot
        self.width = width

    def _create_step_metrics(self, all_events, step_metrics={}):
        for event in all_events:
            if "Step:ModeKeys" not in event.event_name:
                continue
            if event.event_name not in step_metrics:
                step_metrics[event.event_name] = []
            step_metrics[event.event_name].append(event.duration / MICROS)
            
        return step_metrics
    
    def plot(starttime_since_epoch_in_micros=0, endtime_since_epoch_in_micros=None, select_metrics=None):
        if endtime_since_epoch_in_micros == None:
            endtime_since_epoch_in_micros = self.metrics_reader.get_timestamp_of_latest_available_file()
        all_events = self.metrics_reader.get_events(starttime_since_epoch_in_micros, endtime_since_epoch_in_micros)
        self.last_timestamp = endtime_since_epoch_in_micros
        
        self.step_metrics = self._create_step_metrics(all_events)
        self.create_plot(self.step_metrics)

    def clear():
        self.step_metrics = {}
        self.sources = {}

    def _get_probs_binedges(self, metrics_arr):
        steps_np = np.array(metrics_arr)
        values = self.steps_np[:-1]
        min_value = np.min(values)
        max_value = np.median(values) + 2 * np.std(values)

        # define histogram bins
        bins = np.arange(min_value, max_value, (max_value - min_value) / 50.0)
        probs, binedges = np.histogram(self.steps_np[:-1], bins=bins)
        bincenters = 0.5 * (binedges[1:] + binedges[:-1])
        return probs, binedges

    def create_plot(self, step_metrics={}):
        metrics = list(step_metrics.keys())
        figures = []
        self.sources = {}

        # create a histogram per metric
        for index, metric in enumerate(metrics):
            p = figure(plot_height=350, plot_width=450)            
            # creates bins
            probs, binedges = self._get_probs_binedges(step_metrics[metric])
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

            # create plot
            p.y_range.start = 0
            p.xaxis.axis_label = metric + " step time in ms"
            p.yaxis.axis_label = "Occurences"
            p.grid.grid_line_color = "white"
            figures.append(p)

        p = gridplot(figures, ncols=4)
        self.target = show(p, notebook_handle=True)

    def update_data(self, current_timestamp):
        # get new events
        events = self.metrics_reader.get_events(self.last_timestamp, current_timestamp)
        self.last_timestamp = current_timestamp
        self.step_metrics = self._create_step_metrics(events, self.step_metrics)
        # update histograms
        for index, metric in enumerate(self.metrics):
            probs, binedges = self._get_probs_binedges(self.step_metrics[metrics])
            # update data
            self.sources[metric].data["top"] = probs
            self.sources[metric].data["left"] = binedges[:-1]
            self.sources[metric].data["right"] = binedges[1:]
        push_notebook()