# Third Party
# Standard Library
import re

import numpy as np
from bokeh.io import output_notebook, push_notebook, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show

output_notebook(hide_banner=True)

MICROS = 1000000.0


class StepHistogram:
    def __init__(self, metrics_reader, width=500):

        self.metrics_reader = metrics_reader

        # get timestamp of latest file and events
        self.last_timestamp = self.metrics_reader.get_timestamp_of_latest_available_file()
        print(f"StepHistogram created, last_timestamp found:{self.last_timestamp}")

        self.step_metrics = {}
        self.sources = {}

        # number of datapoints to plot
        self.width = width

    def _get_filtered_list(self, step_metrics):
        filtered_metrics = []
        available_metrics = list(step_metrics.keys())
        print(f"Select metrics:{self.select_metrics}")
        print(f"Available_metrics: {available_metrics}")
        for metric in self.select_metrics:
            r = re.compile(metric)
            filtered_metrics.extend(list(filter(r.search, available_metrics)))
        print(f"Filtered metrics:{filtered_metrics}")
        # delete the keys which needs to be filtered out
        for key in available_metrics:
            if key not in filtered_metrics:
                del step_metrics[key]
        return step_metrics

    def _create_step_metrics(self, all_events, step_metrics={}):
        for event in all_events:
            if self.show_workers is True:
                event_unique_id = f"{event.event_phase}-nodeid:{str(event.node_id)}"
            else:
                event_unique_id = event.event_phase
            if event_unique_id not in step_metrics:
                step_metrics[event_unique_id] = []
            step_metrics[event_unique_id].append(event.duration / MICROS)
        step_metrics = self._get_filtered_list(step_metrics)
        return step_metrics

    """
    @param starttime: starttime_since_epoch_in_micros . Default vlaue is 0, which means that since start of training
    @param endtime: endtime_since_epoch_in_micros. Default value is metrics_reader.last_timestamp, i.e., latest timestamp seen by metrics_reader
    @param select_metrics: specifies list of metrics regexes that will be shown
    @param show_workers: if this is True, every metrics will be suffixed by node:$WORKER_ID, and for every metrics, graphs will be shown for each worker
    """

    def plot(
        self,
        starttime=0,
        endtime=None,
        select_metrics=["Step:ModeKeys", "Forward-node", "Backward\(post-forward\)-node"],
        show_workers=True,
    ):
        if endtime is None:
            endtime = self.metrics_reader.get_timestamp_of_latest_available_file()
        print(f"stephistogram getting events from {starttime} to {endtime}")
        all_events = self.metrics_reader.get_events(starttime, endtime)
        print(f"Total events fetched:{len(all_events)}")
        self.last_timestamp = endtime
        self.select_metrics = select_metrics
        self.show_workers = show_workers
        self.step_metrics = self._create_step_metrics(all_events)

        self.create_plot(step_metrics=self.step_metrics)

    def clear():
        self.step_metrics = {}
        self.sources = {}

    def _get_probs_binedges(self, metrics_arr):
        steps_np = np.array(metrics_arr)
        values = steps_np[:-1]
        min_value = np.min(values)
        max_value = np.median(values) + 2 * np.std(values)

        # define histogram bins
        bins = np.arange(min_value, max_value, (max_value - min_value) / 50.0)
        probs, binedges = np.histogram(steps_np[:-1], bins=bins)
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
        self.step_metrics = self._create_step_metrics(events, step_metrics=self.step_metrics)
        # update histograms
        for index, metric in enumerate(self.step_metrics):
            if metric in self.sources:
                probs, binedges = self._get_probs_binedges(self.step_metrics[metric])
                # update data
                self.sources[metric].data["top"] = probs
                self.sources[metric].data["left"] = binedges[:-1]
                self.sources[metric].data["right"] = binedges[1:]
        push_notebook(handle=self.target)
