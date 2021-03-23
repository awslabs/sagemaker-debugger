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
        self.system_metrics = {}
        self.select_dimensions = []
        self.select_events = []
        self.sources = {}
        self.target = None
        self.available_dimensions = []
        self.available_events = []

    """
    @param starttime is starttime_since_epoch_in_micros. Default value 0, which means start
    @param endtime is endtime_since_epoch_in_micros. Default value is  MetricsHistogram.last_timestamp , i.e., last_timestamp seen by system_metrics_reader
    @param select_metrics is array of metrics to be selected, Default ["cpu", "gpu"]
    """

    def plot(
        self,
        starttime=0,
        endtime=None,
        select_dimensions=[".*"],
        select_events=[".*"],
        show_workers=True,
    ):
        if endtime == None:
            endtime = self.metrics_reader.get_timestamp_of_latest_available_file()
        all_events = self.metrics_reader.get_events(starttime, endtime)
        print(
            f"Found {len(all_events)} system metrics events from timestamp_in_us:{starttime} to timestamp_in_us:{endtime}"
        )
        self.last_timestamp = endtime
        self.select_dimensions = select_dimensions
        self.select_events = select_events
        self.show_workers = show_workers
        self.system_metrics = self.preprocess_system_metrics(
            all_events=all_events, system_metrics={}
        )
        self.create_plot()

    def clear(self):
        self.system_metrics = {}
        self.sources = {}

    def preprocess_system_metrics(self, all_events=[], system_metrics={}):

        # read all available system metric events and store them in dict
        for event in all_events:
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
            system_metrics[event_unique_id][event.name].append(event.value)

        # compute total utilization per event dimension
        for event_dimension in system_metrics:
            n = len(system_metrics[event_dimension])
            total = [sum(x) for x in zip(*system_metrics[event_dimension].values())]
            system_metrics[event_dimension]["total"] = np.array(total) / n
            self.available_events.append("total")

        # add user defined metrics to the list
        self.filtered_events = []
        print(f"select events:{self.select_events}")
        self.filtered_dimensions = []
        print(f"select dimensions:{self.select_dimensions}")
        for metric in self.select_events:
            r = re.compile(r".*" + metric + r".*")
            self.filtered_events.extend(list(filter(r.search, self.available_events)))
        self.filtered_events = set(self.filtered_events)
        print(f"filtered_events:{self.filtered_events}")
        for metric in self.select_dimensions:
            r = re.compile(metric)  # + r".*")
            self.filtered_dimensions.extend(list(filter(r.search, self.available_dimensions)))
        self.filtered_dimensions = set(self.filtered_dimensions)
        print(f"filtered_dimensions:{self.filtered_dimensions}")

        return system_metrics

    def _get_probs_binedges(self, values):
        # create histogram bins
        bins = np.arange(0, 100, 2)
        probs, binedges = np.histogram(values, bins=bins)
        bincenters = 0.5 * (binedges[1:] + binedges[:-1])
        return probs, binedges

    def create_plot(self):
        figures = []

        # create a histogram per dimension and event
        for dimension in self.filtered_dimensions:
            self.sources[dimension] = {}
            for event in self.filtered_events:
                if event in self.system_metrics[dimension]:
                    p = figure(plot_height=250, plot_width=250)
                    probs, binedges = self._get_probs_binedges(
                        self.system_metrics[dimension][event]
                    )
                    # set data
                    source = ColumnDataSource(
                        data=dict(top=probs, left=binedges[:-1], right=binedges[1:])
                    )
                    self.sources[dimension][event] = source
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
                    p.xaxis.axis_label = dimension + "_" + event
                    p.yaxis.axis_label = "Occurences"
                    p.grid.grid_line_color = "white"
                    figures.append(p)

        p = gridplot(figures, ncols=4)
        self.target = show(p, notebook_handle=True)
        print(f"filtered_dimensions:{self.filtered_dimensions}")

    def update_data(self, current_timestamp):
        # get all events from last to current timestamp
        events = self.metrics_reader.get_events(self.last_timestamp, current_timestamp)
        self.last_timestamp = current_timestamp

        self.system_metrics = self.preprocess_system_metrics(events, self.system_metrics)

        # create a histogram per dimension and event
        for dimension in self.filtered_dimensions:
            for event in self.filtered_events:
                if event in self.system_metrics[dimension]:
                    values = self.system_metrics[dimension][event]

                    # create new histogram bins
                    probs, binedges = self._get_probs_binedges(
                        self.system_metrics[dimension][event]
                    )
                    # update data
                    self.sources[dimension][event].data["top"] = probs
                    self.sources[dimension][event].data["left"] = binedges[:-1]
                    self.sources[dimension][event].data["right"] = binedges[1:]
        push_notebook()
