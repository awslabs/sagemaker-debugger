# First Party
# Third Party
import numpy as np

from smdebug.exceptions import RuleEvaluationConditionMet
from smdebug.profiler.profiler_constants import (
    PT_DATALOADER_INITIALIZE,
    PT_DATALOADER_ITER,
    PT_DATALOADER_WORKER,
    TF_DATALOADER_ITER,
)
from smdebug.rules.rule import Rule


class Dataloader(Rule):
    def __init__(
        self,
        base_trial,
        min_threshold=70,
        max_threshold=200,
        patience=10,
        scan_interval_us=60000000,
    ):
        """
        This rule helps to detect how many dataloader processes are running in parallel and whether the total number is equal the number of available CPU cores.
        :param min_threshold: how many cores should be at least used by dataloading processes. Default 70%
        :param max_threshold: how many cores should be at maximum used by dataloading processes. Default 200%
        :param patience: how many events to capture before running the first evluation. default:10
        :param base_trial: the trial whose execution will invoke the rule
        :param scan_interval_us: interval with which timeline files are scanned. Default is 60000000 us.
        """
        super().__init__(base_trial)
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.patience = patience
        self.scan_interval_us = scan_interval_us
        self.last_timestamp = self.base_trial.first_timestamp

        self.parallel_dataloaders = []
        self.cpus = {}
        self.dataloading_time = []
        self.pin_memory = None
        self.num_workers = None
        self.prefetch = False
        self.report[
            "RuleParameters"
        ] = f"min_threshold:{self.min_threshold}\nmax_threshold:{self.max_threshold}"

    def invoke_at_step(self, step):
        pass

    def invoke(self, step):
        # iterate over timeline events
        current_timestamp = self.last_timestamp + self.scan_interval_us
        self.base_trial.wait_for_data(current_timestamp, self.last_timestamp)
        rule_condition = self.invoke_for_timerange(self.last_timestamp, current_timestamp)
        self.last_timestamp = current_timestamp
        if rule_condition:
            raise RuleEvaluationConditionMet(self.rule_name, step)

    def record_event(self, dataloaders, event):

        self.dataloading_time.append((event.end_time - event.start_time) / 1000000)

        # count dataloading events per second and per process ID
        timestamp_s = int(event.start_time / 1000000)
        if timestamp_s not in dataloaders:
            dataloaders[timestamp_s] = {}
        if "pid" in event.event_args:
            if event.event_args["pid"] not in dataloaders[timestamp_s]:
                dataloaders[timestamp_s][event.event_args["pid"]] = 0
            dataloaders[timestamp_s][event.event_args["pid"]] += 1

    def invoke_for_timerange(
        self, timestamp_start, timestamp_end, sys_events=None, framework_events=None
    ):

        # get number of available cores
        if len(self.cpus) == 0:
            if sys_events is None:
                events = self.base_trial.get_system_metrics(timestamp_start, timestamp_end)
            else:
                events = sys_events
            for event in events:
                if event.dimension == "CPUUtilization" and event.name not in self.cpus:
                    self.cpus[event.name] = 0

        # get framework metric events
        if framework_events is None:
            fw_events = self.base_trial.get_framework_metrics(timestamp_start, timestamp_end)
        else:
            fw_events = framework_events

        dataloaders = {}
        for event in fw_events:

            # reset
            if len(self.parallel_dataloaders) > 1000:
                self.parallel_dataloaders = []

            # PT Dataloader intiliaze
            if event.event_phase == PT_DATALOADER_INITIALIZE:
                if "pin_memory" in event.event_args:
                    self.pin_memory = event.event_args["pin_memory"]
                if "num_workers" in event.event_args:
                    self.num_workers = event.event_args["num_workers"]

            # PT events
            if event.event_phase == PT_DATALOADER_WORKER or event.event_phase == PT_DATALOADER_ITER:

                # number of events processed
                self.report["Datapoints"] += 1

                # record event
                self.record_event(dataloaders, event)
                self.prefetch = None

            # TF prefetch
            if "Prefetch" in event.event_name:
                self.prefetch = True

            # do not run on TF events: will be fixed after GA
            # TF events
            # if event.event_phase == TF_DATALOADER_ITER and "GetNext" in event.event_name:

            # number of events processed
            # self.report["Datapoints"] += 1

            # record event
            # self.record_event(dataloaders, event)

        if len(dataloaders) > 0:

            # count the number of unique dataloader processes per time unit
            for timestamp in dataloaders:
                num_parallel_dataloaders = len(dataloaders[timestamp])
                self.parallel_dataloaders.append(num_parallel_dataloaders)

            # compare number of dataloaders versus available number of cores
            # raise warning if too few dataloaders
            median_dataloader_num = int(np.median(self.parallel_dataloaders))
            if len(self.parallel_dataloaders) > self.patience and median_dataloader_num < len(
                self.cpus
            ) * (self.min_threshold / 100):
                self.logger.info(
                    f"There are {len(self.cpus)} available cores, but on average there were only {median_dataloader_num} parallel dataloader processes running."
                )
                self.report["Violations"] += 1

            # raise warning if too many dataloaders
            if len(self.parallel_dataloaders) > self.patience and median_dataloader_num > len(
                self.cpus
            ) * (self.max_threshold / 100):
                self.logger.info(
                    f"There are {len(self.cpus)} available cores, but on average there were {median_dataloader_num} parallel dataloader processes running which is above the threshold {self.max_threshold}"
                )
                self.report["Violations"] += 1

            # record information for profiler report
            if self.pin_memory != None:
                self.report["Details"]["pin_memory"] = self.pin_memory
            if self.num_workers != None:
                self.report["Details"]["num_workers"] = self.num_workers
            if self.prefetch != None:
                self.report["Details"]["prefetch"] = self.prefetch

            self.report["Details"]["cores"] = len(self.cpus)
            self.report["Details"]["dataloaders"] = median_dataloader_num
            if len(self.dataloading_time) > 0:
                self.report["Details"]["dataloading_time"] = {}
                self.report["Details"]["dataloading_time"]["p25"] = float(
                    np.quantile(self.dataloading_time, 0.25)
                )
                self.report["Details"]["dataloading_time"]["p50"] = float(
                    np.quantile(self.dataloading_time, 0.50)
                )
                self.report["Details"]["dataloading_time"]["p95"] = float(
                    np.quantile(self.dataloading_time, 0.95)
                )

                probs, binedges = np.histogram(self.dataloading_time, bins=100)
                self.report["Details"]["dataloading_time"]["probs"] = probs.tolist()
                self.report["Details"]["dataloading_time"]["binedges"] = binedges.tolist()
        else:
            self.logger.info(f"No dataloading metrics found.")

        if self.report["Violations"] > 0:
            self.report["RuleTriggered"] = 1
            return True

        return False
