# First Party
from smdebug.exceptions import RuleEvaluationConditionMet
from smdebug.rules.rule import Rule


class OverallFrameworkMetrics(Rule):
    def __init__(self, base_trial, scan_interval_us=60 * 1000 * 1000):
        """
        This rule summarizes the time spent in framework metrics such as forward and backward pass, dataloading.
        :param base_trial: the trial whose execution will invoke the rule
        :param scan_interval_us: interval with which timeline files are scanned. Default is 60000000 us.
        """
        super().__init__(base_trial)
        self.scan_interval_us = scan_interval_us
        self.last_timestamp = self.base_trial.first_timestamp
        self.report["RuleParameters"] = f""

        self.gpu_counts = 1
        self.gpu_events = {}
        self.cpu_events = {}
        self.step_phases = {}
        self.forward_events = {}
        self.backward_events = {}
        self.phase_durations = {}
        self.horovod = {}
        self.start = None

        self.report["RuleParameters"] = f""
        self.report["Details"] = {}

    def invoke_at_step(self, step):
        pass

    def reset(self):
        self.gpus = {}
        self.histogram = {}

    def invoke(self, step):
        # iterate over timeline events
        current_timestamp = self.last_timestamp + self.scan_interval_us
        self.base_trial.wait_for_data(current_timestamp, self.last_timestamp)
        rule_condition = self.invoke_for_timerange(self.last_timestamp, current_timestamp)
        self.last_timestamp = current_timestamp
        if rule_condition:
            raise RuleEvaluationConditionMet(self.rule_name, step)

    def invoke_for_timerange(
        self, timestamp_start, timestamp_end, sys_events=None, framework_events=None
    ):

        # get framework metric events
        if framework_events is None:
            events = self.base_trial.get_framework_metrics(timestamp_start, timestamp_end)
        else:
            events = framework_events

        # aggregate framework metrics
        for event in events:

            # CPU functions
            if "cpu" in event.event_phase.lower():
                if event.event_phase not in self.cpu_events:
                    self.cpu_events[event.event_phase] = {}
                if event.event_name not in self.cpu_events[event.event_phase]:
                    self.cpu_events[event.event_phase][event.event_name] = 0
                self.cpu_events[event.event_phase][event.event_name] += (
                    event.end_time - event.start_time
                )

            # GPU functions
            elif "gpu" in event.event_phase.lower():
                if event.event_phase not in self.gpu_events:
                    self.gpu_events[event.event_phase] = {}
                if event.event_name not in self.gpu_events[event.event_phase]:
                    self.gpu_events[event.event_phase][event.event_name] = 0
                self.gpu_events[event.event_phase][event.event_name] += (
                    event.end_time - event.start_time
                )
                self.gpu_counts = len(self.gpu_events)

            # ratio between TRAIN/EVAL and others
            elif "Step" in event.event_phase:
                if "step_num" in event.event_args:
                    if self.start == None:
                        self.start = event.start_time
                    end = event.end_time
                if event.event_phase not in self.step_phases:
                    self.step_phases[event.event_phase] = 0
                self.step_phases[event.event_phase] += event.end_time - event.start_time
                self.step_phases["others"] = (end - self.start) - sum(
                    self.step_phases.values()
                ) / self.gpu_counts

            # forward (only PT)
            elif event.event_phase == "Forward-SubModuleInternal":
                if event.event_name not in self.forward_events:
                    self.forward_events[event.event_name] = 0
                self.forward_events[event.event_name] += event.end_time - event.start_time

            # backward (only PT)
            elif event.event_phase == "Backward-SubModuleInternal":
                if event.event_name not in self.backward_events:
                    self.backward_events[event.event_name] = 0
                self.backward_events[event.event_name] += event.end_time - event.start_time

            # annotated events from backend (dataloader, dataiter, NCCLManager)
            elif (
                ":CPU" not in event.event_phase
                and ":GPU" not in event.event_phase
                and event.file_type != ""
            ):
                if event.event_phase not in self.phase_durations:
                    self.phase_durations[event.event_phase] = 0
                self.phase_durations[event.event_phase] += event.duration

            # Horovod events (source/file_type is an empty string)
            elif event.file_type == "":
                if event.event_name not in self.horovod:
                    self.horovod[event.event_name] = 0
                self.horovod[event.event_name] += event.end_time - event.start_time

        # convert values to percentages and filter small events
        labels_cpu = [key for key in self.gpu_events]
        labels_gpu = [key for key in self.cpu_events]
        labels = labels_cpu
        labels.extend(labels_gpu)
        totals_gpu = [sum(self.gpu_events[key].values()) for key in self.gpu_events]
        totals_cpu = [sum(self.cpu_events[key].values()) for key in self.cpu_events]
        totals = totals_gpu
        totals.extend(totals_cpu)

        sizes = [size / sum(totals) * 100.0 for size in totals]
        self.report["Details"]["ratio"] = {}

        # record information for profiler report
        for label, size in zip(labels, sizes):
            self.report["Details"]["ratio"][label] = size

        # create chart for detailed CPU functions
        if len(self.cpu_events) > 0:

            # convert values to percentages and filter small events
            labels, sizes, times = self.filter_events(self.cpu_events, threshold=2)

            # record information for profiler report
            self.report["Details"]["CPU"] = {}
            self.report["Details"]["CPU_total"] = {}
            for label, size, time in zip(labels, sizes, times):
                self.report["Details"]["CPU"][label] = size
                self.report["Details"]["CPU_total"][label] = time

        # record detailed GPU functions
        if len(self.gpu_events) > 0:

            # convert values to percentages and filter small events
            labels, sizes, times = self.filter_events(self.gpu_events, threshold=2)

            # record information for profiler report
            self.report["Details"]["GPU"] = {}
            self.report["Details"]["GPU_total"] = {}
            for label, size, time in zip(labels, sizes, times):
                self.report["Details"]["GPU"][label] = size
                self.report["Details"]["GPU_total"][label] = time

        # record train/eval phase and others
        labels = self.step_phases.keys()
        sizes = [float(i) / sum(self.step_phases.values()) * 100 for i in self.step_phases.values()]

        # record information for profiler report
        self.report["Details"]["phase"] = {}
        self.report["Details"]["phase_time"] = {}
        for label, size in zip(labels, sizes):
            self.report["Details"]["phase"][label] = size

        # breakdown for generic framework metrics
        if len(self.phase_durations) > 0:

            labels = list(self.phase_durations.keys())
            sizes = [
                float(i) / sum(self.phase_durations.values()) * 100
                for i in self.phase_durations.values()
            ]

            # record information for profiler report
            self.report["Details"]["general"] = {}
            for label, size in zip(labels, sizes):
                self.report["Details"]["general"][label] = size

        # breakdown for forward/backward passes (only PT)
        if len(self.forward_events) > 0 and len(self.backward_events) > 0:

            totals = [sum(self.forward_events.values())]
            totals.append(sum(self.backward_events.values()))

            labels = ["Forward pass", "Backward pass"]
            sizes = [float(i) / sum(totals) * 100 for i in totals]

            # record information for profiler report
            self.report["Details"]["forward_backward"] = {}
            for label, size in zip(labels, sizes):
                self.report["Details"]["forward_backward"][label] = size

        # breakdown of Horovod events
        if len(self.horovod) > 0:

            # filter out small events
            filtered_events = {}
            total = sum(self.horovod.values())
            for event in self.horovod:
                if self.horovod[event] > 0 and self.horovod[event] / total > 0.02:
                    filtered_events[event] = self.horovod[event]

            labels = list(filtered_events.keys())
            sizes = [
                float(i) / sum(filtered_events.values()) * 100 for i in filtered_events.values()
            ]

            # record information for profiler report
            self.report["Details"]["horovod"] = {}
            for label, size in zip(labels, sizes):
                self.report["Details"]["horovod"][label] = size

    def filter_events(self, dict_events, threshold):

        # get list of common operators
        operators = {}
        for phase_name in dict_events:
            for operator_name in dict_events[phase_name]:
                if operator_name not in operators:
                    operators[operator_name] = 0
                operators[operator_name] += dict_events[phase_name][operator_name]
        labels = list(operators.keys())

        # convert total time values per operator to percentages
        sizes = [float(operators[i]) / sum(operators.values()) * 100 for i in operators.keys()]
        times = list(operators.values())

        labels_filtered = []
        sizes_filtered = []
        times_filtered = []

        # filter out everything that is below threshold (do avoid messy charts)
        for index, (s, l, t) in enumerate(zip(sizes, labels, times)):
            if s > threshold:
                labels_filtered.append(l)
                sizes_filtered.append(s)
                times_filtered.append(t)

        # re-calculate perentages and find most expensive
        sizes_filtered = [i / sum(sizes_filtered) * 100 for i in sizes_filtered]

        return labels_filtered, sizes_filtered, times_filtered
