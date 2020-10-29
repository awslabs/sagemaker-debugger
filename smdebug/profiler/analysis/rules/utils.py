def filter_events(dict_events, threshold):

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


def aggregate_framework_metrics(events, report, buffer, timestamp_us=None):
    cpu_events = buffer["cpu_events"]
    gpu_events = buffer["gpu_events"]
    step_phases = buffer["step_phases"]
    forward_events = buffer["forward_events"]
    backward_events = buffer["backward_events"]
    phase_durations = buffer["phase_durations"]
    horovod = buffer["horovod"]

    for event in events:

        if timestamp_us != None and (
            event.start_time < timestamp_us and event.end_time > timestamp_us
        ):
            # CPU functions
            if "cpu" in event.event_phase.lower():
                if event.event_phase not in cpu_events:
                    cpu_events[event.event_phase] = {}
                if event.event_name not in cpu_events[event.event_phase]:
                    cpu_events[event.event_phase][event.event_name] = 0
                cpu_events[event.event_phase][event.event_name] += event.end_time - event.start_time

            # GPU functions
            elif "gpu" in event.event_phase.lower():
                if event.event_phase not in gpu_events:
                    gpu_events[event.event_phase] = {}
                if event.event_name not in gpu_events[event.event_phase]:
                    gpu_events[event.event_phase][event.event_name] = 0
                gpu_events[event.event_phase][event.event_name] += event.end_time - event.start_time

            # ratio between TRAIN/EVAL and others
            elif "Step" in event.event_phase:
                if event.event_phase not in step_phases:
                    step_phases[event.event_phase] = 0
                step_phases[event.event_phase] += event.end_time - event.start_time

            # forward (only PT)
            elif event.event_phase == "Forward-SubModuleInternal":
                if event.event_name not in forward_events:
                    forward_events[event.event_name] = 0
                forward_events[event.event_name] += event.end_time - event.start_time

            # backward (only PT)
            elif event.event_phase == "Backward-SubModuleInternal":
                if event.event_name not in backward_events:
                    backward_events[event.event_name] = 0
                backward_events[event.event_name] += event.end_time - event.start_time

            # annotated events from backend (dataloader, dataiter, NCCLManager)
            elif (
                ":CPU" not in event.event_phase
                and ":GPU" not in event.event_phase
                and event.file_type != ""
            ):
                if event.event_phase not in phase_durations:
                    phase_durations[event.event_phase] = 0
                phase_durations[event.event_phase] += event.duration

            # Horovod events (source/file_type is an empty string)
            elif event.file_type == "":
                if event.event_name not in horovod:
                    horovod[event.event_name] = 0
                horovod[event.event_name] += event.end_time - event.start_time

    # convert values to percentages and filter small events
    labels_cpu = [key for key in cpu_events]
    labels_gpu = [key for key in gpu_events]
    labels = labels_cpu
    labels.extend(labels_gpu)
    totals_gpu = [sum(gpu_events[key].values()) for key in gpu_events]
    totals_cpu = [sum(cpu_events[key].values()) for key in cpu_events]
    totals = totals_gpu
    totals.extend(totals_cpu)

    sizes = [size / sum(totals) * 100.0 for size in totals]
    report["Details"]["ratio"] = {}

    # record information for profiler report
    for label, size in zip(labels, sizes):
        report["Details"]["ratio"][label] = size

    # create chart for detailed CPU functions
    if len(cpu_events) > 0:

        # convert values to percentages and filter small events
        labels, sizes, times = filter_events(cpu_events, threshold=2)

        # record information for profiler report
        report["Details"]["CPU"] = {}
        report["Details"]["CPU_total"] = {}
        for label, size, time in zip(labels, sizes, times):
            report["Details"]["CPU"][label] = size
            report["Details"]["CPU_total"][label] = time

    # record detailed GPU functions
    if len(gpu_events) > 0:

        # convert values to percentages and filter small events
        labels, sizes, times = filter_events(gpu_events, threshold=2)

        # record information for profiler report
        report["Details"]["GPU"] = {}
        report["Details"]["GPU_total"] = {}
        for label, size, time in zip(labels, sizes, times):
            report["Details"]["GPU"][label] = size
            report["Details"]["GPU_total"][label] = time

    # record train/eval phase and others
    labels = step_phases.keys()
    sizes = [float(i) / sum(step_phases.values()) * 100 for i in step_phases.values()]

    # record information for profiler report
    report["Details"]["phase"] = {}
    report["Details"]["phase_time"] = {}
    for label, size in zip(labels, sizes):
        report["Details"]["phase"][label] = size

    # breakdown for generic framework metrics
    if len(phase_durations) > 0:

        labels = list(phase_durations.keys())
        sizes = [float(i) / sum(phase_durations.values()) * 100 for i in phase_durations.values()]

        # record information for profiler report
        report["Details"]["general"] = {}
        for label, size in zip(labels, sizes):
            report["Details"]["general"][label] = size

    # breakdown for forward/backward passes (only PT)
    if len(forward_events) > 0 and len(backward_events) > 0:

        totals = [sum(forward_events.values())]
        totals.append(sum(backward_events.values()))

        labels = ["Forward pass", "Backward pass"]
        sizes = [float(i) / sum(totals) * 100 for i in totals]

        # record information for profiler report
        report["Details"]["forward_backward"] = {}
        for label, size in zip(labels, sizes):
            report["Details"]["forward_backward"][label] = size

    # breakdown of Horovod events
    if len(horovod) > 0:

        # filter out small events
        filtered_events = {}
        total = sum(horovod.values())
        for event in horovod:
            if horovod[event] > 0 and horovod[event] / total > 0.02:
                filtered_events[event] = horovod[event]

        labels = list(filtered_events.keys())
        sizes = [float(i) / sum(filtered_events.values()) * 100 for i in filtered_events.values()]

        # record information for profiler report
        report["Details"]["horovod"] = {}
        for label, size in zip(labels, sizes):
            report["Details"]["horovod"][label] = size
