def scale_down_configuration(self, rmanager, pipelinestageindex, overallocation, throughput, available_resources):
    to_be_deleted = []
    target_throughput = float((1 - overallocation)) * throughput
    pipelinestage = self.pipelinestages[pipelinestageindex]

    while True:
        weighted_pcr_ranking = self.get_weighted_performance_to_cost_ratio_ranking(rmanager, pipelinestage.resourcetype,
                                                                                   available_resources)

        print('scale_down_configuration', weighted_pcr_ranking)

        removed_at_least_one = False
        for resource_id in weighted_pcr_ranking.keys():
            resource = rmanager.get_resource(resource_id, active=True)
            if pipelinestage.resourcetype == 'CPU':
                resource_name = resource.cpu.name
            else:
                resource_name = resource.gpu.name
            exectime = resource.get_exectime(pipelinestage.name, pipelinestage.resourcetype)
            if exectime == 0:
                exectime = rmanager.get_exectime(resource_name, pipelinestage.name)
                resource_throughput = 1 / exectime
            else:
                resource_throughput = 1 / exectime

            throughput = throughput - resource_throughput

            print('scale_down_configuration', resource_id, resource_name, resource_throughput, throughput,
                  target_throughput)

            if throughput - target_throughput >= 0:
                to_be_deleted.append(resource_id)
                available_resources.remove(resource_id)
                removed_at_least_one = True
                break
            else:
                throughput += resource_throughput

        if removed_at_least_one == False:
            break
    return to_be_deleted


def scale_up_configuration_limit(self, rmanager, pipelinestageindex, input_pressure, throughput, resource_limit,
                                 throughput_limit):
    to_be_added = {}
    target_throughput = input_pressure - throughput
    if throughput_limit != 0:
        target_throughput = throughput_limit - throughput
    pipelinestage = self.pipelinestages[pipelinestageindex]
    added_throughput = 0
    total_acquired = 0

    if target_throughput <= 0:
        return to_be_added

    while True:
        weighted_pcr_ranking = self.get_weighted_performance_to_cost_ratio_ranking_all(rmanager,
                                                                                       pipelinestage.resourcetype)

        print('scale_up_configuration_limit', weighted_pcr_ranking)

        for resource_name in weighted_pcr_ranking.keys():
            exectime = rmanager.get_exectime(resource_name, pipelinestage.name)

            if exectime == 0:
                exectime = rmanager.get_exectime(resource_name, pipelinestage.name)
                resource_throughput = 1 / exectime
            else:
                resource_throughput = 1 / exectime

            resource_available = rmanager.request_resource(resource_name)

            print('scale_up_configuration_limit', resource_name, resource_throughput, resource_available, throughput,
                  target_throughput, resource_limit)

            if resource_available == True:
                if added_throughput + resource_throughput > added_throughput:
                    if resource_name not in to_be_added:
                        to_be_added[resource_name] = 1
                        added_throughput = added_throughput + resource_throughput
                    else:
                        to_be_added[resource_name] += 1
                        added_throughput = added_throughput + resource_throughput
                    total_acquired += 1
                    break

        if total_acquired >= resource_limit:
            break

        if added_throughput >= target_throughput:
            break
    return to_be_added


def calculate_pipeline_stats(self, rmanager, current_time, free_cpus, free_gpus):
    throughputs = {}
    pending_workloads = {}
    computation_pressures = {}
    available_resources = {}
    max_throughputs = {}
    upcoming_throughputs = {}

    total_cpu_input_pressure = 0
    total_gpu_input_pressure = 0
    total_cpu_throughput = 0
    total_gpu_throughput = 0

    pipelinestageindex = 0
    current_free_cpus = copy.deepcopy(free_cpus)
    current_free_gpus = copy.deepcopy(free_gpus)

    while pipelinestageindex < len(self.pipelinestages):
        pipelinestage = self.pipelinestages[pipelinestageindex]

        if pipelinestage.resourcetype == 'CPU':
            free_resources = current_free_cpus
        else:
            free_resources = current_free_gpus

        available_resources[str(pipelinestageindex)] = copy.deepcopy(pipelinestage.phases[0].current_executors)

        for free_resource_id in free_resources:
            free_resource = rmanager.get_resource(free_resource_id, True)

            if free_resource.active == False:
                if free_resource.temporary_assignment == str(pipelinestageindex):
                    available_resources[str(pipelinestageindex)].append(free_resource_id)
                    free_resources.remove(free_resource_id)

        throughputs[str(pipelinestageindex)] = pipelinestage.get_current_throughput(0)

        if pipelinestageindex != 0:
            if pipelinestage.resourcetype == 'CPU':
                pending_workloads[str(pipelinestageindex)] = pipelinestage.phases[0].current_count - len(
                    pipelinestage.phases[0].current_executors) + pipelinestage.phases[0].get_queued_work(rmanager,
                                                                                                         'CPU',
                                                                                                         current_time)
            else:
                pending_workloads[str(pipelinestageindex)] = pipelinestage.phases[0].current_count - len(
                    pipelinestage.phases[0].current_executors) + pipelinestage.phases[0].get_queued_work(rmanager,
                                                                                                         'GPU',
                                                                                                         current_time)

        if pipelinestage.resourcetype == 'CPU':
            if pipelinestageindex < len(self.pipelinestages) - 1:
                total_gpu_input_pressure += throughputs[str(pipelinestageindex)]
        else:
            if pipelinestageindex < len(self.pipelinestages) - 1:
                total_cpu_input_pressure += throughputs[str(pipelinestageindex)]

        if pipelinestage.resourcetype == 'CPU':
            total_cpu_throughput += throughputs[str(pipelinestageindex)]
        else:
            total_gpu_throughput += throughputs[str(pipelinestageindex)]

        if pipelinestageindex == 0:
            computation_pressures[str(pipelinestageindex)] = [0, throughputs[str(pipelinestageindex)]]
        else:
            prev_pipelinestage = self.pipelinestages[pipelinestageindex - 1]
            throughputs[str(pipelinestageindex - 1)] = prev_pipelinestage.get_current_throughput(0)
            computation_pressures[str(pipelinestageindex)] = [throughputs[str(pipelinestageindex - 1)],
                                                              throughputs[str(pipelinestageindex)]]

        pipelinestageindex += 1

    pipelinestageindex = len(self.pipelinestages) - 1

    while pipelinestageindex >= 0:

        pipelinestage = self.pipelinestages[pipelinestageindex]

        if pipelinestageindex - 2 >= 0:
            prev_sametype_pipelinestage = self.pipelinestages[pipelinestageindex - 2]
        else:
            prev_sametype_pipelinestage = None

        if pipelinestage.resourcetype == 'CPU':
            free_resources = current_free_cpus
        else:
            free_resources = current_free_gpus

        if throughputs[str(pipelinestageindex)] <= 0:
            if computation_pressures[str(pipelinestageindex)][0] <= 0:
                # available_resources[str(pipelinestageindex)] = []
                max_throughputs[str(pipelinestageindex)] = 0
            else:
                available_resources[str(pipelinestageindex)].extend(copy.deepcopy(free_resources))
                max_throughputs[str(pipelinestageindex)] = pipelinestage.get_free_resource_throughput(rmanager,
                                                                                                      available_resources[
                                                                                                          str(
                                                                                                              pipelinestageindex)])
                free_resources.clear()
        else:
            available_resources[str(pipelinestageindex)].extend(free_resources)
            max_throughputs[str(pipelinestageindex)] = pipelinestage.get_free_resource_throughput(rmanager,
                                                                                                  available_resources[
                                                                                                      str(
                                                                                                          pipelinestageindex)])
            free_resources.clear()

        if prev_sametype_pipelinestage != None:
            upcoming_throughputs[str(pipelinestageindex)] = pipelinestage.get_free_resource_throughput(rmanager,
                                                                                                       available_resources[
                                                                                                           str(
                                                                                                               pipelinestageindex - 2)])
            upcoming_throughputs[str(pipelinestageindex)] += max_throughputs[str(pipelinestageindex)]
        else:
            upcoming_throughputs[str(pipelinestageindex)] = max_throughputs[str(pipelinestageindex)]

        pipelinestageindex -= 1

    return throughputs, pending_workloads, computation_pressures, available_resources, max_throughputs, total_cpu_input_pressure, total_gpu_input_pressure, total_cpu_throughput, total_gpu_throughput, upcoming_throughputs


def reconfiguration_up_down_underallocations(self, rmanager, current_time, free_cpus, free_gpus, imbalance_limit,
                                             throughput_target):
    print('reconfiguration_up_down_underallocations ()', free_cpus, free_gpus)

    throughputs, pending_workloads, computation_pressures, available_resources, max_throughputs, total_cpu_input_pressure, total_gpu_input_pressure, total_cpu_throughput, total_gpu_throughput, upcoming_throughputs = self.calculate_pipeline_stats(
        rmanager, current_time, free_cpus, free_gpus)

    underallocations = {}
    underallocations[str(0)] = 0.0

    gpus_to_be_added = {}
    cpus_to_be_added = {}

    pipelinestageindex = 1
    while pipelinestageindex < len(self.pipelinestages):
        pipelinestage = self.pipelinestages[pipelinestageindex]

        if pipelinestage.resourcetype == 'CPU':
            total_throughput = total_cpu_throughput
        else:
            total_throughput = total_gpu_throughput

        if max_throughputs[str(pipelinestageindex)] == 0:
            if throughputs[str(pipelinestageindex - 1)] > 0:
                underallocations[str(pipelinestageindex)] = 1.0
            else:
                underallocations[str(pipelinestageindex)] = 0.0
        elif throughputs[str(str(pipelinestageindex))] < max_throughputs[str(pipelinestageindex)]:
            underallocations[str(pipelinestageindex)] = 0.0
        else:
            if computation_pressures[str(pipelinestageindex)][0] > 0:
                underallocations[str(pipelinestageindex)] = (computation_pressures[str(pipelinestageindex)][0] -
                                                             max_throughputs[str(pipelinestageindex)]) / \
                                                            computation_pressures[str(pipelinestageindex)][0]
            else:
                underallocations[str(pipelinestageindex)] = 0.0

        pending_workitems = pipelinestage.phases[0].current_count - len(available_resources[str(pipelinestageindex)])

        throughput_limit = 0

        if pipelinestageindex < len(self.pipelinestages) - 1:
            next_pipelinestage_upcoming_throughput = upcoming_throughputs[str(pipelinestageindex + 1)]

            prev_pipelinestage_throughput = computation_pressures[str(pipelinestageindex)][0]

            if next_pipelinestage_upcoming_throughput < prev_pipelinestage_throughput:
                throughput_limit = next_pipelinestage_upcoming_throughput
            else:
                throughput_limit = prev_pipelinestage_throughput

        if str(pipelinestageindex) in underallocations and pending_workitems > 0 and (
                (underallocations[str(pipelinestageindex)] >= 1.0 and total_throughput <= 0) or underallocations[
            str(pipelinestageindex)] > 0 and throughputs[str(pipelinestageindex)] == total_throughput):
            print('reconfiguration_up_down_underallocations 1 ()', pipelinestage.name, underallocations,
                  computation_pressures[str(pipelinestageindex)], max_throughputs[str(pipelinestageindex)],
                  available_resources, pending_workloads)

            to_be_added = self.scale_up_configuration_limit(rmanager, pipelinestageindex,
                                                            computation_pressures[str(pipelinestageindex)][0],
                                                            max_throughputs[str(pipelinestageindex)], pending_workitems,
                                                            throughput_limit)

            # to_be_added = self.scale_up_configuration_limit_imbalance_limit(rmanager, pipelinestageindex,
            #                                                computation_pressures[str(pipelinestageindex)][0],
            #                                                max_throughputs[str(pipelinestageindex)],
            #                                                pending_workitems, imbalance_limit)
            if pipelinestage.resourcetype == 'CPU':
                print('CPUs to be added', to_be_added)
                cpus_to_be_added[str(pipelinestageindex)] = to_be_added
            else:
                print('GPUs to be added', to_be_added)
                gpus_to_be_added[str(pipelinestageindex)] = to_be_added
        elif str(pipelinestageindex) in underallocations and pending_workitems > 0 and (
                underallocations[str(pipelinestageindex)] <= 0.0 and total_throughput <= 0):
            print('reconfiguration_up_down_underallocations 2 ()', pipelinestage.name, underallocations,
                  computation_pressures[str(pipelinestageindex)], max_throughputs[str(pipelinestageindex)],
                  available_resources, pending_workloads)
            to_be_added = self.scale_up_configuration_limit(rmanager, pipelinestageindex,
                                                            computation_pressures[str(pipelinestageindex)][0],
                                                            max_throughputs[str(pipelinestageindex)],
                                                            pending_workitems, throughput_limit)

            # to_be_added = self.scale_up_configuration_limit_imbalance_limit(rmanager, pipelinestageindex,
            #                                                computation_pressures[str(pipelinestageindex)][0],
            #                                                max_throughputs[str(pipelinestageindex)],
            #                                                pending_workitems, imbalance_limit)
            if pipelinestage.resourcetype == 'CPU':
                print('CPUs to be added', to_be_added)
                cpus_to_be_added[str(pipelinestageindex)] = to_be_added
            else:
                print('GPUs to be added', to_be_added)
                gpus_to_be_added[str(pipelinestageindex)] = to_be_added

        pipelinestageindex += 1

    return cpus_to_be_added, gpus_to_be_added


def reconfiguration_up_down_overallocations(self, rmanager, current_time, free_cpus, free_gpus, imbalance_limit,
                                            throughput_target):
    print('reconfiguration_up_down_overallocations ()', free_cpus, free_gpus)

    throughputs, pending_workloads, computation_pressures, available_resources, max_throughputs, total_cpu_input_pressure, total_gpu_input_pressure, total_cpu_throughput, total_gpu_throughput, upcoming_throughputs = self.calculate_pipeline_stats(
        rmanager, current_time, free_cpus, free_gpus)

    overallocations = {}

    overallocations[str(0)] = 0.0

    gpus_to_be_dropped = []
    cpus_to_be_dropped = []

    pipelinestageindex = 1
    while pipelinestageindex < len(self.pipelinestages):
        pipelinestage = self.pipelinestages[pipelinestageindex]

        if pipelinestage.resourcetype == 'CPU':
            total_throughput = total_cpu_throughput
        else:
            total_throughput = total_gpu_throughput

        if max_throughputs[str(pipelinestageindex)] == 0:
            overallocations[str(pipelinestageindex)] = 0.0
        elif throughputs[str(str(pipelinestageindex))] < max_throughputs[str(pipelinestageindex)]:
            if computation_pressures[str(pipelinestageindex)][0] > 0:
                overallocations[str(pipelinestageindex)] = (max_throughputs[str(pipelinestageindex)] -
                                                            computation_pressures[str(pipelinestageindex)][0]) / \
                                                           max_throughputs[str(pipelinestageindex)]
            else:
                overallocations[str(pipelinestageindex)] = (max_throughputs[str(pipelinestageindex)] - throughputs[
                    str(pipelinestageindex)]) / max_throughputs[str(pipelinestageindex)]
        else:
            overallocations[str(pipelinestageindex)] = 0.0

        if str(pipelinestageindex) in overallocations and overallocations[str(pipelinestageindex)] > 0:
            print('reconfiguration ()', pipelinestage.name, overallocations[str(pipelinestageindex)],
                  computation_pressures[str(pipelinestageindex)], max_throughputs[str(pipelinestageindex)],
                  available_resources)
            to_be_dropped = self.scale_down_configuration(rmanager, pipelinestageindex,
                                                          overallocations[str(pipelinestageindex)],
                                                          max_throughputs[str(pipelinestageindex)],
                                                          available_resources[str(pipelinestageindex)])

            # to_be_dropped = self.scale_down_configuration_imbalance_limit(rmanager, pipelinestageindex,
            #                                              overallocations[str(pipelinestageindex)],
            #                                              max_throughputs[str(pipelinestageindex)],
            #                                              available_resources[str(pipelinestageindex)],
            #                                                              imbalance_limit)

            if pipelinestage.resourcetype == 'CPU':
                print('CPUs to be dropped', to_be_dropped)
                cpus_to_be_dropped.extend(to_be_dropped)
            else:
                print('GPUs to be dropped', to_be_dropped)
                gpus_to_be_dropped.extend(to_be_dropped)

        pipelinestageindex += 1

    return cpus_to_be_dropped, gpus_to_be_dropped


def reconfiguration_drop(self, rmanager, current_time, free_cpus, free_gpus, imbalance_limit, throughput_target):
    gpus_to_be_dropped = []
    cpus_to_be_dropped = []

    print('reconfiguration_drop ()', free_cpus, free_gpus)

    throughputs, pending_workloads, computation_pressures, available_resources, max_throughputs, total_cpu_input_pressure, total_gpu_input_pressure, total_cpu_throughput, total_gpu_throughput, upcoming_throughputs = self.calculate_pipeline_stats(
        rmanager, current_time, free_cpus, free_gpus)

    if total_gpu_throughput <= 0 and total_gpu_input_pressure <= 0:
        gpu_weighted_pcr_ranking = self.get_weighted_performance_to_cost_ratio_ranking(rmanager, 'GPU', free_gpus)

        to_be_dropped = []

        for gpu_id in gpu_weighted_pcr_ranking.keys():
            gpu = rmanager.get_resource(gpu_id, active=True)
            to_be_dropped.append(gpu.id)
        print('GPUs to be dropped', to_be_dropped)
        gpus_to_be_dropped.extend(to_be_dropped)

    if total_cpu_throughput <= 0 and total_cpu_input_pressure <= 0:
        cpu_weighted_pcr_ranking = self.get_weighted_performance_to_cost_ratio_ranking(rmanager, 'CPU', free_cpus)

        to_be_dropped = []

        for cpu_id in cpu_weighted_pcr_ranking.keys():
            cpu = rmanager.get_resource(cpu_id, active=True)
            to_be_dropped.append(cpu.id)
        print('CPUs to be dropped', to_be_dropped)
        cpus_to_be_dropped.extend(to_be_dropped)

    print('total throughput', total_cpu_throughput, total_gpu_throughput)
    print('total input pressure', total_cpu_input_pressure, total_gpu_input_pressure)
    print('computation pressures', computation_pressures)
    print('max throughputs', max_throughputs)
    print('available resources', available_resources)
    print('pending_workloads ', pending_workloads)

    return cpus_to_be_dropped, gpus_to_be_dropped
