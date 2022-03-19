'''
        gpus_droplist = []
        cpus_droplist = []
        gpus_addlist = []
        cpus_addlist = []

        pipelinestageindex = len (self.pipelinestages) - 1

        while pipelinestageindex >= 0:
            pipelinestage = self.pipelinestages[pipelinestageindex]
            input_pressure = computation_pressures[str(pipelinestageindex)][0]
            pressure_difference = max_throughputs[str(pipelinestageindex)] - input_pressure
            throughput = throughputs[str(pipelinestageindex)]
            max_throughput = max_throughputs[str(pipelinestageindex)]
            underutilization = underutilizations[str(pipelinestageindex)]

            if underutilization > 0:
                if input_pressure > 0:
                    if throughput <= 0:
                        print('reconfiguration 0()', 'startup delay', pipelinestage.name, underutilization, input_pressure, pressure_difference, throughput, max_throughput, pipelinestage.phases[0].current_count)
                    if pressure_difference > 0:
                        print('reconfiguration 1()', 'faster pipelinestage', pipelinestage.name, underutilization, input_pressure, pressure_difference, throughput, max_throughput, pipelinestage.phases[0].current_count)
                    else:
                        print('reconfiguration 2()', 'slower pipelinestage', pipelinestage.name, underutilization, input_pressure, pressure_difference, throughput, max_throughput, pipelinestage.phases[0].current_count)
                else:
                    if throughput <= 0:
                        print('reconfiguration 3()', 'empty stage', pipelinestage.name, underutilization, input_pressure, pressure_difference, throughput, max_throughput, pipelinestage.phases[0].current_count)
                    else:
                        print('reconfiguration 4()', 'tail stage', pipelinestage.name, underutilization, input_pressure, pressure_difference, throughput, max_throughput, pipelinestage.phases[0].current_count)
            else:
                if input_pressure > 0:
                    if throughput <= 0:
                        print('reconfiguration 5()', 'empty stage', pipelinestage.name, underutilization, input_pressure, pressure_difference, throughput, max_throughput, pipelinestage.phases[0].current_count)
                    else:
                        if pressure_difference > 0:
                            print('reconfiguration 6()', 'faster pipelinestage', pipelinestage.name, underutilization, input_pressure, pressure_difference, throughput, max_throughput, pipelinestage.phases[0].current_count)
                        else:
                            print('reconfiguration 7()', 'slower pipelinestage', pipelinestage.name, underutilization, input_pressure, pressure_difference, throughput, max_throughput, pipelinestage.phases[0].current_count)
                else:
                    if throughput <= 0:
                        print ('reconfiguration 8()', 'empty stage', pipelinestage.name, underutilization, input_pressure, pressure_difference, throughput, max_throughput, pipelinestage.phases[0].current_count)
                    else:
                        print('reconfiguration 9()', 'tail stage', pipelinestage.name, underutilization, input_pressure, pressure_difference, throughput, max_throughput, pipelinestage.phases[0].current_count)


            pipelinestageindex -= 1

        if gpu_incoming_traffic <= 0 and total_gpu_throughput == 0:
            #calculate how long will this last ?
            time_to_last = 0
            for pipelinestageindex in pending_cpuworkloads.keys ():
                if cpu_throughputs[pipelinestageindex] == 0:
                    continue
                time_to_last += pending_cpuworkloads[pipelinestageindex] / cpu_throughputs[pipelinestageindex]
            print ('zero gpu_incoming_traffic to last', time_to_last)

            gpu_weighted_pcr_ranking = self.get_weighted_performance_to_cost_ratio_ranking (rmanager, 'GPU')

            to_be_dropped = []

            for gpu_id in gpu_weighted_pcr_ranking.keys ():
                gpu = rmanager.get_resource (gpu_id)
                if rmanager.get_availability (gpu.gpu.name) > 0.8 and rmanager.get_startup_time (gpu.gpu.name) < time_to_last:
                    to_be_dropped.append (gpu.id)
            print ('GPUs to be dropped', to_be_dropped)
            return to_be_dropped
        else:
            pass

        if cpu_incoming_traffic <= 0 and total_cpu_throughput == 0:
            # calculate how long will this last ?
            time_to_last = 0
            for pipelinestageindex in pending_gpuworkloads.keys():
                if gpu_throughputs[pipelinestageindex] == 0:
                    continue
                time_to_last += pending_gpuworkloads[pipelinestageindex] / gpu_throughputs[pipelinestageindex]
            print('zero gpu_incoming_traffic to last', time_to_last)

            cpu_weighted_pcr_ranking = self.get_weighted_performance_to_cost_ratio_ranking(rmanager, 'CPU')

            to_be_dropped = []

            for cpu_id in cpu_weighted_pcr_ranking.keys():
                cpu = rmanager.get_resource(cpu_id)
                if rmanager.get_availability (cpu.cpu.name) > 0.8 and rmanager.get_startup_time (cpu.cpu.name) < time_to_last:
                    to_be_dropped.append (cpu.id)
            print('CPUs to be dropped', to_be_dropped)
            return to_be_dropped
        else:
            pass
        '''