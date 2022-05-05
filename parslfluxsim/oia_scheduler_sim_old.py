import time
from parslfluxsim.FirstCompleteFirstServe_sim import FirstCompleteFirstServe
from parslfluxsim.resources_sim import ResourceManager
from parslflux.pipeline import PipelineManager
from parslfluxsim.input_sim import InputManager2
from execution_sim import ExecutionSim, ExecutionSimThread
from aws_cloud_sim.arc_to_aws_mapping import ARCMapping
import simpy
from performance_analysis import add_performance_data


class OAI_Scheduler:
    def __init__(self, env):
        self.env = env
        self.idle_periods = {}
        self.cpuallocations = {}
        self.gpuallocations = {}
        self.worker_threads = {}
        self.workers = {}

    def add_worker(self, rmanager, cpuok, gpuok, cputype, gputype, provision_type, bidding_price, pipelinestageindex):
        if provision_type == 'on_demand':
            activepool = True
        else:
            activepool = False

        new_resource, provision_time = rmanager.add_resource(cpuok, gpuok, cputype, gputype, provision_type, activepool,
                                                             bidding_price, pipelinestageindex)

        '''
        if cpuok == True:
            new_resource.set_idle_start_time ('CPU', self.env.now)
        else:
            new_resource.set_idle_start_time ('GPU', self.env.now)
        '''

        print(new_resource.cpu, new_resource.gpu)

        if new_resource.cpu != None:
            cpu_thread = ExecutionSimThread(self.env, new_resource, 'CPU', self.performancedata, provision_type,
                                            provision_time)
        else:
            cpu_thread = None

        if new_resource.gpu != None:
            gpu_thread = ExecutionSimThread(self.env, new_resource, 'GPU', self.performancedata, provision_type,
                                            provision_time)
        else:
            gpu_thread = None

        self.worker_threads[new_resource.id] = [cpu_thread, gpu_thread]

        if cpu_thread != None:
            cpu_thread_exec = ExecutionSim(self.env, cpu_thread)
        else:
            cpu_thread_exec = None
        if gpu_thread != None:
            gpu_thread_exec = ExecutionSim(self.env, gpu_thread)
        else:
            gpu_thread_exec = None

        self.workers[new_resource.id] = [cpu_thread_exec, gpu_thread_exec]

    def delete_worker(self, rmanager, resourcetype, resource_id):
        if resourcetype == 'CPU':
            worker = self.workers[resource_id][0]
        else:
            worker = self.workers[resource_id][1]

        worker_exec = worker.get_exec()

        worker_exec.interrupt('cancel')

        self.worker_threads.pop(resource_id, None)
        self.workers.pop(resource_id, None)

        rmanager.delete_resource(resourcetype, resource_id, active=True)

    def scale_up_algo_0(self, rmanager, pmanager, phase_tracker):
        if phase_tracker == -1:
            return
        current_resources = rmanager.get_resources()
        cpu_idle_periods, gpu_idle_periods = pmanager.prediction_idle_periods[str(phase_tracker + 1)]

        print(self.env.now, phase_tracker, cpu_idle_periods)
        print(self.env.now, phase_tracker, gpu_idle_periods)

        empty_cpus = []
        empty_gpus = []

        for resource in current_resources:
            # print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            cpu_empty, gpu_empty = resource.is_empty()
            # print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

            if cpu_empty == True:
                empty_cpus.append(resource)
            if gpu_empty == True:
                empty_gpus.append(resource)

        for empty_cpu in empty_cpus:
            for cpu_idle_period in cpu_idle_periods:
                if empty_cpu.id in cpu_idle_period[2].keys():
                    idle_periods = cpu_idle_period[2][empty_cpu.id]
                    idle_period_index = 0
                    for idle_period in idle_periods:
                        if idle_period[2] > 0:
                            if self.env.now >= idle_period[0]:
                                print(empty_cpu.id, idle_period)
                                self.delete_worker(rmanager, 'CPU', empty_cpu.id)
                                if empty_cpu.cpu.name not in self.cpuallocations:
                                    self.cpuallocations[empty_cpu.cpu.name] = []
                                    self.cpuallocations[empty_cpu.cpu.name].append(idle_period[1])
                            break
                        idle_period_index += 1

                    if idle_period_index < len(idle_periods):
                        idle_periods.pop(idle_period_index)

        for empty_gpu in empty_gpus:
            for gpu_idle_period in gpu_idle_periods:
                if empty_gpu.id in gpu_idle_period[2].keys():
                    idle_periods = gpu_idle_period[2][empty_gpu.id]
                    idle_period_index = 0
                    for idle_period in idle_periods:
                        if idle_period[2] > 0:
                            if self.env.now >= idle_period[0]:
                                print(empty_gpu.id, idle_period)
                                self.delete_worker(rmanager, 'GPU', empty_gpu.id)
                                if empty_gpu.gpu.name not in self.gpuallocations:
                                    self.gpuallocations[empty_gpu.gpu.name] = []
                                    self.gpuallocations[empty_gpu.gpu.name].append(idle_period[1])
                            break
                        idle_period_index += 1

                    if idle_period_index < len(idle_periods):
                        idle_periods.pop(idle_period_index)

        for cputype in self.cpuallocations.keys():
            allocation_timeline = self.cpuallocations[cputype]
            if len(allocation_timeline) > 0 and self.env.now >= allocation_timeline[0]:
                self.add_worker(rmanager, True, False, cputype, None)
                allocation_timeline.pop(0)

        for gputype in self.gpuallocations.keys():
            allocation_timeline = self.gpuallocations[gputype]
            if len(allocation_timeline) > 0 and self.env.now >= allocation_timeline[0]:
                self.add_worker(rmanager, False, True, None, gputype)
                allocation_timeline.pop(0)

    def convert_resource_idle_periods(self, rmanager, pmanager, phase_tracker):
        cpu_idle_periods, gpu_idle_periods = pmanager.prediction_idle_periods[str(phase_tracker + 1)]

        i_cpu_idle_periods = {}
        i_gpu_idle_periods = {}

        for cpu_idle_period in cpu_idle_periods:
            resource_idle_periods = cpu_idle_period[2]
            for resource_key in resource_idle_periods.keys():
                resource = rmanager.get_resource(resource_key)
                idle_periods = resource_idle_periods[resource_key]
                for idle_period in idle_periods:
                    if idle_period[2] > 0:
                        if resource_key not in i_cpu_idle_periods.keys():
                            i_cpu_idle_periods[resource_key] = []
                            i_cpu_idle_periods[resource_key].append([idle_period[0], idle_period[1]])
                        else:
                            i_cpu_idle_periods[resource_key].append([idle_period[0], idle_period[1]])

        for gpu_idle_period in gpu_idle_periods:
            resource_idle_periods = gpu_idle_period[2]
            for resource_key in resource_idle_periods.keys():
                resource = rmanager.get_resource(resource_key)
                idle_periods = resource_idle_periods[resource_key]
                for idle_period in idle_periods:
                    if idle_period[2] > 0:
                        if resource_key not in i_gpu_idle_periods.keys():
                            i_gpu_idle_periods[resource_key] = []
                            i_gpu_idle_periods[resource_key].append([idle_period[0], idle_period[1]])
                        else:
                            i_gpu_idle_periods[resource_key].append([idle_period[0], idle_period[1]])

        new_cpu_idle_periods = {}
        new_gpu_idle_periods = {}

        for cpu_id in i_cpu_idle_periods.keys():
            cpu_idle_periods = i_cpu_idle_periods[cpu_id]
            resource = rmanager.get_resource(cpu_id)
            resource_startup_time = rmanager.get_startup_time(resource.cpu.name)
            idle_period_list = []
            for idle_period in cpu_idle_periods:
                if len(idle_period_list) <= 0:
                    idle_period_list.append(idle_period)
                else:
                    if idle_period[0] - idle_period_list[-1][1] <= resource_startup_time:
                        idle_period_list[-1][1] = idle_period[1]
                    else:
                        idle_period_list.append(idle_period)
            i_cpu_idle_periods[cpu_id] = idle_period_list

            if resource.cpu.name not in new_cpu_idle_periods.keys():
                new_cpu_idle_periods[resource.cpu.name] = []
                new_cpu_idle_periods[resource.cpu.name].extend(idle_period_list)
            else:
                new_cpu_idle_periods[resource.cpu.name].extend(idle_period_list)

        for gpu_id in i_gpu_idle_periods.keys():
            gpu_idle_periods = i_gpu_idle_periods[gpu_id]
            resource = rmanager.get_resource(gpu_id)
            resource_startup_time = rmanager.get_startup_time(resource.gpu.name)
            idle_period_list = []
            for idle_period in gpu_idle_periods:
                if len(idle_period_list) <= 0:
                    idle_period_list.append(idle_period)
                else:
                    if idle_period[0] - idle_period_list[-1][1] <= resource_startup_time:
                        idle_period_list[-1][1] = idle_period[1]
                    else:
                        idle_period_list.append(idle_period)
            i_gpu_idle_periods[gpu_id] = idle_period_list

            if resource.gpu.name not in new_gpu_idle_periods.keys():
                new_gpu_idle_periods[resource.gpu.name] = []
                new_gpu_idle_periods[resource.gpu.name].extend(idle_period_list)
            else:
                new_gpu_idle_periods[resource.gpu.name].extend(idle_period_list)

        print('after conversion CPU', new_cpu_idle_periods)
        print('after conversion GPU', new_gpu_idle_periods)

        pmanager.prediction_idle_periods[str(phase_tracker + 1)] = [new_cpu_idle_periods, new_gpu_idle_periods]

    def reconfiguration_no_prediction_up_down_underallocations_first(self, rmanager, pmanager, idle_cpus, idle_gpus):
        cpus_to_be_added, gpus_to_be_added = pmanager.reconfiguration_up_down_underallocations(rmanager, self.env.now,
                                                                                               idle_cpus, idle_gpus,
                                                                                               self.imbalance_limit,
                                                                                               self.throughput_target)

        final_cpus_to_be_dropped = {}
        final_gpus_to_be_dropped = {}

        cpus_to_be_dropped, gpus_to_be_dropped = pmanager.reconfiguration_up_down_overallocations(rmanager,
                                                                                                  self.env.now,
                                                                                                  idle_cpus, idle_gpus,
                                                                                                  self.imbalance_limit,
                                                                                                  self.throughput_target)

        for cpu_id in cpus_to_be_dropped:
            resource = rmanager.get_resource(cpu_id, True)
            if resource.cpu.name not in final_cpus_to_be_dropped:
                final_cpus_to_be_dropped[resource.cpu.name] = {}
                final_cpus_to_be_dropped[resource.cpu.name]['busy'] = []
                final_cpus_to_be_dropped[resource.cpu.name]['free'] = []
                if cpu_id not in idle_cpus:
                    final_cpus_to_be_dropped[resource.cpu.name]['busy'].append(cpu_id)
                else:
                    final_cpus_to_be_dropped[resource.cpu.name]['free'].append(cpu_id)

                if cpu_id in idle_cpus:
                    idle_cpus.remove(cpu_id)
            else:
                if cpu_id not in idle_cpus:
                    final_cpus_to_be_dropped[resource.cpu.name]['busy'].append(cpu_id)
                else:
                    final_cpus_to_be_dropped[resource.cpu.name]['free'].append(cpu_id)

                if cpu_id in idle_cpus:
                    idle_cpus.remove(cpu_id)

        for gpu_id in gpus_to_be_dropped:
            resource = rmanager.get_resource(gpu_id, True)
            if resource.gpu.name not in final_gpus_to_be_dropped:
                final_gpus_to_be_dropped[resource.gpu.name] = {}
                final_gpus_to_be_dropped[resource.gpu.name]['busy'] = []
                final_gpus_to_be_dropped[resource.gpu.name]['free'] = []
                if gpu_id not in idle_gpus:
                    final_gpus_to_be_dropped[resource.gpu.name]['busy'].append(gpu_id)
                else:
                    final_gpus_to_be_dropped[resource.gpu.name]['free'].append(gpu_id)

                if gpu_id in idle_gpus:
                    idle_gpus.remove(gpu_id)
            else:
                if gpu_id not in idle_gpus:
                    final_gpus_to_be_dropped[resource.gpu.name]['busy'].append(gpu_id)
                else:
                    final_gpus_to_be_dropped[resource.gpu.name]['free'].append(gpu_id)

                if gpu_id in idle_gpus:
                    idle_gpus.remove(gpu_id)

        cpus_to_be_dropped, gpus_to_be_dropped = pmanager.reconfiguration_drop(rmanager, self.env.now, idle_cpus,
                                                                               idle_gpus, self.imbalance_limit,
                                                                               self.throughput_target)

        for cpu_id in cpus_to_be_dropped:
            resource = rmanager.get_resource(cpu_id, True)
            if resource.cpu.name not in final_cpus_to_be_dropped:
                final_cpus_to_be_dropped[resource.cpu.name] = {}
                final_cpus_to_be_dropped[resource.cpu.name]['busy'] = []
                final_cpus_to_be_dropped[resource.cpu.name]['free'] = []
                if cpu_id not in idle_cpus:
                    final_cpus_to_be_dropped[resource.cpu.name]['busy'].append(cpu_id)
                else:
                    final_cpus_to_be_dropped[resource.cpu.name]['free'].append(cpu_id)
                idle_cpus.remove(cpu_id)
            else:
                if cpu_id not in idle_cpus:
                    final_cpus_to_be_dropped[resource.cpu.name]['busy'].append(cpu_id)
                else:
                    final_cpus_to_be_dropped[resource.cpu.name]['free'].append(cpu_id)
                idle_cpus.remove(cpu_id)

        for gpu_id in gpus_to_be_dropped:
            resource = rmanager.get_resource(gpu_id, True)
            if resource.gpu.name not in final_gpus_to_be_dropped:
                final_gpus_to_be_dropped[resource.gpu.name] = {}
                final_gpus_to_be_dropped[resource.gpu.name]['busy'] = []
                final_gpus_to_be_dropped[resource.gpu.name]['free'] = []
                if gpu_id not in idle_gpus:
                    final_gpus_to_be_dropped[resource.gpu.name]['busy'].append(gpu_id)
                else:
                    final_gpus_to_be_dropped[resource.gpu.name]['free'].append(gpu_id)
                idle_gpus.remove(gpu_id)
            else:
                if gpu_id not in idle_gpus:
                    final_gpus_to_be_dropped[resource.gpu.name]['busy'].append(gpu_id)
                else:
                    final_gpus_to_be_dropped[resource.gpu.name]['free'].append(gpu_id)
                idle_gpus.remove(gpu_id)

        for cpu_name in final_cpus_to_be_dropped:
            for cpu_id in final_cpus_to_be_dropped[cpu_name]['free']:
                self.delete_worker(rmanager, 'CPU', cpu_id)

        for gpu_name in final_gpus_to_be_dropped:
            for gpu_id in final_gpus_to_be_dropped[gpu_name]['free']:
                self.delete_worker(rmanager, 'GPU', gpu_id)

        for pipelinestageindex in cpus_to_be_added.keys():
            to_be_added = cpus_to_be_added[pipelinestageindex]
            for cpu_name in to_be_added.keys():
                count = to_be_added[cpu_name]
                for i in range(0, count):
                    self.add_worker(rmanager, True, False, cpu_name, None, 'on_demand', None, pipelinestageindex)

        for pipelinestageindex in gpus_to_be_added.keys():
            to_be_added = gpus_to_be_added[pipelinestageindex]
            for gpu_name in to_be_added.keys():
                count = to_be_added[gpu_name]
                for i in range(0, count):
                    self.add_worker(rmanager, False, True, None, gpu_name, 'on_demand', None, pipelinestageindex)

    def reconfiguration_no_prediction_up_down_overallocations_first(self, rmanager, pmanager, idle_cpus, idle_gpus):

        print('------------------------------------------------------')
        final_cpus_to_be_dropped = {}
        final_gpus_to_be_dropped = {}

        cpus_to_be_dropped, gpus_to_be_dropped = pmanager.reconfiguration_up_down_overallocations(rmanager,
                                                                                                  self.env.now,
                                                                                                  idle_cpus, idle_gpus,
                                                                                                  self.imbalance_limit,
                                                                                                  self.throughput_target)

        for cpu_id in cpus_to_be_dropped:
            resource = rmanager.get_resource(cpu_id, True)
            if resource.cpu.name not in final_cpus_to_be_dropped:
                final_cpus_to_be_dropped[resource.cpu.name] = {}
                final_cpus_to_be_dropped[resource.cpu.name]['busy'] = []
                final_cpus_to_be_dropped[resource.cpu.name]['free'] = []
                if cpu_id not in idle_cpus:
                    final_cpus_to_be_dropped[resource.cpu.name]['busy'].append(cpu_id)
                else:
                    final_cpus_to_be_dropped[resource.cpu.name]['free'].append(cpu_id)

                if cpu_id in idle_cpus:
                    idle_cpus.remove(cpu_id)
            else:
                if cpu_id not in idle_cpus:
                    final_cpus_to_be_dropped[resource.cpu.name]['busy'].append(cpu_id)
                else:
                    final_cpus_to_be_dropped[resource.cpu.name]['free'].append(cpu_id)

                if cpu_id in idle_cpus:
                    idle_cpus.remove(cpu_id)

        for gpu_id in gpus_to_be_dropped:
            resource = rmanager.get_resource(gpu_id, True)
            if resource.gpu.name not in final_gpus_to_be_dropped:
                final_gpus_to_be_dropped[resource.gpu.name] = {}
                final_gpus_to_be_dropped[resource.gpu.name]['busy'] = []
                final_gpus_to_be_dropped[resource.gpu.name]['free'] = []
                if gpu_id not in idle_gpus:
                    final_gpus_to_be_dropped[resource.gpu.name]['busy'].append(gpu_id)
                else:
                    final_gpus_to_be_dropped[resource.gpu.name]['free'].append(gpu_id)

                if gpu_id in idle_gpus:
                    idle_gpus.remove(gpu_id)
            else:
                if gpu_id not in idle_gpus:
                    final_gpus_to_be_dropped[resource.gpu.name]['busy'].append(gpu_id)
                else:
                    final_gpus_to_be_dropped[resource.gpu.name]['free'].append(gpu_id)

                if gpu_id in idle_gpus:
                    idle_gpus.remove(gpu_id)

        '''
        for cpu_id in cpus_to_be_dropped:
            if cpu_id not in idle_cpus:
                print ('CPU', cpu_id, 'not idle to be dropped')
                continue
            self.delete_worker(rmanager, 'CPU', cpu_id)
            idle_cpus.remove(cpu_id)
        for gpu_id in gpus_to_be_dropped:
            if gpu_id not in idle_gpus:
                print ('GPU', gpu_id, 'not idle to be dropped')
                continue
            self.delete_worker(rmanager, 'GPU', gpu_id)
            idle_gpus.remove(gpu_id)
        '''

        cpus_to_be_dropped, gpus_to_be_dropped = pmanager.reconfiguration_drop(rmanager, self.env.now, idle_cpus,
                                                                               idle_gpus, self.imbalance_limit,
                                                                               self.throughput_target)

        for cpu_id in cpus_to_be_dropped:
            resource = rmanager.get_resource(cpu_id, True)
            if resource.cpu.name not in final_cpus_to_be_dropped:
                final_cpus_to_be_dropped[resource.cpu.name] = {}
                final_cpus_to_be_dropped[resource.cpu.name]['busy'] = []
                final_cpus_to_be_dropped[resource.cpu.name]['free'] = []
                if cpu_id not in idle_cpus:
                    final_cpus_to_be_dropped[resource.cpu.name]['busy'].append(cpu_id)
                else:
                    final_cpus_to_be_dropped[resource.cpu.name]['free'].append(cpu_id)
                idle_cpus.remove(cpu_id)
            else:
                if cpu_id not in idle_cpus:
                    final_cpus_to_be_dropped[resource.cpu.name]['busy'].append(cpu_id)
                else:
                    final_cpus_to_be_dropped[resource.cpu.name]['free'].append(cpu_id)
                idle_cpus.remove(cpu_id)

        for gpu_id in gpus_to_be_dropped:
            resource = rmanager.get_resource(gpu_id, True)
            if resource.gpu.name not in final_gpus_to_be_dropped:
                final_gpus_to_be_dropped[resource.gpu.name] = {}
                final_gpus_to_be_dropped[resource.gpu.name]['busy'] = []
                final_gpus_to_be_dropped[resource.gpu.name]['free'] = []
                if gpu_id not in idle_gpus:
                    final_gpus_to_be_dropped[resource.gpu.name]['busy'].append(gpu_id)
                else:
                    final_gpus_to_be_dropped[resource.gpu.name]['free'].append(gpu_id)
                idle_gpus.remove(gpu_id)
            else:
                if gpu_id not in idle_gpus:
                    final_gpus_to_be_dropped[resource.gpu.name]['busy'].append(gpu_id)
                else:
                    final_gpus_to_be_dropped[resource.gpu.name]['free'].append(gpu_id)
                idle_gpus.remove(gpu_id)

        '''
        for cpu_id in cpus_to_be_dropped:
            if cpu_id not in idle_cpus:
                print ('CPU', cpu_id, 'not idle to be dropped')
                continue
            self.delete_worker(rmanager, 'CPU', cpu_id)
            idle_cpus.remove(cpu_id)
        for gpu_id in gpus_to_be_dropped:
            if gpu_id not in idle_gpus:
                print('GPU', gpu_id, 'not idle to be dropped')
                continue
            self.delete_worker(rmanager, 'GPU', gpu_id)
            idle_gpus.remove(gpu_id)

        '''

        cpus_to_be_added, gpus_to_be_added = pmanager.reconfiguration_up_down_underallocations(rmanager, self.env.now,
                                                                                               idle_cpus, idle_gpus,
                                                                                               self.imbalance_limit,
                                                                                               self.throughput_target)

        for pipelinestageindex in cpus_to_be_added.keys():
            to_be_added = cpus_to_be_added[pipelinestageindex]

            for cpu_name in to_be_added.keys():
                to_be_added_count = to_be_added[cpu_name]

                for i in range(0, to_be_added_count):
                    if cpu_name not in final_cpus_to_be_dropped:
                        break
                    if len(final_cpus_to_be_dropped[cpu_name]['busy']) > 0:
                        to_be_added[cpu_name] -= 1
                        final_cpus_to_be_dropped[cpu_name]['busy'].pop(0)
                    elif len(final_cpus_to_be_dropped[cpu_name]['free']) > 0:
                        to_be_added[cpu_name] -= 1
                        final_cpus_to_be_dropped[cpu_name]['busy'].pop(0)

        for pipelinestageindex in gpus_to_be_added.keys():
            to_be_added = gpus_to_be_added[pipelinestageindex]

            for gpu_name in to_be_added.keys():
                to_be_added_count = to_be_added[gpu_name]

                for i in range(0, to_be_added_count):
                    if gpu_name not in final_gpus_to_be_dropped:
                        break
                    if len(final_gpus_to_be_dropped[gpu_name]['busy']) > 0:
                        to_be_added[gpu_name] -= 1
                        final_gpus_to_be_dropped[gpu_name]['busy'].pop(0)
                    elif len(final_gpus_to_be_dropped[gpu_name]['free']) > 0:
                        to_be_added[gpu_name] -= 1
                        final_gpus_to_be_dropped[gpu_name]['busy'].pop(0)

        for cpu_name in final_cpus_to_be_dropped:
            for cpu_id in final_cpus_to_be_dropped[cpu_name]['free']:
                self.delete_worker(rmanager, 'CPU', cpu_id)

        for gpu_name in final_gpus_to_be_dropped:
            for gpu_id in final_gpus_to_be_dropped[gpu_name]['free']:
                self.delete_worker(rmanager, 'GPU', gpu_id)

        for pipelinestageindex in cpus_to_be_added.keys():
            to_be_added = cpus_to_be_added[pipelinestageindex]
            for cpu_name in to_be_added.keys():
                count = to_be_added[cpu_name]
                for i in range(0, count):
                    self.add_worker(rmanager, True, False, cpu_name, None, 'on_demand', None, pipelinestageindex)

        for pipelinestageindex in gpus_to_be_added.keys():
            to_be_added = gpus_to_be_added[pipelinestageindex]
            for gpu_name in to_be_added.keys():
                count = to_be_added[gpu_name]
                for i in range(0, count):
                    self.add_worker(rmanager, False, True, None, gpu_name, 'on_demand', None, pipelinestageindex)

        '''
        for cpu_type in new_cpus_to_be_added.keys():
            count = new_cpus_to_be_added[cpu_type]
            for i in range(0, count):
                self.add_worker(rmanager, True, False, cpu_type, None, 'on_demand', None)

        for gpu_type in new_gpus_to_be_added.keys():
            count = new_gpus_to_be_added[gpu_type]
            for i in range(0, count):
                self.add_worker(rmanager, False, True, None, gpu_type, 'on_demand', None)
        '''

        print('------------------------------------------------------')

    def reconfiguration_no_prediction(self, rmanager, pmanager, idle_cpus, idle_gpus):
        print('-----------------------------------------------------')
        cpus_to_be_dropped, gpus_to_be_dropped, cpus_to_be_added, gpus_to_be_added = pmanager.reconfiguration_down(
            rmanager, self.env.now, idle_cpus, idle_gpus)

        new_cpus_to_be_added = {}
        new_gpus_to_be_added = {}

        for pipelinestageindex in cpus_to_be_added.keys():
            to_be_added = cpus_to_be_added[pipelinestageindex]
            for cpu_name in to_be_added.keys():
                if cpu_name in new_cpus_to_be_added.keys():
                    new_cpus_to_be_added[cpu_name] += to_be_added[cpu_name]
                else:
                    new_cpus_to_be_added[cpu_name] = to_be_added[cpu_name]

        for pipelinestageindex in gpus_to_be_added.keys():
            to_be_added = gpus_to_be_added[pipelinestageindex]
            for gpu_name in to_be_added.keys():
                if gpu_name in new_gpus_to_be_added.keys():
                    new_gpus_to_be_added[gpu_name] += to_be_added[gpu_name]
                else:
                    new_gpus_to_be_added[gpu_name] = to_be_added[gpu_name]

        new_cpus_to_be_dropped = []
        new_gpus_to_be_dropped = []

        for cpu_id in cpus_to_be_dropped:
            resource = rmanager.get_resource(cpu_id)
            if resource.cpu.name in new_cpus_to_be_added:
                new_cpus_to_be_added[resource.cpu.name] -= 1
                if new_cpus_to_be_added[resource.cpu.name] <= 0:
                    new_cpus_to_be_added.pop(resource.cpu.name, None)
            else:
                new_cpus_to_be_dropped.append(resource.id)

        for gpu_id in gpus_to_be_dropped:
            resource = rmanager.get_resource(gpu_id)
            if resource.gpu.name in gpus_to_be_added:
                new_gpus_to_be_added[resource.gpu.name] -= 1
                if new_gpus_to_be_added[resource.gpu.name] <= 0:
                    new_gpus_to_be_added.pop(resource.gpu.name, None)
            else:
                new_gpus_to_be_dropped.append(resource.id)

        for cpu_id in new_cpus_to_be_dropped:
            if cpu_id not in idle_cpus:
                print('CPU', cpu_id, 'not idle to be dropped')
                continue
            self.delete_worker(rmanager, 'CPU', cpu_id)
        for gpu_id in new_gpus_to_be_dropped:
            if gpu_id not in idle_gpus:
                print('GPU', gpu_id, 'not idle to be dropped')
                continue
            self.delete_worker(rmanager, 'GPU', gpu_id)

        for cpu_type in new_cpus_to_be_added.keys():
            count = new_cpus_to_be_added[cpu_type]
            for i in range(0, count):
                self.add_worker(rmanager, True, False, cpu_type, None, 'on_demand', None)

        for gpu_type in new_gpus_to_be_added.keys():
            count = new_gpus_to_be_added[gpu_type]
            for i in range(0, count):
                self.add_worker(rmanager, False, True, None, gpu_type, 'on_demand', None)

        print('-----------------------------------------------------')

        return

    def replenish_workitems(self, imanager, pmanager, scheduling_policy, batchsize):
        first_pipelinestage = pmanager.get_pipelinestage(None, 'CPU')
        if first_pipelinestage == None:
            first_resourcetype = 'GPU'
        else:
            first_resourcetype = 'CPU'

        count = 0
        for i in range(0, batchsize):
            new_workitem = scheduling_policy.create_workitem(imanager, pmanager, None, first_resourcetype)
            if new_workitem == None:
                break
            count += 1

        return count

    def report_idle_periods(self, rmanager, since_time, current_time, last_phase_closed_index):
        print('report_idle_periods ()')
        resources = rmanager.get_resources('active', True)

        cpu = {}
        gpu = {}
        for resource in resources:
            cpu_idle_periods, gpu_idle_periods = resource.report_idle_periods(since_time, current_time)
            '''
            if cpu_idle_periods != None:
                print ('CPU', cpu_idle_periods)
            if gpu_idle_periods != None:
                print ('GPU', gpu_idle_periods)
            '''

            if cpu_idle_periods != None:
                total_idle_period = 0
                start = None
                end = None
                for period in cpu_idle_periods:
                    total_idle_period += period[1] - period[0]
                print(resource.id, 'CPU', since_time, current_time, total_idle_period,
                      (total_idle_period / (current_time - since_time) * 100))
                cpu[resource.id] = total_idle_period

            if gpu_idle_periods != None:
                total_idle_period = 0
                for period in gpu_idle_periods:
                    total_idle_period += period[1] - period[0]
                print(resource.id, 'GPU', since_time, current_time, total_idle_period,
                      (total_idle_period / (current_time - since_time) * 100))
                gpu[resource.id] = total_idle_period

        self.idle_periods[str(last_phase_closed_index)] = [cpu, gpu]

    def add_idle_periods(self, rmanager, now):
        resources = rmanager.get_resources()

        for resource in resources:
            resource.add_idle_period('CPU', now)
            resource.add_idle_period('GPU', now)

    def clear_completion_times(self, rmanager):
        resources = rmanager.get_resources()

        for resource in resources:
            resource.clear_completion_times()

    def set_init_idle_start_times(self, rmanager, now):
        resources = rmanager.get_resources('active', True)

        for resource in resources:
            resource.set_idle_start_time('CPU', now)
            resource.set_idle_start_time('GPU', now)

    def set_idle_start_times(self, rmanager, now):
        resources = rmanager.get_resources('active', True)

        for resource in resources:
            cpu_idle, gpu_idle = resource.is_idle()
            if cpu_idle == True and resource.cpu.idle_start_time < now:
                resource.cpu.add_idle_period(now)
                resource.cpu.set_idle_start_time(now)

            if gpu_idle == True and resource.gpu.idle_start_time < now:
                resource.gpu.add_idle_period(now)
                resource.gpu.set_idle_start_time(now)

    def init_clusters(self, rmanager, compute_type):
        self.worker_threads = {}
        self.workers = {}

        resource_types = rmanager.parse_resources(compute_type)

        resource_types = self.mapping.replace_arc_with_aws(resource_types)

        for resource_type in resource_types:
            count = resource_types[resource_type]
            for i in range(0, count):
                self.add_worker(rmanager, True, False, compute_type, None, 'on_demand', None, None)

    def explore(self, rmanager, pipelinestage, compute_type):
        pass

    def run_no_prediction_pin(self, rmanager, imanager, pmanager):
        scheduling_policy = FirstCompleteFirstServe("FirstCompleteFirstServe", self.env, pmanager)

        while True:
            new_workitem = scheduling_policy.create_workitem(imanager, pmanager)
            if new_workitem == None:
                break

        last_phase_closed_time = self.env.now

        self.reconfiguration_last_time = None

        no_of_phases_closed = 0

        try:
            while True:
                pipelinestages = pmanager.pipelinestages

                for pipelinestage in pipelinestages:
                    if pipelinestage.get_exploration() == True:
                        self.explore(pipelinestage)

                resources = rmanager.get_resources('active', True)
                for resource in resources:
                    # print ('###########################')
                    resource.get_status(rmanager, pmanager, self.worker_threads[resource.id], self.outputfile)
                    # print ('###########################')
                    # print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    scheduling_policy.remove_complete_workitem(resource, pmanager, self.env, imanager)
                    # print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                empty_cpus = []
                empty_gpus = []

                for resource in resources:
                    # print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                    cpu_empty, gpu_empty = resource.is_empty()
                    # print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

                    if cpu_empty == True:
                        empty_cpus.append(resource)
                    if gpu_empty == True:
                        empty_gpus.append(resource)

                if len(empty_cpus) > 0:
                    # print ('****************************')
                    scheduling_policy.add_new_workitems_DFS(rmanager, imanager, pmanager, empty_cpus, 'CPU')
                    # print ('****************************')
                if len(empty_gpus) > 0:
                    # print ('&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
                    scheduling_policy.add_new_workitems_DFS(rmanager, imanager, pmanager, empty_gpus, 'GPU')
                    # print ('&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

                # close the completed phases
                last_phase_closed_index = pmanager.close_phases_fixed(rmanager, False)

                idle_cpus = []
                idle_gpus = []

                # print ('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                for resource in resources:
                    cpu_idle, gpu_idle = resource.is_idle()

                    if cpu_idle == True:
                        idle_cpus.append(resource)
                    if gpu_idle == True:
                        idle_gpus.append(resource)

                    # print (idle_cpus)
                    # print (idle_gpus)
                    # print ('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

                for idle_cpu in idle_cpus:
                    # print ('scheduling cpu', idle_cpu.id)
                    idle_cpu.schedule(rmanager, pmanager, 'CPU', self.workers[idle_cpu.id][0].get_exec(), self.env)

                for idle_gpu in idle_gpus:
                    # print ('scheduling gpu', idle_gpu.id)
                    idle_gpu.schedule(rmanager, pmanager, 'GPU', self.workers[idle_gpu.id][1].get_exec(), self.env)

                pmanager.record_throughput(self.env)

                # scaling code goes here

                idle_cpus = []
                idle_gpus = []
                for resource in resources:
                    cpu_idle, gpu_idle = resource.is_idle()

                    if cpu_idle == True:
                        idle_cpus.append(resource.id)
                    if gpu_idle == True:
                        idle_gpus.append(resource.id)

                inactive_resources = rmanager.get_resources('active', False)

                for resource in inactive_resources:
                    if resource.cpu != None:
                        idle_cpus.append(resource.id)
                    if resource.gpu != None:
                        idle_gpus.append(resource.id)

                if pmanager.get_pct_complete_no_prediction() >= 10:
                    if self.reconfiguration_last_time == None or self.reconfiguration_last_time + (
                            self.reconfiguration_time_delta / 60) < self.env.now:
                        if self.algo == 'down':
                            self.reconfiguration_no_prediction(rmanager, pmanager, idle_cpus, idle_gpus)
                        elif self.algo == 'overallocation':
                            self.reconfiguration_no_prediction_up_down_overallocations_first(rmanager, pmanager,
                                                                                             idle_cpus,
                                                                                             idle_gpus)
                        elif self.algo == 'underallocation':
                            self.reconfiguration_no_prediction_up_down_underallocations_first(rmanager, pmanager,
                                                                                              idle_cpus,
                                                                                              idle_gpus)
                        self.reconfiguration_last_time = self.env.now

                # predict the execution pattern
                if last_phase_closed_index != None:
                    self.set_idle_start_times(rmanager, self.env.now)
                    self.report_idle_periods(rmanager, last_phase_closed_time, self.env.now, last_phase_closed_index)
                    last_phase_closed_time = self.env.now
                    no_of_phases_closed += 1
                    phase_tracker = last_phase_closed_index

                idle_cpus = []
                idle_gpus = []

                resources = rmanager.get_resources('active', True)

                for resource in resources:
                    cpu_idle, gpu_idle = resource.is_idle()
                    if cpu_idle == True:
                        idle_cpus.append(resource)
                    if gpu_idle == True:
                        idle_gpus.append(resource)

                inactive_resources = rmanager.get_resources('active', False)

                for resource in inactive_resources:
                    if resource.cpu != None:
                        idle_cpus.append(resource.id)
                    if resource.gpu != None:
                        idle_gpus.append(resource.id)

                # print (len (idle_cpus), len(idle_gpus), rmanager.get_cpu_resources_count(), rmanager.get_gpu_resources_count())
                # if len(idle_cpus) == rmanager.get_cpu_resources_count(active=True) and len(
                #        idle_gpus) == rmanager.get_gpu_resources_count(active=True) and last_phase_closed_index == pmanager.no_of_columns - 1:
                if pmanager.get_pct_complete_no_prediction() >= 100:
                    self.set_idle_start_times(rmanager, self.env.now)
                    if self.env.now > last_phase_closed_time:
                        self.report_idle_periods(rmanager, last_phase_closed_time, self.env.now,
                                                 last_phase_closed_index)
                    print('all tasks complete', self.env.now)

                    print(idle_cpus)
                    print(idle_gpus)

                    for idle_cpu in idle_cpus:
                        self.delete_worker(rmanager, 'CPU', idle_cpu.id)
                    for idle_gpu in idle_gpus:
                        self.delete_worker(rmanager, 'GPU', idle_gpu.id)
                    cpu_cost, gpu_cost = rmanager.get_total_cost()
                    print('total cost', cpu_cost, gpu_cost, self.env.now)

                    # add_performance_data(self.algo, cpu_cost, gpu_cost, self.env.now, self.reconfiguration_time_delta, self.imbalance_limit)

                    # pmanager.print_stage_queue_data_1()
                    pmanager.print_stage_queue_data_2(rmanager)
                    # pmanager.print_stage_queue_data_3 (self.idle_periods)
                    break

                yield self.env.timeout(5 / 3600)

        except simpy.Interrupt as i:
            print('WOW!')

    def run_no_prediction(self, rmanager, imanager, pmanager):
        print('OAI_scheduler_2 ()', 'waiting for 5 secs')

        self.init_clusters(rmanager)

        scheduling_policy = FirstCompleteFirstServe("FirstCompleteFirstServe", self.env, pmanager)

        while True:
            new_workitem = scheduling_policy.create_workitem(imanager, pmanager)
            if new_workitem == None:
                break

        last_phase_closed_time = self.env.now

        # self.set_init_idle_start_times(rmanager, self.env.now)
        self.reconfiguration_last_time = None

        no_of_phases_closed = 0
        phase_tracker = -1

        try:
            while True:
                resources = rmanager.get_resources('active', True)
                for resource in resources:
                    # print ('###########################')
                    resource.get_status(rmanager, pmanager, self.worker_threads[resource.id], self.outputfile)
                    # print ('###########################')
                    # print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    scheduling_policy.remove_complete_workitem(resource, pmanager, self.env, imanager)
                    # print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                empty_cpus = []
                empty_gpus = []

                for resource in resources:
                    # print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                    cpu_empty, gpu_empty = resource.is_empty()
                    # print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

                    if cpu_empty == True:
                        empty_cpus.append(resource)
                    if gpu_empty == True:
                        empty_gpus.append(resource)

                if len(empty_cpus) > 0:
                    # print ('****************************')
                    scheduling_policy.add_new_workitems_DFS(rmanager, imanager, pmanager, empty_cpus, 'CPU')
                    # print ('****************************')
                if len(empty_gpus) > 0:
                    # print ('&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
                    scheduling_policy.add_new_workitems_DFS(rmanager, imanager, pmanager, empty_gpus, 'GPU')
                # print ('&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

                # close the completed phases
                last_phase_closed_index = pmanager.close_phases_fixed(rmanager, False)

                idle_cpus = []
                idle_gpus = []

                # print ('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                for resource in resources:
                    cpu_idle, gpu_idle = resource.is_idle()

                    if cpu_idle == True:
                        idle_cpus.append(resource)
                    if gpu_idle == True:
                        idle_gpus.append(resource)

                    # print (idle_cpus)
                    # print (idle_gpus)
                    # print ('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

                for idle_cpu in idle_cpus:
                    # print ('scheduling cpu', idle_cpu.id)
                    idle_cpu.schedule(rmanager, pmanager, 'CPU', self.workers[idle_cpu.id][0].get_exec(), self.env)

                for idle_gpu in idle_gpus:
                    # print ('scheduling gpu', idle_gpu.id)
                    idle_gpu.schedule(rmanager, pmanager, 'GPU', self.workers[idle_gpu.id][1].get_exec(), self.env)

                pmanager.record_throughput(self.env)

                # scaling code goes here

                idle_cpus = []
                idle_gpus = []
                for resource in resources:
                    cpu_idle, gpu_idle = resource.is_idle()

                    if cpu_idle == True:
                        idle_cpus.append(resource.id)
                    if gpu_idle == True:
                        idle_gpus.append(resource.id)

                inactive_resources = rmanager.get_resources('active', False)

                for resource in inactive_resources:
                    if resource.cpu != None:
                        idle_cpus.append(resource.id)
                    if resource.gpu != None:
                        idle_gpus.append(resource.id)

                if pmanager.get_pct_complete_no_prediction() >= 10:
                    if self.reconfiguration_last_time == None or self.reconfiguration_last_time + (
                            self.reconfiguration_time_delta / 60) < self.env.now:
                        if self.algo == 'down':
                            self.reconfiguration_no_prediction(rmanager, pmanager, idle_cpus, idle_gpus)
                        elif self.algo == 'overallocation':
                            self.reconfiguration_no_prediction_up_down_overallocations_first(rmanager, pmanager,
                                                                                             idle_cpus,
                                                                                             idle_gpus)
                        elif self.algo == 'underallocation':
                            self.reconfiguration_no_prediction_up_down_underallocations_first(rmanager, pmanager,
                                                                                              idle_cpus,
                                                                                              idle_gpus)
                        self.reconfiguration_last_time = self.env.now

                # predict the execution pattern
                if last_phase_closed_index != None:
                    self.set_idle_start_times(rmanager, self.env.now)
                    self.report_idle_periods(rmanager, last_phase_closed_time, self.env.now, last_phase_closed_index)
                    last_phase_closed_time = self.env.now
                    no_of_phases_closed += 1
                    phase_tracker = last_phase_closed_index

                idle_cpus = []
                idle_gpus = []

                resources = rmanager.get_resources('active', True)

                for resource in resources:
                    cpu_idle, gpu_idle = resource.is_idle()
                    if cpu_idle == True:
                        idle_cpus.append(resource)
                    if gpu_idle == True:
                        idle_gpus.append(resource)

                inactive_resources = rmanager.get_resources('active', False)

                for resource in inactive_resources:
                    if resource.cpu != None:
                        idle_cpus.append(resource.id)
                    if resource.gpu != None:
                        idle_gpus.append(resource.id)

                # print (len (idle_cpus), len(idle_gpus), rmanager.get_cpu_resources_count(), rmanager.get_gpu_resources_count())
                # if len(idle_cpus) == rmanager.get_cpu_resources_count(active=True) and len(
                #        idle_gpus) == rmanager.get_gpu_resources_count(active=True) and last_phase_closed_index == pmanager.no_of_columns - 1:
                if pmanager.get_pct_complete_no_prediction() >= 100:
                    self.set_idle_start_times(rmanager, self.env.now)
                    if self.env.now > last_phase_closed_time:
                        self.report_idle_periods(rmanager, last_phase_closed_time, self.env.now,
                                                 last_phase_closed_index)
                    print('all tasks complete', self.env.now)

                    print(idle_cpus)
                    print(idle_gpus)

                    for idle_cpu in idle_cpus:
                        self.delete_worker(rmanager, 'CPU', idle_cpu.id)
                    for idle_gpu in idle_gpus:
                        self.delete_worker(rmanager, 'GPU', idle_gpu.id)
                    cpu_cost, gpu_cost = rmanager.get_total_cost()
                    print('total cost', cpu_cost, gpu_cost, self.env.now)

                    # add_performance_data(self.algo, cpu_cost, gpu_cost, self.env.now, self.reconfiguration_time_delta, self.imbalance_limit)

                    # pmanager.print_stage_queue_data_1()
                    pmanager.print_stage_queue_data_2(rmanager)
                    # pmanager.print_stage_queue_data_3 (self.idle_periods)
                    break

                yield self.env.timeout(5 / 3600)
        except simpy.Interrupt as i:
            print('WOW!')

    def run_prediction(self, rmanager, imanager, pmanager):
        print('OAI_scheduler_2 ()', 'waiting for 5 secs')

        scheduling_policy = FirstCompleteFirstServe("FirstCompleteFirstServe", self.env, pmanager)

        resources = rmanager.get_resources()

        first_pipelinestage = pmanager.get_pipelinestage(None, 'CPU')
        if first_pipelinestage == None:
            first_resourcetype = 'GPU'
        else:
            first_resourcetype = 'CPU'

        while True:
            new_workitem = scheduling_policy.create_workitem(imanager, pmanager, None, first_resourcetype)
            if new_workitem == None:
                break

        last_phase_closed_time = self.env.now

        self.set_init_idle_start_times(rmanager, self.env.now)

        no_of_phases_closed = 0
        phase_tracker = -1
        last_phase_closed_index = None

        try:
            while True:
                for resource in resources:
                    # print ('###########################')
                    resource.get_status(rmanager, pmanager, self.worker_threads[resource.id], self.outputfile)
                    # print ('###########################')
                    # print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    scheduling_policy.remove_complete_workitem(resource, pmanager, self.env)
                    # print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                # scaling code goes here

                # self.scale_up_algo_0 (rmanager, pmanager, phase_tracker)

                empty_cpus = []
                empty_gpus = []

                for resource in resources:
                    # print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                    cpu_empty, gpu_empty = resource.is_empty()
                    # print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

                    if cpu_empty == True:
                        empty_cpus.append(resource)
                    if gpu_empty == True:
                        empty_gpus.append(resource)

                if no_of_phases_closed >= 1:
                    perform_idleness_check = True
                else:
                    perform_idleness_check = False
                if len(empty_cpus) > 0:
                    # print ('****************************')
                    scheduling_policy.add_new_workitems(rmanager, pmanager, empty_cpus, 'CPU', phase_tracker,
                                                        perform_idleness_check)
                    # print ('****************************')
                if len(empty_gpus) > 0:
                    # print ('&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
                    scheduling_policy.add_new_workitems(rmanager, pmanager, empty_gpus, 'GPU', phase_tracker,
                                                        perform_idleness_check)
                # print ('&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

                # close the completed phases
                last_phase_closed_index = pmanager.close_phases_fixed(rmanager, False)

                idle_cpus = []
                idle_gpus = []

                # print ('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                for resource in resources:
                    cpu_idle, gpu_idle = resource.is_idle()

                    if cpu_idle == True:
                        idle_cpus.append(resource)
                    if gpu_idle == True:
                        idle_gpus.append(resource)

                    # print (idle_cpus)
                    # print (idle_gpus)
                    # print ('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

                for idle_cpu in idle_cpus:
                    # print ('scheduling cpu', idle_cpu.id)
                    idle_cpu.schedule(rmanager, pmanager, 'CPU', self.workers[idle_cpu.id][0].get_exec(), self.env)

                for idle_gpu in idle_gpus:
                    # print ('scheduling gpu', idle_gpu.id)
                    idle_gpu.schedule(rmanager, pmanager, 'GPU', self.workers[idle_gpu.id][1].get_exec(), self.env)

                # predict the execution pattern
                if last_phase_closed_index != None:
                    self.set_idle_start_times(rmanager, self.env.now)
                    self.report_idle_periods(rmanager, last_phase_closed_time, self.env.now, last_phase_closed_index)
                    last_phase_closed_time = self.env.now
                    no_of_phases_closed += 1
                    phase_tracker = last_phase_closed_index
                    if no_of_phases_closed >= 1 and no_of_phases_closed < pmanager.no_of_columns:
                        pmanager.predict_execution_fixed(rmanager, self.env.now, self.batchsize,
                                                         last_phase_closed_index, self.no_of_prediction_phases)
                        self.convert_resource_idle_periods(rmanager, pmanager, phase_tracker)

                idle_cpus = []
                idle_gpus = []

                for resource in resources:
                    cpu_idle, gpu_idle = resource.is_idle()
                    if cpu_idle == True:
                        idle_cpus.append(resource)
                    if gpu_idle == True:
                        idle_gpus.append(resource)

                # print (len (idle_cpus), len(idle_gpus), rmanager.get_cpu_resources_count(), rmanager.get_gpu_resources_count())
                if len(idle_cpus) == rmanager.get_cpu_resources_count() and len(
                        idle_gpus) == rmanager.get_gpu_resources_count() and last_phase_closed_index == pmanager.no_of_columns - 1:
                    # last_phase_closed_index = pmanager.close_phases_fixed(rmanager, True)
                    self.set_idle_start_times(rmanager, self.env.now)
                    if self.env.now > last_phase_closed_time:
                        self.report_idle_periods(rmanager, last_phase_closed_time, self.env.now,
                                                 last_phase_closed_index)
                    print('all tasks complete', self.env.now)
                    # pmanager.print_stage_queue_data_1()
                    pmanager.print_stage_queue_data_2()
                    # pmanager.print_stage_queue_data_3 (self.idle_periods)
                    break

                yield self.env.timeout(5 / 3600)
        except simpy.Interrupt as i:
            print('WOW!')