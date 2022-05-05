import time
from parslfluxsim.FirstCompleteFirstServe_sim import FirstCompleteFirstServe
from parslfluxsim.ExplorationScheduling import ExplorationScheduling
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
        self.exploration_threads = {}
        self.explorers = {}

    def add_worker (self, rmanager, cpuok, gpuok, cputype, gputype, provision_type, bidding_price, pipelinestageindex, exploration):
        if provision_type == 'on_demand':
            activepool = True
        else:
            activepool = False

        new_resource, provision_time = rmanager.add_resource (cpuok, gpuok, cputype, gputype, provision_type, activepool, bidding_price, pipelinestageindex, exploration)

        print (new_resource.cpu, new_resource.gpu)

        if new_resource.cpu != None:
            cpu_thread = ExecutionSimThread(self.env, new_resource, 'CPU', self.performancedata, provision_type, provision_time)
        else:
            cpu_thread = None

        if new_resource.gpu != None:
            gpu_thread = ExecutionSimThread(self.env, new_resource, 'GPU', self.performancedata, provision_type, provision_time)
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

        return new_resource

    def delete_worker (self, rmanager, resourcetype, resource_id, exploration):
        if resourcetype == 'CPU':
            worker = self.workers[resource_id][0]
        else:
            worker = self.workers[resource_id][1]

        worker_exec = worker.get_exec()

        worker_exec.interrupt ('cancel')

        self.worker_threads.pop (resource_id, None)

        self.workers.pop (resource_id, None)

        rmanager.delete_resource (resourcetype, resource_id, exploration, active=True)


    def reconfiguration_no_prediction_up_down_underallocations_first (self, rmanager, pmanager, idle_cpus, idle_gpus):
        cpus_to_be_added, gpus_to_be_added = pmanager.reconfiguration_up_down_underallocations (rmanager, self.env.now, idle_cpus, idle_gpus, self.imbalance_limit, self.throughput_target)

        final_cpus_to_be_dropped = {}
        final_gpus_to_be_dropped = {}

        cpus_to_be_dropped, gpus_to_be_dropped = pmanager.reconfiguration_up_down_overallocations (rmanager, self.env.now, idle_cpus, idle_gpus, self.imbalance_limit, self.throughput_target)

        for cpu_id in cpus_to_be_dropped:
            resource = rmanager.get_resource (cpu_id, True)
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


        cpus_to_be_dropped, gpus_to_be_dropped = pmanager.reconfiguration_drop (rmanager, self.env.now, idle_cpus, idle_gpus, self.imbalance_limit, self.throughput_target)

        for cpu_id in cpus_to_be_dropped:
            resource = rmanager.get_resource (cpu_id, True)
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
                idle_gpus.remove (gpu_id)
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
                for i in range (0, count):
                    self.add_worker(rmanager, True, False, cpu_name, None, 'on_demand', None, pipelinestageindex)

        for pipelinestageindex in gpus_to_be_added.keys():
            to_be_added = gpus_to_be_added[pipelinestageindex]
            for gpu_name in to_be_added.keys():
                count = to_be_added[gpu_name]
                for i in range (0, count):
                    self.add_worker(rmanager, False, True, None, gpu_name, 'on_demand', None, pipelinestageindex)


    def reconfiguration_no_prediction_up_down_overallocations_first (self, rmanager, pmanager, idle_cpus, idle_gpus):

        print ('------------------------------------------------------')
        final_cpus_to_be_dropped = {}
        final_gpus_to_be_dropped = {}

        cpus_to_be_dropped, gpus_to_be_dropped = pmanager.reconfiguration_up_down_overallocations (rmanager, self.env.now, idle_cpus, idle_gpus, self.imbalance_limit, self.throughput_target)

        for cpu_id in cpus_to_be_dropped:
            resource = rmanager.get_resource (cpu_id, True)
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

        cpus_to_be_dropped, gpus_to_be_dropped = pmanager.reconfiguration_drop (rmanager, self.env.now, idle_cpus, idle_gpus, self.imbalance_limit, self.throughput_target)

        for cpu_id in cpus_to_be_dropped:
            resource = rmanager.get_resource (cpu_id, True)
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
                idle_gpus.remove (gpu_id)
            else:
                if gpu_id not in idle_gpus:
                    final_gpus_to_be_dropped[resource.gpu.name]['busy'].append(gpu_id)
                else:
                    final_gpus_to_be_dropped[resource.gpu.name]['free'].append(gpu_id)
                idle_gpus.remove(gpu_id)


        cpus_to_be_added, gpus_to_be_added = pmanager.reconfiguration_up_down_underallocations(rmanager, self.env.now,
                                                                                               idle_cpus, idle_gpus, self.imbalance_limit, self.throughput_target)


        for pipelinestageindex in cpus_to_be_added.keys ():
            to_be_added = cpus_to_be_added[pipelinestageindex]

            for cpu_name in to_be_added.keys ():
                to_be_added_count = to_be_added[cpu_name]

                for i in range(0, to_be_added_count):
                    if cpu_name not in final_cpus_to_be_dropped:
                        break
                    if len (final_cpus_to_be_dropped[cpu_name]['busy']) > 0:
                        to_be_added[cpu_name] -= 1
                        final_cpus_to_be_dropped[cpu_name]['busy'].pop (0)
                    elif len (final_cpus_to_be_dropped[cpu_name]['free']) > 0:
                        to_be_added[cpu_name] -= 1
                        final_cpus_to_be_dropped[cpu_name]['busy'].pop (0)

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
                for i in range (0, count):
                    self.add_worker(rmanager, True, False, cpu_name, None, 'on_demand', None, pipelinestageindex)

        for pipelinestageindex in gpus_to_be_added.keys():
            to_be_added = gpus_to_be_added[pipelinestageindex]
            for gpu_name in to_be_added.keys():
                count = to_be_added[gpu_name]
                for i in range (0, count):
                    self.add_worker(rmanager, False, True, None, gpu_name, 'on_demand', None, pipelinestageindex)

        print('------------------------------------------------------')

    def reconfiguration_no_prediction (self, rmanager, pmanager, idle_cpus, idle_gpus):
        print ('-----------------------------------------------------')
        cpus_to_be_dropped, gpus_to_be_dropped, cpus_to_be_added, gpus_to_be_added = pmanager.reconfiguration_down (rmanager, self.env.now, idle_cpus, idle_gpus)

        new_cpus_to_be_added = {}
        new_gpus_to_be_added = {}

        for pipelinestageindex in cpus_to_be_added.keys ():
            to_be_added = cpus_to_be_added[pipelinestageindex]
            for cpu_name in to_be_added.keys ():
                if cpu_name in new_cpus_to_be_added.keys ():
                    new_cpus_to_be_added[cpu_name] += to_be_added[cpu_name]
                else:
                    new_cpus_to_be_added[cpu_name] = to_be_added[cpu_name]

        for pipelinestageindex in gpus_to_be_added.keys ():
            to_be_added = gpus_to_be_added[pipelinestageindex]
            for gpu_name in to_be_added.keys ():
                if gpu_name in new_gpus_to_be_added.keys ():
                    new_gpus_to_be_added[gpu_name] += to_be_added[gpu_name]
                else:
                    new_gpus_to_be_added[gpu_name] = to_be_added[gpu_name]



        new_cpus_to_be_dropped = []
        new_gpus_to_be_dropped = []

        for cpu_id in cpus_to_be_dropped:
            resource = rmanager.get_resource (cpu_id)
            if resource.cpu.name in new_cpus_to_be_added:
                new_cpus_to_be_added[resource.cpu.name] -= 1
                if new_cpus_to_be_added[resource.cpu.name] <= 0:
                    new_cpus_to_be_added.pop (resource.cpu.name, None)
            else:
                new_cpus_to_be_dropped.append (resource.id)

        for gpu_id in gpus_to_be_dropped:
            resource = rmanager.get_resource (gpu_id)
            if resource.gpu.name in gpus_to_be_added:
                new_gpus_to_be_added[resource.gpu.name] -= 1
                if new_gpus_to_be_added[resource.gpu.name] <= 0:
                    new_gpus_to_be_added.pop (resource.gpu.name, None)
            else:
                new_gpus_to_be_dropped.append (resource.id)


        for cpu_id in new_cpus_to_be_dropped:
            if cpu_id not in idle_cpus:
                print ('CPU', cpu_id, 'not idle to be dropped')
                continue
            self.delete_worker(rmanager, 'CPU', cpu_id)
        for gpu_id in new_gpus_to_be_dropped:
            if gpu_id not in idle_gpus:
                print('GPU', gpu_id, 'not idle to be dropped')
                continue
            self.delete_worker(rmanager, 'GPU', gpu_id)

        for cpu_type in new_cpus_to_be_added.keys():
            count = new_cpus_to_be_added[cpu_type]
            for i in range (0, count):
                self.add_worker(rmanager, True, False, cpu_type, None, 'on_demand', None)

        for gpu_type in new_gpus_to_be_added.keys():
            count = new_gpus_to_be_added[gpu_type]
            for i in range (0, count):
                self.add_worker(rmanager, False, True, None, gpu_type, 'on_demand', None)

        print('-----------------------------------------------------')

        return

    def report_idle_periods (self, rmanager, since_time, current_time, last_phase_closed_index):
        print ('report_idle_periods ()')
        resources = rmanager.get_resources ('active', True)

        cpu = {}
        gpu = {}
        for resource in resources:
            cpu_idle_periods, gpu_idle_periods = resource.report_idle_periods (since_time, current_time)

            if cpu_idle_periods != None:
                total_idle_period = 0
                start = None
                end = None
                for period in cpu_idle_periods:
                    total_idle_period += period[1] - period[0]
                print (resource.id, 'CPU', since_time, current_time, total_idle_period, (total_idle_period / (current_time - since_time) * 100))
                cpu[resource.id] = total_idle_period

            if gpu_idle_periods != None:
                total_idle_period = 0
                for period in gpu_idle_periods:
                    total_idle_period += period[1] - period[0]
                print (resource.id, 'GPU', since_time, current_time, total_idle_period, (total_idle_period / (current_time - since_time) * 100))
                gpu[resource.id] = total_idle_period

        self.idle_periods [str (last_phase_closed_index)] = [cpu, gpu]

    def add_idle_periods (self, rmanager, now):
        resources = rmanager.get_resources ()

        for resource in resources:
            resource.add_idle_period ('CPU', now)
            resource.add_idle_period ('GPU', now)

    def clear_completion_times (self, rmanager):
        resources = rmanager.get_resources ()

        for resource in resources:
            resource.clear_completion_times ()

    def set_init_idle_start_times (self, rmanager, now):
        resources = rmanager.get_resources ('active', True)

        for resource in resources:
            resource.set_idle_start_time ('CPU', now)
            resource.set_idle_start_time ('GPU', now)

    def set_idle_start_times (self, rmanager, now):
        resources = rmanager.get_resources ('active', True)

        for resource in resources:
            cpu_idle, gpu_idle = resource.is_idle ()
            if cpu_idle == True and resource.cpu.idle_start_time < now:
                resource.cpu.add_idle_period (now)
                resource.cpu.set_idle_start_time (now)

            if gpu_idle == True and resource.gpu.idle_start_time < now:
                resource.gpu.add_idle_period (now)
                resource.gpu.set_idle_start_time (now)

    def init_clusters (self, rmanager, compute_type):

        resource_types = rmanager.parse_resources (compute_type)

        resource_types = self.mapping.replace_arc_with_aws (resource_types)

        for resource_type in resource_types:
            count = resource_types[resource_type]
            for i in range (0, count):
                self.add_worker(rmanager, True, False, resource_type, None, 'on_demand', None, None)

    def explore (self, rmanager, pipelinestage):
        if pipelinestage.priority != 0 and pipelinestage.get_pending_count () <= 0:
            return False

        compute_type = pipelinestage.resourcetype
        exploration_resource_types = rmanager.parse_resources (compute_type)
        exploration_resource_types = self.mapping.replace_arc_with_aws(exploration_resource_types)

        cpu_type = False
        gpu_type = False

        if compute_type == 'CPU':
            cpu_type = True
        else:
            gpu_type = True

        for exploration_resource_type in exploration_resource_types:
            count = exploration_resource_types[exploration_resource_type]
            if compute_type == 'CPU':
                cpu_resource_type = exploration_resource_type
                gpu_resource_type = None
            else:
                cpu_resource_type = None
                gpu_resource_type = exploration_resource_type

            for i in range (0, count):
                new_resource = self.add_worker (rmanager, cpu_type, gpu_type, cpu_resource_type, gpu_resource_type, 'on_demand', None, None, True)
                pipelinestage.add_exploration_resource (new_resource)

        return True

    def schedule_exploration (self, rmanager, pmanager, pipelinestage, exploration_scheduling_policy, scheduling_policy):
        exploration_scheduling_policy.add_workitem_exploration (rmanager, pmanager, pipelinestage, scheduling_policy)

        exploration_resources = pipelinestage.get_explorers()
        for exploration_resource_id in exploration_resources:
            resource = rmanager.get_resource(exploration_resource_id, True)

            if pipelinestage.get_exploration_scheduled (resource.id) == False and resource.get_active () == True:
                print ('schedule_exploration ()', pipelinestage.name, resource.id, 'scheduling')
                if pipelinestage.resourcetype == 'CPU':
                    resource.schedule(rmanager, pmanager, 'CPU', self.workers[resource.id][0].get_exec(), self.env)
                else:
                    resource.schedule(rmanager, pmanager, 'GPU', self.workers[resource.id][1].get_exec(), self.env)

                pipelinestage.set_exploration_scheduled (resource.id, True)

    def end_exploration (self, rmanager, imanager, pmanager, scheduling_policy, pipelinestage):
        exploration_resources = pipelinestage.get_explorers ()

        for exploration_resource_id in exploration_resources:
            if pipelinestage.get_exploration_ended (exploration_resource_id) == True:
                continue

            resource = rmanager.get_resource(exploration_resource_id, True)
            diff = resource.get_status(rmanager, pmanager, self.worker_threads[resource.id], self.outputfile)

            print ('end_exploration ()', pipelinestage.name, exploration_resource_id, diff)

            if diff == None:
                continue

            if pipelinestage.get_exploration_ended_count () <= 0:
                print ('end_exploration', 'removing workitem')
                scheduling_policy.remove_complete_workitem (resource, pmanager, self.env, imanager)
            else:
                if pipelinestage.resourcetype == 'CPU':
                    resource.pop_if_complete('CPU')
                else:
                    resource.pop_if_complete('GPU')

            pipelinestage.set_exploration_ended(exploration_resource_id, True, diff)

        print ('end_exploration ()', pipelinestage.name, pipelinestage.get_exploration_ended_count ())

        if pipelinestage.get_all_exploration_ended () == True:
            performance_to_cost_ratio = pmanager.get_performance_to_cost_ratio_ranking (rmanager, pipelinestage.index, exploration_resources)
            deletion_list = list (performance_to_cost_ratio.keys ())
            pinned_resource = deletion_list.pop(0)
            for explorer_id in deletion_list:
                if pipelinestage.resourcetype == 'GPU':
                    self.delete_worker(rmanager, 'GPU', explorer_id, True)
                else:
                    self.delete_worker(rmanager, 'CPU', explorer_id, True)

                pipelinestage.remove_exploration_resource(explorer_id)

            pipelinestage.remove_exploration_resource (pinned_resource)

            pipelinestage.add_pinned_resource (pinned_resource)


    def run_no_prediction_pin (self, rmanager, imanager, pmanager):
        scheduling_policy = FirstCompleteFirstServe("FirstCompleteFirstServe", self.env, pmanager)
        exploration_scheduling_policy = ExplorationScheduling ("Exploration", self.env, pmanager)

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
                    if pipelinestage.get_exploration_needed () == True:
                        print ('--------------------------------------')
                        print (pipelinestage.name, 'exloration_needed')
                        ret = self.explore (rmanager, pipelinestage)

                        if ret == True:
                            pipelinestage.set_exploration_needed (False)
                            pipelinestage.set_exploration_scheduling_needed (True)

                        print('--------------------------------------')

                    elif pipelinestage.get_exploration_scheduling_needed () == True:
                        print ('######################################')
                        print(pipelinestage.name, 'exloration_scheduling_needed')
                        self.schedule_exploration (rmanager, pmanager, pipelinestage, exploration_scheduling_policy, scheduling_policy)
                        if pipelinestage.get_all_exploration_scheduled () == True:
                            pipelinestage.set_exploration_scheduling_needed (False)
                            pipelinestage.set_exploration_ending_needed (True)
                        print('######################################')
                    elif pipelinestage.get_exploration_ending_needed ()  == True:
                        print ('*************************************')
                        print(pipelinestage.name, 'exloration_ending_needed')
                        self.end_exploration (rmanager, imanager, pmanager, scheduling_policy, pipelinestage)
                        if pipelinestage.get_all_exploration_ended () == True:
                            pipelinestage.set_exploration_ending_needed (False)
                        print('*************************************')

                for pipelinestage in pipelinestages:
                    if pipelinestage.get_exploration_needed () == False and pipelinestage.get_exploration_scheduling_needed () == False and pipelinestage.get_exploration_ending_needed ()  == False:
                        pinned_resource_ids = pipelinestage.get_pinned_resources (rmanager, True)
                        for resource_id in pinned_resource_ids:
                            pinned_resource = rmanager.get_resource (resource_id, True)
                            # print ('###########################')
                            pinned_resource.get_status (rmanager, pmanager, self.worker_threads[pinned_resource.id], self.outputfile)
                            # print ('###########################')
                            # print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                            scheduling_policy.remove_complete_workitem (pinned_resource, pmanager, self.env, imanager)
                            # print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                        empty_cpus = []
                        empty_gpus = []

                        for resource_id in pinned_resource_ids:
                            pinned_resource = rmanager.get_resource (resource_id, True)
                            # print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                            cpu_empty, gpu_empty = pinned_resource.is_empty()
                            # print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

                            if cpu_empty == True:
                                empty_cpus.append(pinned_resource)
                            if gpu_empty == True:
                                empty_gpus.append(pinned_resource)

                        if len(empty_cpus) > 0:
                            # print ('****************************')
                            scheduling_policy.add_new_workitems_DFS_pipelinestage(rmanager, imanager, pmanager, empty_cpus, 'CPU', str(pipelinestage.index))
                            # print ('****************************')
                        if len(empty_gpus) > 0:
                            # print ('&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
                            scheduling_policy.add_new_workitems_DFS_pipelinestage(rmanager, imanager, pmanager, empty_gpus, 'GPU', str (pipelinestage.index))
                            # print ('&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

                        idle_cpus = []
                        idle_gpus = []

                        # print ('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                        for resource_id in pinned_resource_ids:
                            pinned_resource = rmanager.get_resource(resource_id, True)
                            cpu_idle, gpu_idle = pinned_resource.is_idle()

                            if cpu_idle == True:
                                idle_cpus.append(pinned_resource)
                            if gpu_idle == True:
                                idle_gpus.append(pinned_resource)

                            # print (idle_cpus)
                            # print (idle_gpus)
                            # print ('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

                        for idle_cpu in idle_cpus:
                            # print ('scheduling cpu', idle_cpu.id)
                            idle_cpu.schedule(rmanager, pmanager, 'CPU', self.workers[idle_cpu.id][0].get_exec(), self.env)

                        for idle_gpu in idle_gpus:
                            # print ('scheduling gpu', idle_gpu.id)
                            idle_gpu.schedule(rmanager, pmanager, 'GPU', self.workers[idle_gpu.id][1].get_exec(), self.env)

                last_phase_closed_index = pmanager.close_phases_fixed(rmanager, False)
                pmanager.record_throughput(self.env)

                # scaling code goes here

                '''
                idle_cpus = []
                idle_gpus = []
                for resource_id in pinned_resource_ids:
                    pinned_resource = rmanager.get_resource(resource_id, True)
                    cpu_idle, gpu_idle = pinned_resource.is_idle()

                    if cpu_idle == True:
                        idle_cpus.append(pinned_resource.id)
                    if gpu_idle == True:
                        idle_gpus.append(pinned_resource.id)

                inactive_resources_ids = pipelinestage.get_pinned_resources (rmanager, False)

                for resource_id in inactive_resources_ids:
                    pinned_resource = rmanager.get_resource(resource_id, True)
                    if pinned_resource.cpu != None:
                        idle_cpus.append(pinned_resource.id)
                    if pinned_resource.gpu != None:
                        idle_gpus.append(pinned_resource.id)


                if pmanager.get_pct_complete_no_prediction() >= 10:
                    if self.reconfiguration_last_time == None or self.reconfiguration_last_time + (
                        self.reconfiguration_time_delta / 60) < self.env.now:
                        if self.algo == 'down':
                            self.reconfiguration_no_prediction(rmanager, pmanager, idle_cpus, idle_gpus)
                        elif self.algo == 'overallocation':
                            self.reconfiguration_no_prediction_up_down_overallocations_first(rmanager, pmanager, idle_cpus,
                                                                                     idle_gpus)
                        elif self.algo == 'underallocation':
                            self.reconfiguration_no_prediction_up_down_underallocations_first(rmanager, pmanager, idle_cpus,
                                                                                      idle_gpus)
                        self.reconfiguration_last_time = self.env.now

                '''

                # predict the execution pattern
                if last_phase_closed_index != None:
                    self.set_idle_start_times(rmanager, self.env.now)
                    self.report_idle_periods(rmanager, last_phase_closed_time, self.env.now, last_phase_closed_index)
                    last_phase_closed_time = self.env.now
                    no_of_phases_closed += 1
                    phase_tracker = last_phase_closed_index

                idle_cpus = []
                idle_gpus = []

                for pipelinestage in pmanager.pipelinestages:
                    pinned_resource_ids = pipelinestage.get_pinned_resources (rmanager, True)
                    for resource_id in pinned_resource_ids:
                        pinned_resource = rmanager.get_resource(resource_id, True)
                        cpu_idle, gpu_idle = pinned_resource.is_idle()

                        if cpu_idle == True:
                            idle_cpus.append(pinned_resource)
                        if gpu_idle == True:
                            idle_gpus.append(pinned_resource)

                    inactive_resources_ids = pipelinestage.get_pinned_resources(rmanager, False)

                    for resource_id in inactive_resources_ids:
                        pinned_resource = rmanager.get_resource(resource_id, True)
                        if pinned_resource.cpu != None:
                            idle_cpus.append(pinned_resource)
                        if pinned_resource.gpu != None:
                            idle_gpus.append(pinned_resource)

                if pmanager.get_pct_complete_no_prediction() >= 100:
                    self.set_idle_start_times(rmanager, self.env.now)
                    if self.env.now > last_phase_closed_time:
                        self.report_idle_periods(rmanager, last_phase_closed_time, self.env.now,
                                                 last_phase_closed_index)
                    print('all tasks complete', self.env.now)

                    print(idle_cpus)
                    print(idle_gpus)

                    for idle_cpu in idle_cpus:
                        self.delete_worker(rmanager, 'CPU', idle_cpu.id, False)
                    for idle_gpu in idle_gpus:
                        self.delete_worker(rmanager, 'GPU', idle_gpu.id, False)
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


    def run_no_prediction (self, rmanager, imanager, pmanager):
        print('OAI_scheduler_2 ()', 'waiting for 5 secs')

        self.init_clusters (rmanager)

        scheduling_policy = FirstCompleteFirstServe("FirstCompleteFirstServe", self.env, pmanager)

        while True:
            new_workitem = scheduling_policy.create_workitem(imanager, pmanager)
            if new_workitem == None:
                break

        last_phase_closed_time = self.env.now

        #self.set_init_idle_start_times(rmanager, self.env.now)
        self.reconfiguration_last_time = None

        no_of_phases_closed = 0
        phase_tracker = -1

        try:
            while True:
                resources = rmanager.get_resources('active', True)
                for resource in resources:
                    # print ('###########################')
                    resource.get_status (rmanager, pmanager, self.worker_threads[resource.id], self.outputfile)
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

                pmanager.record_throughput (self.env)

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
                        idle_cpus.append (resource.id)
                    if resource.gpu != None:
                        idle_gpus.append (resource.id)

                if pmanager.get_pct_complete_no_prediction() >= 10:
                    if self.reconfiguration_last_time == None or self.reconfiguration_last_time + (self.reconfiguration_time_delta / 60) < self.env.now:
                        if self.algo == 'down':
                            self.reconfiguration_no_prediction(rmanager, pmanager, idle_cpus, idle_gpus)
                        elif self.algo == 'overallocation':
                            self.reconfiguration_no_prediction_up_down_overallocations_first(rmanager, pmanager, idle_cpus,
                                                                                          idle_gpus)
                        elif self.algo == 'underallocation':
                            self.reconfiguration_no_prediction_up_down_underallocations_first(rmanager, pmanager, idle_cpus,
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
                        idle_cpus.append(resource)
                    if resource.gpu != None:
                        idle_gpus.append(resource)

                # print (len (idle_cpus), len(idle_gpus), rmanager.get_cpu_resources_count(), rmanager.get_gpu_resources_count())
                #if len(idle_cpus) == rmanager.get_cpu_resources_count(active=True) and len(
                #        idle_gpus) == rmanager.get_gpu_resources_count(active=True) and last_phase_closed_index == pmanager.no_of_columns - 1:
                if pmanager.get_pct_complete_no_prediction () >= 100:
                    self.set_idle_start_times(rmanager, self.env.now)
                    if self.env.now > last_phase_closed_time:
                        self.report_idle_periods(rmanager, last_phase_closed_time, self.env.now,
                                                 last_phase_closed_index)
                    print('all tasks complete', self.env.now)

                    print (idle_cpus)
                    print (idle_gpus)

                    for idle_cpu in idle_cpus:
                        self.delete_worker(rmanager, 'CPU', idle_cpu.id, False)
                    for idle_gpu in idle_gpus:
                        self.delete_worker(rmanager, 'GPU', idle_gpu.id, False)
                    cpu_cost, gpu_cost = rmanager.get_total_cost()
                    print('total cost', cpu_cost, gpu_cost, self.env.now)

                    #add_performance_data(self.algo, cpu_cost, gpu_cost, self.env.now, self.reconfiguration_time_delta, self.imbalance_limit)

                    # pmanager.print_stage_queue_data_1()
                    pmanager.print_stage_queue_data_2(rmanager)
                    # pmanager.print_stage_queue_data_3 (self.idle_periods)
                    break

                yield self.env.timeout(5 / 3600)
        except simpy.Interrupt as i:
            print('WOW!')