from parslfluxsim.FirstCompleteFirstServe_sim import FirstCompleteFirstServe
from parslfluxsim.allocator_sim import Allocator
from parslfluxsim.scaling_sim import Scaler
import simpy


class OAI_Scheduler:
    def __init__(self, env):
        self.env = env
        self.idle_periods = {}

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

    def perform_initial_allocation (self, rmanager, pmanager, dmanager, allocator, exploration):
        for pipelinestage in pmanager.pipelinestages:

            compute_type = pipelinestage.resourcetype

            if exploration == True:
                all_domains = dmanager.get_domains()
                for domain in all_domains:
                    exploration_resource_types = domain.get_resource_types(pipelinestage.resourcetype)

                    cpu_ok = False
                    gpu_ok = False

                    if pipelinestage.resourcetype == 'CPU':
                        cpu_ok = True
                    else:
                        gpu_ok = True

                    for exploration_resource_type in exploration_resource_types:
                        new_resource = allocator.add_worker(rmanager, domain, cpu_ok, gpu_ok, exploration_resource_type, \
                                                            'on_demand', None, pipelinestage, -1)
            else:
                performance_to_cost_ratio_ranking = pmanager.performance_to_cost_ranking_pipelinestage (rmanager, pipelinestage.index)
                resource_type = list(performance_to_cost_ratio_ranking.keys())[0]
                cpu_ok = False
                gpu_ok = False
                if compute_type == 'CPU':
                    cpu_ok = True
                else:
                    gpu_ok = True

                domain = rmanager.get_resourcetype_info (resource_type, 'domain', 'on_demand')

                new_resource = allocator.add_worker(rmanager, domain, cpu_ok, gpu_ok, resource_type, 'on_demand',\
                                                    None, pipelinestage, -1)

        print ('initial allocation done')

    def populate_bagofworkitems (self, imanager, pmanager):
        for pipelinestage in pmanager.pipelinestages:
            pipelinestage.populate_bagofworkitems (imanager)

    def run_no_prediction_pin_core (self, rmanager, imanager, pmanager, dmanager, allocator, scaler, exploration):
        scheduling_policy = FirstCompleteFirstServe(self.env)

        self.populate_bagofworkitems (imanager, pmanager)
        print ('workitems created')

        try:
            self.perform_initial_allocation (rmanager, pmanager, dmanager, allocator, exploration)

            pipelinestage_completions = {}
            for pipelinestage in pmanager.pipelinestages:
                pipelinestage_completions[pipelinestage.name] = False

            while True:
                for pipelinestage in pmanager.pipelinestages:

                    allocator.get_status(rmanager, dmanager, pipelinestage, scheduling_policy)

                    pinned_resource_ids = pipelinestage.get_pinned_resources (rmanager, True)
                    for resource_id in pinned_resource_ids:
                        pinned_resource = rmanager.get_resource (resource_id, True)
                        if pinned_resource.get_active () == False:
                            continue

                        pinned_resource.get_status (rmanager, allocator.worker_threads[pinned_resource.id], self.outputfile)
                        ret = scheduling_policy.remove_complete_workitem (pinned_resource, pipelinestage, imanager, rmanager, dmanager)

                        if ret == True:
                            if exploration == True and pinned_resource.get_explored () == False:
                                pinned_resource.set_explored(True)
                            else:
                                pipelinestage.add_completion(1)

                    empty_resources = []
                    pinned_resource_ids = pipelinestage.get_pinned_resources(rmanager, True)

                    for resource_id in pinned_resource_ids:
                        pinned_resource = rmanager.get_resource (resource_id, True)
                        if pinned_resource.get_active () == False:
                            continue

                        is_empty = pinned_resource.is_empty (pipelinestage.resourcetype)

                        if is_empty == True:
                            empty_resources.append(pinned_resource)

                    if len(empty_resources) > 0:
                        scheduling_policy.add_new_workitems_DFS_pipelinestage(rmanager, imanager, dmanager, empty_resources, pipelinestage)

                    idle_resources = []

                    for resource_id in pinned_resource_ids:
                        pinned_resource = rmanager.get_resource(resource_id, True)
                        is_idle = pinned_resource.is_idle(pipelinestage.resourcetype)

                        if is_idle == True:
                            idle_resources.append(pinned_resource)

                    for idle_resource in idle_resources:
                        if pipelinestage.resourcetype == 'CPU':
                            idle_resource.schedule (rmanager, pmanager, pipelinestage.resourcetype,\
                                                    allocator.workers[idle_resource.id][0].get_exec(), self.env)
                        else:
                            idle_resource.schedule (rmanager, pmanager, pipelinestage.resourcetype, \
                                                    allocator.workers[idle_resource.id][1].get_exec(), self.env)

                # scaling code goes here
                if exploration == False:
                    scaler.scale_up_2x (rmanager, pmanager, dmanager, allocator)
                #pmanager.reconfiguration (rmanager, self.env)

                for pipelinestage in pmanager.pipelinestages:
                    idle_resources = []
                    active_pinned_resource_ids = pipelinestage.get_pinned_resources (rmanager, True)
                    explored_resources = []

                    for resource_id in active_pinned_resource_ids:
                        active_pinned_resource = rmanager.get_resource(resource_id, True)
                        if active_pinned_resource.get_explored () == True:
                            explored_resources.append(active_pinned_resource)

                        is_idle = active_pinned_resource.is_idle(pipelinestage.resourcetype)

                        if is_idle == True:
                            idle_resources.append(active_pinned_resource)

                    inactive_resources_ids = pipelinestage.get_pinned_resources(rmanager, False)
                    for resource_id in inactive_resources_ids:
                        inactive_pinned_resource = rmanager.get_resource(resource_id, True)
                        idle_resources.append(inactive_pinned_resource)


                    if exploration == False:
                        if pipelinestage.get_pending_workitems_count () <= 0:
                            pipelinestage_completions[pipelinestage.name] = True
                            for idle_resource in idle_resources:
                                if idle_resource.pfs.total_entries <= 0:
                                    allocator.delete_worker(rmanager, dmanager, idle_resource.id, pipelinestage)
                    else:
                        #print (pipelinestage.name, len (pipelinestage.pinned_resources), len (explored_resources))
                        if len (pipelinestage.pinned_resources) == len (explored_resources):
                            pipelinestage_completions[pipelinestage.name] = True


                all_stages_complete = True

                for pipelinestage in pmanager.pipelinestages:
                    if pipelinestage_completions[pipelinestage.name] == False:
                        all_stages_complete = False
                        break

                if all_stages_complete == True:
                    if exploration == True:
                        for pipelinestage in pmanager.pipelinestages:
                            active_pinned_resource_ids = pipelinestage.get_pinned_resources(rmanager, True)
                            for pinned_resource_id in active_pinned_resource_ids:
                                allocator.delete_worker(rmanager, dmanager, pinned_resource_id, pipelinestage)
                    break

                yield self.env.timeout(5 / 3600)

        except simpy.Interrupt as i:
            print('WOW!')

        print('all tasks complete', self.env.now)

        cpu_cost, gpu_cost = rmanager.get_total_cost()
        print('total cost', self.env.now, cpu_cost, gpu_cost)
        if exploration == False:
            pmanager.print_stage_queue_data_2(rmanager)

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