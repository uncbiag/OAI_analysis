def reconfiguration_no_prediction_up_down_underallocations_first(self, rmanager, pmanager, idle_cpus, idle_gpus):
    cpus_to_be_added, gpus_to_be_added = pmanager.reconfiguration_up_down_underallocations(rmanager, self.env.now,
                                                                                           idle_cpus, idle_gpus,
                                                                                           self.imbalance_limit,
                                                                                           self.throughput_target)

    final_cpus_to_be_dropped = {}
    final_gpus_to_be_dropped = {}

    cpus_to_be_dropped, gpus_to_be_dropped = pmanager.reconfiguration_up_down_overallocations(rmanager, self.env.now,
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

    cpus_to_be_dropped, gpus_to_be_dropped = pmanager.reconfiguration_drop(rmanager, self.env.now, idle_cpus, idle_gpus,
                                                                           self.imbalance_limit, self.throughput_target)

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

    cpus_to_be_dropped, gpus_to_be_dropped = pmanager.reconfiguration_up_down_overallocations(rmanager, self.env.now,
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

    cpus_to_be_dropped, gpus_to_be_dropped = pmanager.reconfiguration_drop(rmanager, self.env.now, idle_cpus, idle_gpus,
                                                                           self.imbalance_limit, self.throughput_target)

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

    print('------------------------------------------------------')


def reconfiguration_no_prediction(self, rmanager, pmanager, idle_cpus, idle_gpus):
    print('-----------------------------------------------------')
    cpus_to_be_dropped, gpus_to_be_dropped, cpus_to_be_added, gpus_to_be_added = pmanager.reconfiguration_down(rmanager,
                                                                                                               self.env.now,
                                                                                                               idle_cpus,
                                                                                                               idle_gpus)

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

def run1(self, rmanager, imanager, pmanager, batchsize):
    print('OAI_scheduler_2 ()', 'waiting for 5 secs')

    scheduling_policy = FirstCompleteFirstServe("FirstCompleteFirstServe", self.env)

    resources = rmanager.get_resources()

    '''
    while True:
        new_workitem = scheduling_policy.create_workitem (imanager, pmanager, None, first_resourcetype)
        if new_workitem == None:
            break
    '''

    last_replenish_time = self.env.now
    self.replenish_workitems(imanager, pmanager, scheduling_policy, batchsize)
    replenish_done = False

    try:
        while True:
            for resource in resources:
                # print ('###########################')
                resource.get_status(rmanager, pmanager, self.worker_threads[resource.id], self.outputfile)
                # print ('###########################')
                # print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                scheduling_policy.remove_complete_workitem(resource, pmanager, self.env)
                # print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!')

            # scaling up code goes here

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
                scheduling_policy.add_new_workitems(rmanager, imanager, pmanager, empty_cpus, 'CPU')
                # print ('****************************')
            if len(empty_gpus) > 0:
                # print ('&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
                scheduling_policy.add_new_workitems(rmanager, imanager, pmanager, empty_gpus, 'GPU')
            # print ('&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

            # close the completed phases
            pmanager.close_phases(rmanager, self.env.now)

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
            # pmanager.predict_execution(rmanager, pmanager, self.env.now)
            if replenish_done == True:
                pmanager.predict_execution_fixed(rmanager, pmanager, self.env.now, batchsize)
                replenish_done = False

            idle_cpus = []
            idle_gpus = []

            for resource in resources:
                cpu_idle, gpu_idle = resource.is_idle()
                if cpu_idle == True:
                    idle_cpus.append(resource)
                if gpu_idle == True:
                    idle_gpus.append(resource)

            if len(idle_cpus) == len(resources) and len(idle_gpus) == len(resources):
                self.add_idle_periods(rmanager, self.env.now)
                self.report_idle_periods(rmanager, last_replenish_time, self.env.now)
                print('\n\n\n\n')
                self.clear_completion_times(rmanager)
                last_replenish_time = self.env.now
                replenish_done = True
                count = self.replenish_workitems(imanager, pmanager, scheduling_policy, batchsize)
                if count == 0:
                    print('all tasks complete')
                    pmanager.print_stage_queue_data()
                    break

            yield self.env.timeout(5 / 3600)
    except simpy.Interrupt as i:
        print('WOW!')


def run2(self, rmanager, imanager, pmanager, batchsize):
    print('OAI_scheduler_2 ()', 'waiting for 5 secs')

    scheduling_policy = FirstCompleteFirstServe("FirstCompleteFirstServe", self.env)

    first_pipelinestage = pmanager.get_pipelinestage(None, 'CPU')
    if first_pipelinestage == None:
        first_resourcetype = 'GPU'
    else:
        first_resourcetype = 'CPU'

    resources = rmanager.get_resources()

    while True:
        new_workitem = scheduling_policy.create_workitem(imanager, pmanager, None, first_resourcetype)
        if new_workitem == None:
            break

    try:
        while True:
            for resource in resources:
                # print ('###########################')
                resource.get_status(rmanager, pmanager, self.worker_threads[resource.id], self.outputfile)
                # print ('###########################')
                # print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                scheduling_policy.remove_complete_workitem(resource, pmanager, self.env)
                # print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!')

            # scaling up code goes here

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
                scheduling_policy.add_new_workitems(rmanager, imanager, pmanager, empty_cpus, 'CPU')
                # print ('****************************')
            if len(empty_gpus) > 0:
                # print ('&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
                scheduling_policy.add_new_workitems(rmanager, imanager, pmanager, empty_gpus, 'GPU')
            # print ('&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

            # close the completed phases
            pmanager.close_phases(rmanager, self.env.now)

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
            # pmanager.predict_execution(rmanager, pmanager, self.env.now)

            idle_cpus = []
            idle_gpus = []

            for resource in resources:
                cpu_idle, gpu_idle = resource.is_idle()
                if cpu_idle == True:
                    idle_cpus.append(resource)
                if gpu_idle == True:
                    idle_gpus.append(resource)

            if len(idle_cpus) == len(resources) and len(idle_gpus) == len(resources):
                print('all tasks complete')
                pmanager.print_stage_queue_data()
                break

            yield self.env.timeout(5 / 3600)
    except simpy.Interrupt as i:
        print('WOW!')

    def parse_resources_old (self):
        yaml_resourcefile = open (self.resourcefile)
        resources = yaml.load (yaml_resourcefile, Loader=yaml.FullLoader)

        for cputype in resources['available']['CPU']:
            if cputype['provision_type'] not in self.resourcetypeinfo:
                self.resourcetypeinfo['provision_type'] = {}
            self.resourcetypeinfo[cputype['provision_type']][cputype['id']] = {}
            self.resourcetypeinfo[cputype['provision_type']][cputype['id']]['startuptime'] = cputype['startuptime']
            self.resourcetypeinfo[cputype['provision_type']][cputype['id']]['resourcetype'] = 'CPU'
            self.resourcetypeinfo[cputype['provision_type']][cputype['id']]['availability'] = 1.0
            self.resourcetypeinfo[cputype['provision_type']][cputype['id']]['cost'] = cputype['cost']
            self.resourcetypeinfo[cputype['provision_type']][cputype['id']]['count'] = {}
            self.resourcetypeinfo[cputype['provision_type']][cputype['id']]['count']['time'] = [self.env.now]
            self.resourcetypeinfo[cputype['provision_type']][cputype['id']]['count']['count'] = [cputype['count']]
            count = cputype['count']
            for i in range (0, count):
                new_resource = Resource ('c' + str(self.cpuid_counter), self, cputype['provision_type'], self.env)
                new_resource.add_cpu(cputype['id'], cputype['cost'], self.env.now)
                self.active_pool_nodes.append(new_resource)
                self.cpuid_counter += 1
                self.active_cpunodes_count += 1

        for gputype in resources['available']['GPU']:
            if gputype['provision_type'] not in self.resourcetypeinfo:
                self.resourcetypeinfo['provision_type'] = {}
            self.resourcetypeinfo[gputype['provision_type']][gputype['id']] = {}
            self.resourcetypeinfo[gputype['provision_type']][gputype['id']]['startuptime'] = gputype['startuptime']
            self.resourcetypeinfo[gputype['provision_type']][gputype['id']]['resourcetype'] = 'GPU'
            self.resourcetypeinfo[gputype['provision_type']][gputype['id']]['availability'] = 1.0
            self.resourcetypeinfo[gputype['provision_type']][gputype['id']]['cost'] = gputype['cost']
            self.resourcetypeinfo[gputype['provision_type']][gputype['id']]['count'] = {}
            self.resourcetypeinfo[gputype['provision_type']][gputype['id']]['count']['time'] = [self.env.now]
            self.resourcetypeinfo[gputype['provision_type']][gputype['id']]['count']['count'] = [gputype['count']]
            count = gputype['count']
            for i in range (0, count):
                new_resource = Resource ('g' + str(self.gpuid_counter), self, gputype['provision_type'], self.env)
                new_resource.add_gpu(gputype['id'], gputype['cost'], self.env.now)
                self.active_pool_nodes.append(new_resource)
                self.gpuid_counter += 1
                self.active_gpunodes_count += 1

        #self.availablenodesdict = self.nodesdict

        return self.active_pool_nodes