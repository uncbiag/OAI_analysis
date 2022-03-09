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