import time
from parslfluxsim.FirstCompleteFirstServe_sim import FirstCompleteFirstServe
from parslfluxsim.resources_sim import ResourceManager
from parslflux.pipeline import PipelineManager
from parslfluxsim.input_sim import InputManager2
from execution_sim import ExecutionSim, ExecutionSimThread
import simpy

class OAI_Scheduler:
    def __init__(self, env):
        self.env = env
        self.max_first_stage_size = 30

    def add_worker (self, rmanager, resource, cpuok, gpuok, cputype, gputype):
        new_resource = rmanager.get_new_resource ()
        if resource != None and cpuok == True:
            new_resource.add_cpu (resource.cpu.name)
        if resource != None and gpuok == True:
            new_resource.add_gpu (resource.gpu.name)

        if resource == None:
            if cpuok == True:
                new_resource.add_cpu (cputype)
            if gpuok == True:
                new_resource.add_gpu (gputype)

        if new_resource.cpu != None:
            new_resource.cpu.reinit ()

        if new_resource.gpu != None:
            new_resource.gpu.reinit ()

        if new_resource.cpu != None:
            cpu_thread = ExecutionSimThread(self.env, new_resource, 'CPU', self.performancedata)
        else:
            cpu_thread = None

        if resource.gpu != None:
            gpu_thread = ExecutionSimThread(self.env, new_resource, 'GPU', self.performancedata)
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

    def replenish_workitems (self, imanager, pmanager, scheduling_policy, batchsize):
        first_pipelinestage = pmanager.get_pipelinestage(None, 'CPU')
        if first_pipelinestage == None:
            first_resourcetype = 'GPU'
        else:
            first_resourcetype = 'CPU'

        count = 0
        for i in range (0, batchsize):
            new_workitem = scheduling_policy.create_workitem (imanager, pmanager, None, first_resourcetype)
            if new_workitem == None:
                break
            count += 1

        return count

    def report_idle_periods (self, rmanager, since_time, current_time):
        print ('report_idle_periods ()')
        resources = rmanager.get_resources ()

        for resource in resources:
            cpu_idle_periods, gpu_idle_periods = resource.report_idle_periods (since_time, current_time)

            if cpu_idle_periods != None:
                print ('CPU', cpu_idle_periods)
            if gpu_idle_periods != None:
                print ('GPU', gpu_idle_periods)

            if cpu_idle_periods != None:
                total_idle_period = 0
                for period in cpu_idle_periods:
                    total_idle_period += period[1] - period[0]
                print (resource.id, 'CPU', since_time, current_time, total_idle_period, (total_idle_period / (current_time - since_time) * 100))

            if gpu_idle_periods != None:
                total_idle_period = 0
                for period in gpu_idle_periods:
                    total_idle_period += period[1] - period[0]
                print (resource.id, 'GPU', current_time - since_time, total_idle_period, (total_idle_period / (current_time - since_time) * 100))

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
        resources = rmanager.get_resources ()

        for resource in resources:
            resource.set_idle_start_time ('CPU', now)
            resource.set_idle_start_time ('GPU', now)

    def set_idle_start_times (self, rmanager, now):
        resources = rmanager.get_resources ()

        for resource in resources:
            cpu_idle, gpu_idle = resource.is_idle ()
            if cpu_idle == True and resource.cpu.idle_start_time < now:
                resource.cpu.add_idle_period (now)
                resource.cpu.set_idle_start_time (now)

            if gpu_idle == True and resource.gpu.idle_start_time < now:
                resource.gpu.add_idle_period (now)
                resource.gpu.set_idle_start_time (now)

    def run1 (self, rmanager, imanager, pmanager, batchsize):
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

                #scaling up code goes here

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

                #close the completed phases
                pmanager.close_phases (rmanager, self.env.now)

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

                #predict the execution pattern
                #pmanager.predict_execution(rmanager, pmanager, self.env.now)
                if replenish_done == True:
                    pmanager.predict_execution_fixed (rmanager, pmanager, self.env.now, batchsize)
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
                    self.add_idle_periods (rmanager, self.env.now)
                    self.report_idle_periods(rmanager, last_replenish_time, self.env.now)
                    print ('\n\n\n\n')
                    self.clear_completion_times (rmanager)
                    last_replenish_time = self.env.now
                    replenish_done = True
                    count = self.replenish_workitems(imanager, pmanager, scheduling_policy, batchsize)
                    if count == 0:
                        print('all tasks complete')
                        pmanager.print_stage_queue_data()
                        break

                yield self.env.timeout(5/3600)
        except simpy.Interrupt as i:
            print ('WOW!')

    def run2 (self, rmanager, imanager, pmanager, batchsize):
        print('OAI_scheduler_2 ()', 'waiting for 5 secs')

        scheduling_policy = FirstCompleteFirstServe("FirstCompleteFirstServe", self.env)

        first_pipelinestage = pmanager.get_pipelinestage(None, 'CPU')
        if first_pipelinestage == None:
            first_resourcetype = 'GPU'
        else:
            first_resourcetype = 'CPU'

        resources = rmanager.get_resources()

        while True:
            new_workitem = scheduling_policy.create_workitem (imanager, pmanager, None, first_resourcetype)
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

                #scaling up code goes here

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

                #close the completed phases
                pmanager.close_phases (rmanager, self.env.now)

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

                #predict the execution pattern
                #pmanager.predict_execution(rmanager, pmanager, self.env.now)

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

                yield self.env.timeout(5/3600)
        except simpy.Interrupt as i:
            print ('WOW!')

    def run (self, rmanager, imanager, pmanager, batchsize):
        print('OAI_scheduler_2 ()', 'waiting for 5 secs')

        scheduling_policy = FirstCompleteFirstServe("FirstCompleteFirstServe", self.env)

        resources = rmanager.get_resources()

        first_pipelinestage = pmanager.get_pipelinestage(None, 'CPU')
        if first_pipelinestage == None:
            first_resourcetype = 'GPU'
        else:
            first_resourcetype = 'CPU'

        while True:
            new_workitem = scheduling_policy.create_workitem (imanager, pmanager, None, first_resourcetype)
            if new_workitem == None:
                break

        last_phase_closed_time = self.env.now

        self.set_init_idle_start_times (rmanager, self.env.now)

        try:
            while True:
                for resource in resources:
                    # print ('###########################')
                    resource.get_status(rmanager, pmanager, self.worker_threads[resource.id], self.outputfile)
                    # print ('###########################')
                    # print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    scheduling_policy.remove_complete_workitem(resource, pmanager, self.env)
                    # print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                #scaling up code goes here

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
                last_phase_closed_index = pmanager.close_phases_fixed(rmanager, self.env.now, False)

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

                #predict the execution pattern
                #pmanager.predict_execution(rmanager, pmanager, self.env.now)
                if last_phase_closed_index != None:
                    self.set_idle_start_times (rmanager, self.env.now)
                    self.report_idle_periods(rmanager, last_phase_closed_time, self.env.now)
                    last_phase_closed_time = self.env.now
                    pmanager.predict_execution_fixed (rmanager, pmanager, self.env.now, batchsize, last_phase_closed_index)

                idle_cpus = []
                idle_gpus = []

                for resource in resources:
                    cpu_idle, gpu_idle = resource.is_idle()
                    if cpu_idle == True:
                        idle_cpus.append(resource)
                    if gpu_idle == True:
                        idle_gpus.append(resource)

                #print (len (idle_cpus), len(idle_gpus), rmanager.get_cpu_resources_count(), rmanager.get_gpu_resources_count())
                if len(idle_cpus) == rmanager.get_cpu_resources_count() and len(
                        idle_gpus) == rmanager.get_gpu_resources_count():
                    last_phase_closed_index = pmanager.close_phases_fixed(rmanager, self.env.now, True)
                    self.set_idle_start_times(rmanager, self.env.now)
                    if self.env.now > last_phase_closed_time:
                        self.report_idle_periods(rmanager, last_phase_closed_time, self.env.now)
                    print('all tasks complete')
                    pmanager.print_stage_queue_data()
                    break

                yield self.env.timeout(5/3600)
        except simpy.Interrupt as i:
            print ('WOW!')