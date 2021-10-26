import time
from parslfluxsim.FirstCompleteFirstServe_sim import FirstCompleteFirstServe
from parslfluxsim.resources_sim import ResourceManager
from parslflux.pipeline import PipelineManager
from parslfluxsim.input_sim import InputManager2
import simpy

class OAI_Scheduler:
    def __init__(self, env):
        self.env = env

    def run (self, rmanager, imanager, pmanager):
        print('OAI_scheduler_2 ()', 'waiting for 5 secs')

        scheduling_policy = FirstCompleteFirstServe("FirstCompleteFirstServe", self.env)

        resources = rmanager.get_resources()

        try:
            while True:
                for resource in resources:
                    # print ('###########################')
                    resource.get_status(pmanager, self.worker_threads[resource.id], self.outputfile)
                    # print ('###########################')
                    # print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    scheduling_policy.remove_complete_workitem(resource)
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
                    break

                yield self.env.timeout(5/3600)
        except simpy.Interrupt as i:
            print ('WOW!')