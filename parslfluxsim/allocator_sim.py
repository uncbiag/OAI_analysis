from parslfluxsim.execution_sim import ExecutionSim, ExecutionSimThread
from parslfluxsim.resources_sim import ResourceManager

class Allocator:
    def __init__(self, env):
        self.env = env
        self.workers = {}
        self.worker_threads = {}

    def get_status (self, rmanager, dmanager, pipelinestage, scheduling_policy):
        pinned_resource_ids  = pipelinestage.get_pinned_resources(rmanager, status=True)

        for pinned_resource_id in pinned_resource_ids:
            if pipelinestage.resourcetype == 'CPU':
                worker = self.workers[pinned_resource_id][0]
            else:
                worker = self.workers[pinned_resource_id][1]

            pinned_resource = rmanager.get_resource (pinned_resource_id, True)

            if worker.status == 'CANCELLED':
                ret = scheduling_policy.remove_failed_workitem(pinned_resource, pipelinestage)
                self.delete_worker(rmanager, dmanager, pinned_resource_id, pipelinestage)


    def add_worker(self, rmanager, domain, cpuok, gpuok, resource_type, provision_type, bidding_price, pipelinestage, reservation_quota):
        if provision_type == 'on_demand':
            activepool = True
        else:
            activepool = False

        if reservation_quota == -1:
            reservation_quota = domain.get_reservation_quota ()

        new_resource, provision_time = rmanager.add_resource(domain, cpuok, gpuok, resource_type, provision_type, \
                                                             activepool, bidding_price, pipelinestage.index, reservation_quota)

        # print ('add_worker ()', new_resource.cpu, new_resource.gpu, provision_time)

        performance_dist = domain.get_performance_dist(resource_type)

        if performance_dist == None:
            print('add_worker ()', resource_type, 'performance dance not found')
            return None

        if new_resource.cpu != None:
            cpu_thread = ExecutionSimThread(self.env, new_resource, 'CPU', performance_dist, provision_type, provision_time,
                                            domain.id)
        else:
            cpu_thread = None

        if new_resource.gpu != None:
            gpu_thread = ExecutionSimThread(self.env, new_resource, 'GPU', performance_dist, provision_type, provision_time,
                                            domain.id)
        else:
            gpu_thread = None

        self.worker_threads[new_resource.id] = [cpu_thread, gpu_thread]

        if cpu_thread != None:
            cpu_thread_exec = ExecutionSim(self.env, cpu_thread, reservation_quota)
        else:
            cpu_thread_exec = None
        if gpu_thread != None:
            gpu_thread_exec = ExecutionSim(self.env, gpu_thread, reservation_quota)
        else:
            gpu_thread_exec = None

        self.workers[new_resource.id] = [cpu_thread_exec, gpu_thread_exec]

        pipelinestage.add_pinned_resource(new_resource.id)

        return new_resource


    def delete_worker(self, rmanager, dmanager, resource_id, pipelinestage):
        if pipelinestage.resourcetype == 'CPU':
            worker = self.workers[resource_id][0]
        else:
            worker = self.workers[resource_id][1]

        worker_exec = worker.get_exec()

        if worker_exec.is_alive == True:
            worker_exec.interrupt('cancel')

        self.worker_threads.pop(resource_id, None)

        self.workers.pop(resource_id, None)

        resource = rmanager.get_resource(resource_id, True)
        domain = dmanager.get_domain(resource.domain_id)

        rmanager.delete_resource(domain, pipelinestage.resourcetype, resource, active=True)

        pipelinestage.remove_pinned_resource(resource_id)