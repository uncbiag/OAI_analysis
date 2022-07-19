from parslfluxsim.execution_sim import ExecutionSim, ExecutionSimThread
from parslfluxsim.resources_sim import ResourceManager
from parslfluxsim.domain_sim import DomainManager
class Allocator:
    def __init__(self, env, wait_threshold):
        self.env = env
        self.workers = {}
        self.worker_threads = {}
        self.wait_threshold = wait_threshold

    def scale_up_resources_domain_limit (self, rmanager, pmanager, pipelinestage, target_throughput, domain):

        performance_to_cost_ratio_ranking = pmanager.performance_to_cost_ranking_pipelinestage_domain(rmanager,
                                                                                                      domain,
                                                                                                      pipelinestage.index)

        to_be_added = {}
        throughput_achieved = 0

        for resource_name in performance_to_cost_ratio_ranking.keys():
            resource_throughput = rmanager.get_throughput (resource_name, pipelinestage.name)

            while True:
                if resource_throughput + throughput_achieved > target_throughput:
                    break

                if resource_name not in to_be_added:
                    request_count = 1
                else:
                    request_count = 1 + to_be_added[resource_name]

                if domain.get_availability(resource_name, request_count) == False:
                    break

                if resource_name not in to_be_added:
                    to_be_added[resource_name] = 0

                to_be_added[resource_name] += 1

                throughput_achieved += resource_throughput

            if target_throughput > throughput_achieved:
                continue
            else:
                break

        return to_be_added, throughput_achieved

    def scale_up_resources_domain (self, rmanager, pmanager, pipelinestage, target_throughput, domain):

        performance_to_cost_ratio_ranking = pmanager.performance_to_cost_ranking_pipelinestage_domain(rmanager,
                                                                                                      domain,
                                                                                                      pipelinestage.index)

        to_be_added = {}
        throughput_achieved = 0

        for resource_name in performance_to_cost_ratio_ranking.keys():
            resource_throughput = rmanager.get_throughput (resource_name, pipelinestage.name)

            while True:
                if resource_throughput + throughput_achieved > target_throughput:
                    break

                wait_time = domain.get_availability(resource_name) - self.env.now
                print ('scale_up_resources_domain ()', resource_name, wait_time)

                if wait_time <= self.wait_threshold:
                    if resource_name not in to_be_added:
                        to_be_added[resource_name] = 0
                    to_be_added[resource_name] += 1
                    throughput_achieved += resource_throughput
                else:
                    break

            if target_throughput > throughput_achieved:
                continue
            else:
                break

        return to_be_added, throughput_achieved

    def scale_up_resources (self, rmanager, pmanager, dmanager, pipelinestage, target_throughput):
        hpc_domains = dmanager.get_hpc_domains ()

        cpuok = False
        gpuok = False

        if pipelinestage.resourcetype == 'CPU':
            cpuok = True
        else:
            gpuok = True

        for hpc_domain in hpc_domains:
            print('scale_resources hpc ()', pipelinestage.name, hpc_domain.type)
            to_be_added, throughput_achieved = self.scale_up_resources_domain (rmanager, pmanager, pipelinestage, target_throughput, hpc_domain)

            print ('scale_resources_hpc ()', pipelinestage.name, target_throughput, throughput_achieved, to_be_added)

            target_throughput -= throughput_achieved
            for resource_name in to_be_added.keys ():
                count = to_be_added[resource_name]

                for i in range (0, count):
                    self.add_worker(rmanager, hpc_domain, cpuok, gpuok, resource_name, 'on_demand', None, pipelinestage, -1)

        cloud_domains = dmanager.get_cloud_domains()
        for cloud_domain in cloud_domains:
            print('scale_resources cloud ()', pipelinestage.name, cloud_domain.type)
            to_be_added, throughput_achieved = self.scale_up_resources_domain(rmanager, pmanager, pipelinestage,
                                                                              target_throughput, cloud_domain)

            print('scale_resources_cloud ()', pipelinestage.name, target_throughput, throughput_achieved, to_be_added)

            target_throughput -= throughput_achieved
            for resource_name in to_be_added.keys():
                count = to_be_added[resource_name]

                for i in range(0, count):
                    self.add_worker(rmanager, cloud_domain, cpuok, gpuok, resource_name, 'on_demand', None, pipelinestage, -1)

    def check_failed (self, rmanager, dmanager, pipelinestage, scheduling_policy):
        pinned_resource_ids  = pipelinestage.get_pinned_resources(rmanager, status=True)

        for pinned_resource_id in pinned_resource_ids:
            if pipelinestage.resourcetype == 'CPU':
                worker = self.workers[pinned_resource_id][0]
            else:
                worker = self.workers[pinned_resource_id][1]

            pinned_resource = rmanager.get_resource (pinned_resource_id)

            if worker.status == 'CANCELLED':
                ret = scheduling_policy.remove_failed_workitem(pinned_resource, pipelinestage)
                self.delete_worker(rmanager, dmanager, pinned_resource_id, pipelinestage)


    def add_worker(self, rmanager, domain, cpuok, gpuok, resource_type, provision_type, bidding_price, pipelinestage, reservation_quota):
        if reservation_quota == -1:
            reservation_quota = domain.get_reservation_quota ()

        new_resource, provision_time = rmanager.add_resource(domain, cpuok, gpuok, resource_type, provision_type, \
                                                             bidding_price, pipelinestage.index, reservation_quota)

        if new_resource == None or provision_time == None:
            print ('add_worker ()', resource_type, provision_type, 'not available')
            return  None

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
            cpu_thread_exec = ExecutionSim(self.env, cpu_thread, reservation_quota, new_resource.id)
        else:
            cpu_thread_exec = None
        if gpu_thread != None:
            gpu_thread_exec = ExecutionSim(self.env, gpu_thread, reservation_quota, new_resource.id)
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

        print ('delete_worker', resource_id)
        worker.cancel_reservation ()

        '''        
        worker_exec = worker.get_exec()

        if worker_exec.is_alive == True:
            worker_exec.interrupt('cancel')
        '''

        self.worker_threads.pop(resource_id, None)

        self.workers.pop(resource_id, None)

        resource = rmanager.get_resource(resource_id)
        domain = dmanager.get_domain(resource.domain_id)

        rmanager.delete_resource(domain, pipelinestage.resourcetype, resource, active=True)

        pipelinestage.remove_pinned_resource(resource_id)