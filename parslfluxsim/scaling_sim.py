from parslfluxsim.domain_sim import DomainManager
from parslfluxsim.resources_sim import ResourceManager
from parslfluxsim.pipeline_sim import PipelineManager, PipelineStage
import math

class Scaler:
    def __init__(self, env, interval):
        self.trigger_interval = interval
        self.deadline = None
        self.env = env
        self.last_scaling_timestamp = self.env.now

    def set_deadline (self, deadline):
        self.deadline = self.env.now + deadline

    def reset (self):
        self.last_scaling_timestamp = self.env.now

    def scale_up_deadline (self, rmanager, pmanager, dmanager, allocator):
        if self.env.now - self.last_scaling_timestamp < float (self.trigger_interval) or self.env.now >= self.deadline:
            return

        print('scale_up_deadline ()', self.env.now)

        pmanager.calculate_pipeline_stats(self.env)
        pmanager.calculate_idleness_cost(rmanager, self.env)

        target_throughput_dict = {}
        for pipelinestage in pmanager.pipelinestages:
            if pipelinestage.get_pending_workitems_count () <= 0:
                continue

            time_left = self.deadline - self.env.now

            opt_throughput = pipelinestage.get_pending_workitems_count() / time_left

            current_throughput = pmanager.get_throughput (pipelinestage)

            no_of_scale_ups_left = math.ceil (float(time_left / self.trigger_interval))

            print (pipelinestage.name, pipelinestage.get_pending_workitems_count(), opt_throughput, \
                   current_throughput, time_left, self.env.now, self.deadline, no_of_scale_ups_left)

            if current_throughput >= opt_throughput and no_of_scale_ups_left <= 0:
                continue

            target_throughput = float(opt_throughput - current_throughput) / no_of_scale_ups_left

            target_throughput_dict[pipelinestage.name] = target_throughput

        #print ('scale_up_deadline ()', target_throughput_dict)

        target_throughput_dict = dict (sorted(target_throughput_dict.items(), key=lambda item:item[1], reverse=True))

        for pipelinestagename in target_throughput_dict.keys():
            pipelinestage = pmanager.get_pipelinestage (pipelinestagename)

            allocator.scale_up_resources(rmanager, pmanager, dmanager, pipelinestage, target_throughput_dict[pipelinestagename])

            print (pipelinestagename, pipelinestage.get_current_throughput())

        self.last_scaling_timestamp = self.env.now

    def scale_up_2x (self, rmanager, pmanager, dmanager, allocator):
        if self.env.now - self.last_scaling_timestamp < float (self.trigger_interval):
            return
        print ('scale_up_2x ()', self.env.now)
        for pipelinestage in pmanager.pipelinestages:
            if pipelinestage.get_pending_workitems_count () <= 0:
                continue
            pinned_resource_ids = pipelinestage.get_pinned_resources(rmanager, status=True)

            for pinned_resource_id in pinned_resource_ids:
                resource = rmanager.get_resource (pinned_resource_id)
                cpu_ok = False
                gpu_ok = False
                if pipelinestage.resourcetype == 'CPU':
                    cpu_ok = True
                    resource_type = resource.cpu.name
                else:
                    gpu_ok = True
                    resource_type = resource.gpu.name
                domain = dmanager.get_domain (resource.domain_id)

                allocator.add_worker(rmanager, domain, cpu_ok, gpu_ok, resource_type, \
                                     'on_demand', None, pipelinestage, -1)

        self.last_scaling_timestamp = self.env.now

