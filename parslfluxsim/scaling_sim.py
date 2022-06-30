from parslfluxsim.domain_sim import DomainManager
from parslfluxsim.resources_sim import ResourceManager
from parslflux.pipeline import PipelineManager, PipelineStage

class Scaler:
    def __init__(self, env, interval):
        self.trigger_interval = interval #in minutes
        self.env = env
        self.last_scaling_timestamp = self.env.now

    def reset (self):
        self.last_scaling_timestamp = self.env.now

    def scale_up_2x (self, rmanager, pmanager, dmanager, allocator):
        if self.env.now - self.last_scaling_timestamp < float (self.trigger_interval / 60):
            return
        print ('scale_up_2x ()', self.env.now)
        for pipelinestage in pmanager.pipelinestages:
            if pipelinestage.get_pending_workitems_count () <= 0:
                continue
            pinned_resource_ids = pipelinestage.get_pinned_resources(rmanager, status=True)

            for pinned_resource_id in pinned_resource_ids:
                resource = rmanager.get_resource (pinned_resource_id, active=True)
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

