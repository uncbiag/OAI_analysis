import yaml
import sys

from parslflux.resources import ResourceManager, Resource 

class PipelineStage:
    def __init__ (self, index, pipelinestage):
        self.index = index
        self.name = pipelinestage['name']
        self.latency = {}
        self.resourcetype = pipelinestage['resource']
        self.budget = 0

    def add_latency (self, latency, cpu, gpu):
        if self.resourcetype == 'CPU':
            self.latency[cpu] = latency
        else:
            self.latency[gpu] = latency

    def get_latency (self, cpu, gpu):
        if cpu in self.latency.keys ():
            return self.latency[cpu]
        elif gpu in self.latency.keys ():
            return self.latency[gpu]

    def get_resource (self):
        return self.resourcetype

    def get_name (self):
        return self.name

    def get_budget (self):
        return self.budget

    def set_budget (self, budget):
        self.budget = budget

    def print_data (self):
        print (self.index, self.name, self.latency, self.resourcetype)

class PipelineManager:
    def __init__ (self, pipelinefile, budget):
        self.pipelinefile = pipelinefile
        self.pipelinestages = []
        self.pipelineindex = 0
        self.budget = budget
        self.base_total_gpu_exec_time = 0
        self.base_total_cpu_exec_time = 0

    def parse_pipelines (self):
        pipelinedatafile = open (self.pipelinefile)
        pipelinedata = yaml.load (pipelinedatafile, Loader = yaml.FullLoader)

        #parse pipeline stages
        index = 0
        for pipelinestage in pipelinedata['pipelinestages']:
            self.pipelinestages.append (PipelineStage(index, pipelinestage))
            index = index + 1

        self.basecpu = pipelinedata['performance']['basecpu']
        self.basegpu = pipelinedata['performance']['basegpu']

        for performance in pipelinedata['performance']['latencies']:
            cpu = performance['cpu']
            gpu = performance['gpu']
            index = 0
            print (performance['latency'])
            for latency in performance['latency']:
                self.pipelinestages[index].add_latency (latency, cpu, gpu)
                index = index + 1

        for pipelinestage in self.pipelinestages:
            if pipelinestage.get_resource () == 'CPU':
                self.base_total_cpu_exec_time += pipelinestage.get_latency (self.basecpu, self.basegpu)
            else:
                self.base_total_gpu_exec_time += pipelinestage.get_latency (self.basecpu, self.basegpu)

        self.base_total_exec_time = self.base_total_cpu_exec_time + self.base_total_gpu_exec_time

    def get_base_total_exec_time (self):
        return self.base_total_exec_time

    def get_base_total_cpu_exec_time (self):
        return self.base_total_cpu_exec_time

    def get_base_total_gpu_exec_time (self):
        return self.base_total_gpu_exec_time

    def encode_pipeline_stages (self, pipelinestages):
        output = ""
        index = 0
        for pipelinestage in pipelinestages:
            if index == len (pipelinestages) - 1:
                output += str (pipelinestage.get_name())
            else:
                output += str (pipelinestage.get_name())
                output += ":"
            index += 1
        return output

    def performance_extrapolate (self, rm):
        basecpurating = rm.get_cpurating_type (self.basecpu)
        print (self.basecpu, self.basegpu)
        basegpurating = rm.get_gpurating_type (self.basegpu)

        for pipelinestage in self.pipelinestages:
            for node in rm.get_resources ():
                if pipelinestage.get_resource () == 'CPU':
                    nodecpurating = node.get_cpurating ()

                    base_cpulatency = pipelinestage.get_latency (self.basecpu, '')
                    node_cpulatency = basecpurating * base_cpulatency / nodecpurating
                    pipelinestage.add_latency (node_cpulatency, node.get_name (),
                                               node.get_gpuname ())
                    node.add_latency (self, list(pipelinestage), node_cpulatency)

                elif pipelinestage.get_resource () == 'GPU':
                    if node.get_gpuname () != '':
                        nodegpurating = node.get_gpurating ()

                        base_gpulatency = pipelinestage.get_latency ('', self.basegpu)
                        print (basegpurating, base_gpulatency, nodegpurating)
                        node_gpulatency = basegpurating * base_gpulatency / nodegpurating
                        pipelinestage.add_latency (node_gpulatency, node.get_name (),
                                                   node.get_gpuname ())
                        node.add_latency (self, list(pipelinestage), node_gpulatency)
                            
                            
    def get_pipelinestage (self):
        pipelinestage = self.pipelinestages[self.pipelineindex]
        self.pipelineindex = self.pipelineindex + 1
        if self.pipelineindex == len(self.pipelinestages):
            self.pipelineindex = 0
            return pipelinestage, True #end of pipeline
        return pipelinestage, False

    def add_back_pipelinestage (self):
        if self.pipelineindex == 0:
            self.pipelineindex = len(self.pipelinestages) - 1
        else:
            self.pipelineindex = self.pipelineindex - 1

    def print_data (self):
        for pipelinestage in self.pipelinestages:
            pipelinestage.print_data ()

if __name__ == "__main__":
    pipelinefile = sys.argv[1]
    p = PipelineManager(pipelinefile)
    p.parse_pipelines ()
    p.print_data ()
