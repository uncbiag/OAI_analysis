import yaml
import sys
import operator
import copy

class Resource:
    def __init__ (self, node, i):
        self.id = "c" + str(i)
        self.hostname = "c" + str (i)
        self.name = node['name']
        self.RAM = node['RAM']
        self.cpurating = node['rating']['avg']
        self.gpuname = ''
        self.available = True
        self.timetowait = 0
        self.cpucost = node['cpucost']
        self.latencies = {}
        self.workerid = ''
        self.main_taskset = {}
        self.support_taskset = {}
        self.current_taskset = {}
        self.cpuchunksize = {}
        self.gpuchunksize = {}
        self.main_iteration = 0
        self.cputype = True
        self.gputype = False
        if 'SSD' in node:
            for ssdnode in node['SSD']:
                if 'range' in ssdnode:
                    for ssdnoderange in ssdnode['range']:
                        if ssdnoderange[0] <= i and i <= ssdnoderange[1]:
                            self.ssdname = ssdnode['name']
                            self.ssdsize = ssdnode['size']
                            self.ssdlocation = ssdnode['location']
                            return
                else:
                    self.ssdname = ssdnode['name']
                    self.ssdsize = ssdnode['size']
                    self.ssdlocation = ssdnode['location']
        else:
            self.ssdname = ''
            self.ssdsize = 0
            self.ssdlocation = ''

    def set_main_taskset (self, iteration, tasksetid, resourcetype):
        if resourcetype == 'CPU':
            if 'GPU' in self.main_taskset.keys ():
                print ('set_main_taskset (): delete M-GPU')

                self.main_taskset.pop ('GPU')
                if self.current_taskset['GPU'] == 'MAIN':
                    self.current_taskset.pop ('GPU')

            if 'CPU' in self.support_taskset.keys ():
                print ('set_main_taskset (): delete S-CPU')
                self.support_taskset.pop ('CPU')

        if resourcetype == 'GPU':
            if 'CPU' in self.main_taskset.keys ():
                print ('set_main_taskset (): delete M-CPU')

                self.main_taskset.pop ('CPU')
                if self.current_taskset['CPU'] == 'MAIN':
                    self.current_taskset.pop ('CPU')

            if 'GPU' in self.support_taskset.keys ():
                print ('set_main_taskset (): delete S-GPU')
                self.support_taskset.pop ('GPU')

        self.main_taskset[str(resourcetype)] = [iteration, tasksetid]

    def get_main_taskset (self, resourcetype):
        if resourcetype not in self.main_taskset.keys():
            return None
        return self.main_taskset[str(resourcetype)]

    def set_support_taskset (self, iteration, tasksetid, resourcetype):
        self.support_taskset[str(resourcetype)] = [iteration, tasksetid]

    def get_support_taskset (self, resourcetype):
        if resourcetype not in self.support_taskset.keys():
            return None
        return self.support_taskset[str(resourcetype)]

    def set_current_taskset (self, resourcetype, iteration): # 'cpu/gpu:support/main'
        self.current_taskset[resourcetype] = iteration

    def get_current_taskset (self, resourcetype):
        if resourcetype not in self.current_taskset.keys():
            return None, None
        if self.current_taskset[resourcetype] == 'MAIN':
            return 'MAIN', self.get_main_taskset(resourcetype)
        else:
            return 'SUPPORT', self.get_support_taskset(resourcetype)

    def set_worker_id (self, workerid):
        self.workerid = workerid

    def get_worker_id (self):
        return self.workerid

    def add_gpu (self, gpu):
        self.gpuname = gpu['name']
        self.gpurating = gpu['rating']
        self.gpucost = gpu['gpucost']
        self.gputype = True

    def get_name (self):
        return self.name

    def get_gpuname (self):
        return self.gpuname

    def get_gpurating (self):
        return self.gpurating

    def set_gpurating (self, rating):
        self.gpurating = rating

    def get_cpurating (self):
        return self.cpurating

    def set_cpurating (self, rating):
        self.cpurating = rating

    def get_chunksize (self, resourcetype, pipelinestages):
        if resourcetype == 'CPU':
            if pipelinestages not in self.cpuchunksize.keys():
                self.cpuchunksize[pipelinestages] = 1
            return self.cpuchunksize[pipelinestages]
        else:
            if pipelinestages not in self.gpuchunksize.keys():
                self.gpuchunksize[pipelinestages] =1
            return self.gpuchunksize[pipelinestages]

    def set_chunksize (self, resourcetype, pipelinestages, chunksize):
        if resourcetype == 'CPU':
            self.cpuchunksize[pipelinestages] = chunksize
        else:
            self.cpuchunksize[pipelinestages] = chunksize

    def set_availability (self, availability):
        self.available = availability

    def get_availability (self, availability):
        return self.available

    def set_timetowait (self, timetowait):
        self.timetowait = timetowait

    def get_timetowait (self):
        return self.timetowait

    def get_cpucost (self):
        return self.cpucost

    def set_cpucost (self, cpucost):
        self.cpucost = cpucost

    def get_gpucost (self):
        return self.gpucost

    def set_gpucost (self, gpucost):
        self.gpucost = gpucost

    def get_hostname (self):
        return self.hostname

    def add_latency (self, pmanager, pipelinestages, latency):
        encoded_pipelinestage = pmanager.encode_pipeline_stages(pipelinestages)

        if encoded_pipelinestage in self.latencies.keys():
            self.latencies[encoded_pipelinestage].append (latency)
        else:
            self.latencies[encoded_pipelinestage] = []
            self.latencies[encoded_pipelinestage].append (latency)

    def get_latency (self, pmanager, pipelinestages):
        encoded_pipelinestage = pmanager.encode_pipeline_stages(pipelinestages)
        if encoded_pipelinestage in self.latencies.keys():
            total_latency = 0
            latencies = self.latencies[encoded_pipelinestage]
            for latency in latencies:
                total_latency += latency
            return total_latency / len(latencies)
        else:
            return -1

    def print_data (self):
        print (self.id, self.name, self.gpuname, self.cpucost, self.gpucost, self.cpurating, self.gpurating)
        #print (self.latencies)

class ResourceManager:
    def __init__ (self, resourcefile, availablefile):
        self.resourcefile = resourcefile
        self.availablefile = availablefile
        self.nodes = []
        self.reservednodes = []
        self.nodescpuratings = []
        self.nodesgpuratings = []
        self.nodescpucost = []
        self.nodesgpucost = []
        self.nodesdict = {}
        self.reservednodesdict = {}
        self.maxcpucost = 0
        self.maxgpucost = 0
        self.total_gpus = 0

    def parse_resources (self):
        yaml_resourcefile = open (self.resourcefile)
        print (yaml.__version__)
        resources = yaml.load (yaml_resourcefile, Loader = yaml.FullLoader)

        arc_resources = resources['arc']
        self.start = arc_resources['range'][0]
        self.end = arc_resources['range'][1]

        #parse nodes
        for node in arc_resources['nodes']:
            for noderange in node['range']:
                for i in range (noderange[0], noderange[1] + 1):
                    self.nodesdict[i] = Resource (node, i)

        #parse gpus
        for gpu in arc_resources['gpus']:
            for gpurange in gpu['range']:
                for i in range (gpurange[0], gpurange[1] + 1):
                    self.nodesdict[i].add_gpu (gpu)
                    self.total_gpus += 1

    def purge_resources (self):
        available_resourcefile = open (self.availablefile)
        availableresources = yaml.load (available_resourcefile, Loader = yaml.FullLoader)
        resources = {}
        reservedresources = {}

        if len (availableresources['available']) > 0:
            for i in availableresources['available']:
                resources[i] = copy.deepcopy (self.nodesdict[i])

        if len (availableresources['reserved']) > 0:
            for i in availableresources['reserved']:
                reservedresources[i] = copy.deepcopy (self.nodesdict[i])

        self.nodesdict = resources
        self.reservednodesdict = reservedresources

    def normalize (self):
        maxcpurating = 0
        maxgpurating = 0
        maxcost = 0
        for i in self.nodesdict.keys ():
            if self.nodesdict[i].get_gpurating () > maxgpurating:
                maxgpurating = self.nodesdict[i].get_gpurating ()
            if self.nodesdict[i].get_cpurating () > maxcpurating:
                maxcpurating = self.nodesdict[i].get_cpurating ()
            if self.nodesdict[i].get_cpucost () > maxcost:
                maxcost = self.nodesdict[i].get_cpucost ()
            if self.nodesdict[i].get_gpucost () > maxcost:
                maxcost = self.nodesdict[i].get_gpucost ()

        self.maxcost = maxcost

        for i in self.nodesdict.keys ():
            self.nodesdict[i].set_gpurating (self.nodesdict[i].get_gpurating () / maxgpurating * 100)
            self.nodesdict[i].set_cpurating (self.nodesdict[i].get_cpurating () / maxcpurating * 100)
            self.nodesdict[i].set_cpucost (self.nodesdict[i].get_cpucost () / maxcost * 100)
            self.nodesdict[i].set_gpucost (self.nodesdict[i].get_gpucost () / maxcost * 100)


        #TODO: normalize for reserved nodes

        self.nodes = copy.deepcopy (list (self.nodesdict.values ()))

        print (len(self.nodes), self.total_gpus)

    def sort_by_cpu_ratings (self):
        self.nodescpuratings = sorted (self.nodes, key=lambda x: x.cpurating)

    def sort_by_gpu_ratings (self):
        self.nodesgpuratings = sorted (self.nodes, key=lambda x: x.gpurating)

    def sort_by_cpucost (self):
        self.nodescpucost = sorted (self.nodes, key=lambda x: x.cpucost)

    def sort_by_gpucost (self):
        self.nodesgpucost = sorted (self.nodes, key=lambda x: x.gpucost)

    def get_resource (self, resource_id):
        for node in self.nodes:
            if node.id == resource_id:
                return node
        return None

    def request_reserved_resource (self):
        if len (self.reservednodesdict.keys()) > 0:
            new_key = self.reservednodesdict.keys()[0]
            new_resource = self.reservednodesdict.pop (new_key)
            self.nodesdict[new_key] = new_resource
            self.nodes.append (new_resource)
            return new_resource

        return None

    def get_resources (self):

        return self.nodes
        '''
        if resourcetype == 'CPU':
            return copy.deepcopy(self.nodes)
        else:
            gpuresources = []
            for resource in self.nodes:
                if resource.get_gpuname() == '':
                    continue
                gpuresources.append(resource)
            return copy.deepcopy(gpuresources)
        '''

    def get_cpurating_type (self, cputype):
        for key in self.nodesdict.keys ():
            if self.nodesdict[key].get_name () == cputype:
                return self.nodesdict[key].get_cpurating ()

    def get_gpurating_type (self, gputype):
        for key in self.nodesdict.keys ():
            if self.nodesdict[key].get_gpuname () == gputype:
                return self.nodesdict[key].get_gpurating ()

    def get_latency (self, resourceid, pipelinestages):
        return self.nodesdict[resourceid].get_latency (pipelinestages)

    def print_data (self):
        for i in self.nodes:
            i.print_data ()
        print ("###################")
        for i in self.nodescpuratings:
            i.print_data ()
        print ("###################")
        for i in self.nodesgpuratings:
            i.print_data ()
        print ("###################")
        for i in self.nodescpucost:
            i.print_data ()
        print ("###################")
        for i in self.nodesgpucost:
            i.print_data ()

if __name__ == "__main__":
    resourcefile = sys.argv[1]
    availablefile = sys.argv[2]
    r = ResourceManager (resourcefile, availablefile)
    r.parse_resources ()
    r.purge_resources ()
    r.normalize ()
    r.sort_by_cpu_ratings ()
    r.sort_by_gpu_ratings ()
    r.sort_by_cpucost ()
    r.sort_by_gpucost ()
    r.print_data ()
