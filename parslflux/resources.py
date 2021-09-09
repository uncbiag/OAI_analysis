import yaml
import sys
import operator
import copy
import datetime

from parslflux.workitem import WorkItem
from parslflux.workqueue import WorkItemQueue

class CPU:
    def __init__ (self, node):
        self.name = node['name']
        self.workqueue = WorkItemQueue ()
        self.busy = False
        self.last_completion_time = None

    def get_name (self):
        return self.name

    def is_busy (self):
        return self.busy

    def set_busy (self, busy):
        self.busy = busy

    def set_last_completion_time (self, time):
        self.last_completion_time = time

    def get_last_completion_time (self):
        if self.last_completion_time == None:
            return None

        return datetime.datetime.strptime (self.last_completion_time, '%Y-%m-%d %H:%M:%S')

class GPU:
    def __init__ (self, gpu):
        self.name = gpu['name']
        self.workqueue = WorkItemQueue ()
        self.busy = False
        self.last_completion_time = None

    def get_name (self):
        return self.name

    def is_busy (self):
        return self.busy

    def set_busy (self, busy):
        self.busy = busy

    def set_last_completion_time (self, time):
        self.last_completion_time = time

    def get_last_completion_time (self):
        if self.last_completion_time == None:
            return None

        return datetime.datetime.strptime (self.last_completion_time, '%Y-%m-%d %H:%M:%S')

class Resource:

    def __init__ (self, i):
        self.id = "c" + str(i)
        self.hostname = "c" + str (i)
        self.cpu = None
        self.gpu = None

    def add_cpu (self, cpu):
        self.cpu = CPU (cpu)

    def add_gpu (self, gpu):
        self.gpu = GPU (gpu)

    def is_idle (self):
        cpu_free = False
        if self.cpu != None:
            if self.cpu.is_busy () == False:
                cpu_free = True

        gpu_free = False
        if self.gpu != None:
            if self.gpu.is_busy () == False:
                gpu_free = True

        print ('is_idle ():', self.id, cpu_free, gpu_free)

        return cpu_free, gpu_free

    def is_empty (self):
        cpu_empty = False

        if self.cpu.workqueue.is_empty () == True:
            cpu_empty = True

        gpu_empty = False

        if self.gpu.workqueue.is_empty () == True:
            gpu_empty = True

        print ('is_empty ():', self.id, cpu_empty, gpu_empty)

        return cpu_empty, gpu_empty

    def schedule (self, pmanager, resourcetype):
        if resourcetype == 'CPU' and self.cpu == None:
            print (self.id, 'CPU not available')
            return
        if resourcetype == 'GPU' and self.gpu == None:
            print (self.id, 'GPU not available')
            return

        if resourcetype == 'CPU' and self.cpu.workqueue.is_empty () == False:
            self.cpu.workqueue.get_workitem ().submit (pmanager)
            self.cpu.set_busy (True)
            self.cpu.set_last_completion_time (None)
        else:
            print (self.id, 'CPU no workitem available to schedule')

        if resourcetype == 'GPU' and self.gpu.workqueue.is_empty () == False:
            self.gpu.workqueue.get_workitem ().submit (pmanager)
            self.gpu.set_busy (True)
            self.gpu.set_last_completion_time (None)
        else:
            print (self.id, 'GPU no workitem available to schedule')

    def get_status (self, pmanager):
        print ('get_status ():', self.id)
        #first cpu
        if self.cpu != None and self.cpu.workqueue.is_empty () == False:
            workitem = self.cpu.workqueue.get_workitem ()
            ret, start_time, end_time = workitem.probe_status ()
            if ret == True:
                print ('cpu workitem complete')
                self.cpu.set_busy (False)
                self.cpu.set_last_completion_time (end_time)
                self.add_count (pmanager.encode_pipeline_stages (self.pipelinestages))
                self.add_exectime (pmanager.encode_pipeline_stages(self.pipelinestages), starttime, endtime)


        #now gpu
        if self.gpu != None and self.gpu.workqueue.is_empty () == False:
            workitem = self.gpu.workqueue.get_workitem ()
            ret, start_time, end_time = workitem.probe_status ()
            if ret == True:
                print ('gpu workitem complete')
                self.gpu.set_busy (False)
                self.gpu.set_last_completion_time (end_time)
                self.add_count (pmanager.encode_pipeline_stages (self.pipelinestages))
                self.add_exectime (pmanager.encode_pipeline_stages(self.pipelinestages), starttime, endtime)

    def get_last_completion_time (self, resourcetype):
        if resourcetype == 'CPU' and self.cpu != None:
            return self.cpu.get_last_completion_time ()

        if resourcetype == 'GPU' and self.gpu != None:
            return self.gpu.get_last_completion_time ()


    def add_workitem (self, workitem, resourcetype):
        print ('add_workitem ()', self.id)
        if resourcetype == 'CPU':
            if self.cpu == None:
                print (self.id, 'CPU not available')
                return
            self.cpu.workqueue.add_workitem (workitem)

        if resourcetype == 'GPU':
            if self.gpu == None:
                print (self.id, 'GPU not available')
                return
            self.gpu.workqueue.add_workitem (workitem)


    def pop_if_complete (self, resourcetype):
        if resourcetype == 'CPU' and self.cpu == None:
            print (self.id, 'CPU not available')
            return None

        if resourcetype == 'GPU' and self.cpu == None:
            print (self.id, 'GPU not available')
            return None

        if resourcetype == 'CPU' and self.cpu.workqueue.is_empty () == False:
            if self.cpu.workqueue.get_workitem ().is_complete () == True:
                print (self.id, 'CPU workitem complete')
                workitem = self.cpu.workqueue.pop_workitem ()
                return workitem

        if resourcetype == 'GPU' and self.gpu.workqueue.is_empty () == False:
            if self.gpu.workqueue.get_workitem ().is_complete () == True:
                print (self.id, 'GPU workitem complete')
                workitem = self.gpu.workqueue.pop_workitem ()
                return workitem

        print (self.id, resourcetype, 'not complete')
        return None

    def get_hostname (self):
        return self.hostname

    def get_count (self, pipelinestages):
        if pipelinestages not in self.counts:
            return 0
        return self.counts[pipelinestages]

    def add_count (self, pipelinestages):
        if pipelinestages in self.counts:
            self.counts[pipelinestages] += 1
        else:
            self.counts[pipelinestages] = 1

    def add_exectime (self, pipelinestages, starttime_s, endtime_s):
        starttime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        endtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timediff = endtime - starttime
        seconds = timediff.total_seconds ()

        if pipelinestages not in self.exectime:
            self.exectimes[pipelinestages] = [seconds, 1]
        else:
            avg_time = self.exectimes[pipelinestages][0]
            count = self.exectimes[pipelinestages][1]
            new_avg_time = ((avg_time * count) + seconds) / (count + 1)
            self.exectimes[pipelinestages] = [new_avg_time, count + 1]

    def get_exectime (self, pipelinestages):
        if pipelinestage not in self.exectimes:
            return 0
        else:
            return self.exectimes[pipelinestages][0]

class ResourceManager:
    def __init__ (self, resourcefile, availablefile):
        self.resourcefile = resourcefile
        self.availablefile = availablefile
        self.nodes = []
        self.reservednodes = []
        self.nodesdict = {}
        self.reservednodesdict = {}

    def parse_resources (self):
        yaml_resourcefile = open (self.resourcefile)
        resources = yaml.load (yaml_resourcefile, Loader = yaml.FullLoader)

        arc_resources = resources['arc']
        self.start = arc_resources['range'][0]
        self.end = arc_resources['range'][1]

        #parse nodes
        for node in arc_resources['nodes']:
            for noderange in node['range']:
                for i in range (noderange[0], noderange[1] + 1):
                    new_resource = Resource (i)
                    new_resource.add_cpu (node)
                    self.nodesdict[str(i)] = new_resource

        #parse gpus
        for gpu in arc_resources['gpus']:
            for gpurange in gpu['range']:
                for i in range (gpurange[0], gpurange[1] + 1):
                    self.nodesdict[str(i)].add_gpu (gpu)

    def purge_resources (self):
        available_resourcefile = open (self.availablefile)
        availableresources = yaml.load (available_resourcefile, Loader = yaml.FullLoader)
        resources = {}
        reservedresources = {}

        if len (availableresources['available']) > 0:
            for i in availableresources['available']:
                resources[str(i)] = copy.deepcopy (self.nodesdict[str(i)])

        if len (availableresources['reserved']) > 0:
            for i in availableresources['reserved']:
                reservedresources[str(i)] = copy.deepcopy (self.nodesdict[str(i)])

        self.nodesdict = resources
        self.reservednodesdict = reservedresources

        self.nodes = copy.deepcopy (list (self.nodesdict.values ()))
        self.reservednodes = copy.deepcopy (list (self.reservednodesdict.values ()))

        print ('nodes:', len(self.nodes), 'reserved nodes:', len (self.reservednodes))

    def get_resource (self, resource_id):
        for node in self.nodes:
            if node.id == resource_id:
                return node
        return None

    def get_resources (self):
        return self.nodes

    def request_reserved_resource (self):
        if len (self.reservednodesdict.keys()) > 0:
            new_key = self.reservednodesdict.keys()[0]
            new_resource = self.reservednodesdict.pop (new_key)
            self.nodesdict[new_key] = new_resource
            self.nodes.append (new_resource)
            return new_resource

        return None
