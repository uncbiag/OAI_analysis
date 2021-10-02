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

        return self.last_completion_time

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

        return self.last_completion_time

class Resource:

    def __init__ (self, i):
        self.id = "c" + str(i)
        self.hostname = "c" + str (i)
        self.cpu = None
        self.gpu = None
        self.exectimes = {}
        self.max_exectimes = {}
        self.counts = {}

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

    def schedule (self, rmanager, pmanager, resourcetype):
        if resourcetype == 'CPU' and self.cpu == None:
            print (self.id, 'CPU not available')
            return
        if resourcetype == 'GPU' and self.gpu == None:
            print (self.id, 'GPU not available')
            return

        if resourcetype == 'CPU' and self.cpu.workqueue.is_empty () == False:
            timeout = self.get_timeout_value (rmanager, pmanager, resourcetype)
            self.cpu.workqueue.get_workitem ().submit (pmanager, timeout)
            self.cpu.set_busy (True)
            self.cpu.set_last_completion_time (None)
            return
        else:
            print (self.id, 'CPU no workitem available to schedule')

        if resourcetype == 'GPU' and self.gpu.workqueue.is_empty () == False:
            timeout = self.get_timeout_value (rmanager, pmanager, resourcetype)
            self.gpu.workqueue.get_workitem ().submit (pmanager, timeout)
            self.gpu.set_busy (True)
            self.gpu.set_last_completion_time (None)
            return
        else:
            print (self.id, 'GPU no workitem available to schedule')

    def get_timeout_value (self, rmanager, pmanager, resourcetype):
        print ('get_timeout_value ():', self.id)

        if resourcetype == 'CPU' and self.cpu != None and self.cpu.workqueue.is_empty () == False:
            workitem = self.cpu.workqueue.get_workitem ()
            workitem_pipelinestages = workitem.get_pipelinestages ()
            encoded_workitem_pipelinestages = pmanager.encode_pipeline_stages (workitem_pipelinestages)

            max_exectime = self.get_max_exectime (encoded_workitem_pipelinestages) * 2

            if max_exectime == 0:#first time execution
                max_exectime = rmanager.get_max_exectime (encoded_workitem_pipelinestages, self.id) * 2

            if max_exectime == 0:#no one has completed their execution
                max_exectime = 20 * 60

            return max_exectime

        if resourcetype == 'GPU' and self.gpu != None and self.gpu.workqueue.is_empty () == False:
            workitem = self.gpu.workqueue.get_workitem ()
            workitem_pipelinestages = workitem.get_pipelinestages ()
            encoded_workitem_pipelinestages = pmanager.encode_pipeline_stages (workitem_pipelinestages)

            max_exectime = self.get_max_exectime (encoded_workitem_pipelinestages) * 2

            if max_exectime == 0:#first time execution
                max_exectime = rmanager.get_max_exectime (encoded_workitem_pipelinestages, self.id) * 2

            if max_exectime == 0:#no one has completed their execution
                max_exectime = 15 * 60

            return max_exectime

    def get_status (self, pmanager):
        print ('get_status ():', self.id)
        #first cpu
        if self.cpu != None and self.cpu.workqueue.is_empty () == False:
            workitem = self.cpu.workqueue.get_workitem ()
            ret, start_time, end_time, status = workitem.probe_status ()
            if ret == True:
                if status == 'SUCCESS':
                    print ('cpu workitem complete')
                    self.cpu.set_busy (False)
                    self.cpu.set_last_completion_time (end_time)
                    self.add_count (pmanager.encode_pipeline_stages (workitem.get_pipelinestages ()))
                    self.add_exectime (pmanager.encode_pipeline_stages(workitem.get_pipelinestages ()), start_time, end_time)
                elif status == 'FAILED':
                    print ('cpu workitem failed')
                    self.cpu.set_busy (False)
                    self.cpu.set_last_completion_time (end_time)
                elif status == 'CANCELLED':
                    print ('cpu workitem cancelled')
                    self.cpu.set_busy (False)
                    self.cpu.set_last_completion_time (end_time)

        #now gpu
        if self.gpu != None and self.gpu.workqueue.is_empty () == False:
            workitem = self.gpu.workqueue.get_workitem ()
            ret, start_time, end_time, status = workitem.probe_status ()
            if ret == True:
                if status == 'SUCCESS':
                    print ('gpu workitem complete')
                    self.gpu.set_busy (False)
                    self.gpu.set_last_completion_time (end_time)
                    self.add_count (pmanager.encode_pipeline_stages (workitem.get_pipelinestages ()))
                    self.add_exectime (pmanager.encode_pipeline_stages(workitem.get_pipelinestages ()), start_time, end_time)
                elif status == 'FAILED':
                    print ('gpu workitem failed')
                    self.gpu.set_busy (False)
                    self.gpu.set_last_completion_time (end_time)
                elif status == 'CANCELLED':
                    print ('gpu workitem cancelled')
                    self.gpu.set_busy (False)
                    self.gpu.set_last_completion_time (end_time)

    def get_last_completion_time (self, resourcetype):
        if resourcetype == 'CPU' and self.cpu != None:
            return self.cpu.get_last_completion_time ()

        if resourcetype == 'GPU' and self.gpu != None:
            return self.gpu.get_last_completion_time ()

    def add_workitem (self, workitem, resourcetype):
        print ('add_workitem ():', self.id, workitem.id, workitem.version)
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
            else:
                 print (self.id, resourcetype, 'not complete')
        else:
             print (self.id, resourcetype, 'nothing scheduled')

        if resourcetype == 'GPU' and self.gpu.workqueue.is_empty () == False:
            if self.gpu.workqueue.get_workitem ().is_complete () == True:
                print (self.id, 'GPU workitem complete')
                workitem = self.gpu.workqueue.pop_workitem ()
                return workitem
            else:
                 print (self.id, resourcetype, 'not complete')

        else:
             print (self.id, resourcetype, 'nothing scheduled')

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

    def add_exectime (self, pipelinestages, starttime, endtime):
        timediff = endtime - starttime
        seconds = timediff.total_seconds ()

        if pipelinestages not in self.exectimes:
            self.exectimes[pipelinestages] = [seconds, 1]
            self.max_exectimes[pipelinestages] = seconds
        else:
            if seconds > self.max_exectimes[pipelinestages]:
                self.max_exectimes[pipelinestages] = seconds
            avg_time = self.exectimes[pipelinestages][0]
            count = self.exectimes[pipelinestages][1]
            new_avg_time = ((avg_time * count) + seconds) / (count + 1)
            self.exectimes[pipelinestages] = [new_avg_time, count + 1]

    def get_max_exectime (self, pipelinestages):
        if pipelinestages not in self.max_exectimes:
            return 0
        else:
            return self.max_exectimes[pipelinestages]

    def get_exectime (self, pipelinestages):
        if pipelinestages not in self.exectimes:
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
            for node in availableresources['available']:
                nodeid = node['id']
                resources[str(nodeid)] = copy.deepcopy (self.nodesdict[str(nodeid)])
                resources[str(nodeid)].output_location = node['output_location']

        if len (availableresources['reserved']) > 0:
            for i in availableresources['reserved']:
                reservedresources[str(i)] = copy.deepcopy (self.nodesdict[str(i)])

        self.nodesdict = resources
        self.reservednodesdict = reservedresources

        self.nodes = copy.deepcopy (list (self.nodesdict.values ()))
        self.reservednodes = copy.deepcopy (list (self.reservednodesdict.values ()))

        print ('nodes:', len(self.nodes), 'reserved nodes:', len (self.reservednodes))

    def get_max_exectime (self, pipelinestages, resource_id):
        resources = self.get_resources ()

        max_exectime = 0
        for resource in resources:
            if resource.id == resource_id:
                continue

            if resource.get_max_exectime (pipelinestages) > max_exectime:
                max_exectime = resource.get_max_exectime (pipelinestages)

        return max_exectime

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
