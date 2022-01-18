import yaml
import copy
import datetime

from parslflux.workqueue import WorkItemQueue

class CPU:
    def __init__ (self, name):
        self.name = name
        self.workqueue = WorkItemQueue ()
        self.busy = False
        self.last_completion_time = None
        self.idle_periods = []

    def reinit (self):
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
        return self.last_completion_time

    def clear_completion_time (self):
        self.last_completion_time = None

    def set_idle_start_time (self, time):
        self.idle_start_time = time

    def add_idle_period (self, now):
        if self.idle_start_time != None:
            self.idle_periods.append ([self.idle_start_time, now])
        #self.idle_periods.append ([self.last_completion_time, now])

    def report_idle_periods(self, starttime, endtime):
        ret = []
        for idle_period in self.idle_periods:
            if (idle_period[0] <= starttime and idle_period[1] > starttime) or \
                (idle_period[0] >= starttime and idle_period[0] <= endtime):
                ret.append(idle_period)

        return ret

class GPU:
    def __init__ (self, name):
        self.name = name
        self.workqueue = WorkItemQueue ()
        self.busy = False
        self.last_completion_time = None
        self.idle_periods = []

    def reinit (self):
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
        return self.last_completion_time

    def clear_completion_time (self):
        self.last_completion_time = None

    def set_idle_start_time (self, time):
        self.idle_start_time = time

    def add_idle_period (self, now):
        if self.idle_start_time != None:
            self.idle_periods.append ([self.idle_start_time, now])
        #self.idle_periods.append ([self.last_completion_time, now])

    def report_idle_periods(self, starttime, endtime):
        ret = []
        for idle_period in self.idle_periods:
            if (idle_period[0] <= starttime and idle_period[1] > starttime) or \
                (idle_period[0] >= starttime and idle_period[0] <= endtime):
                ret.append(idle_period)

        return ret

class Resource:

    def __init__ (self, i):
        self.id = "c" + str(i)
        self.hostname = "c" + str (i)
        self.cpu = None
        self.gpu = None
        self.workqueue = WorkItemQueue ()
        self.busy = False
        self.exectimes = {}
        self.max_exectimes = {}
        self.transfertimes = {}
        self.counts = {}
        self.executionqueues = [-1, -1]

    def add_cpu (self, cpu):
        self.cpu = CPU (cpu)

    def add_gpu (self, gpu):
        self.gpu = GPU (gpu)

    def remove_cpu (self):
        self.cpu = None

    def remove_gpu (self):
        self.gpu = None

    def is_idle (self):
        cpu_free = False
        if self.cpu != None:
            if self.cpu.is_busy () == False:
                cpu_free = True

        gpu_free = False
        if self.gpu != None:
            if self.gpu.is_busy () == False:
                gpu_free = True

        #print ('is_idle ():', self.id, cpu_free, gpu_free)

        return cpu_free, gpu_free

    def is_idle_whole (self):
        if self.busy == True:
            return False
        return True

    def is_empty_whole (self):
        #print (self.workqueue.is_empty ())
        if self.workqueue.is_empty () == True:
            return True

        return False

    def is_empty (self):
        cpu_empty = False

        if self.cpu == None:
            cpu_empty = False
        elif self.cpu.workqueue.is_empty() == True:
            cpu_empty = True

        gpu_empty = False

        if self.gpu == None:
            gpu_empty = False
        elif self.gpu.workqueue.is_empty() == True:
            gpu_empty = True

        #print ('is_empty ():', self.id, cpu_empty, gpu_empty)

        return cpu_empty, gpu_empty

    def clear_completion_times (self):
        if self.cpu != None:
            self.cpu.clear_completion_time ()

        if self.gpu != None:
            self.gpu.clear_completion_time ()

    def set_idle_start_time (self, resourcetype, now):
        if resourcetype == 'CPU' and self.cpu != None:
            self.cpu.set_idle_start_time (now)

        if resourcetype == 'GPU' and self.gpu != None:
            self.gpu.set_idle_start_time (now)

    def add_idle_period (self, resourcetype, now):
        if resourcetype == 'CPU' and self.cpu != None:
            #if self.cpu.get_last_completion_time () != None:
            self.cpu.add_idle_period (now)

        if resourcetype == 'GPU' and self.gpu != None:
            #if self.gpu.get_last_completion_time () != None:
            self.gpu.add_idle_period (now)

    def report_idle_periods (self, starttime, endtime):
        cpu_idle_periods = None
        if self.cpu != None:
            cpu_idle_periods = self.cpu.report_idle_periods (starttime, endtime)

        gpu_idle_periods = None
        if self.gpu != None:
            gpu_idle_periods = self.gpu.report_idle_periods(starttime, endtime)

        return cpu_idle_periods, gpu_idle_periods

    def schedule (self, rmanager, pmanager, resourcetype, thread_exec, env):
        if resourcetype == 'CPU' and self.cpu == None:
            #print (self.id, 'CPU not available')
            return
        if resourcetype == 'GPU' and self.gpu == None:
            #print (self.id, 'GPU not available')
            return

        if resourcetype == 'CPU' and self.cpu.workqueue.is_empty () == False:
            timeout = self.get_timeout_value (rmanager, pmanager, resourcetype)
            self.add_idle_period(resourcetype, env.now)
            self.cpu.workqueue.get_workitem ().submit (pmanager, timeout, thread_exec, env)
            self.cpu.set_busy (True)
            self.cpu.set_last_completion_time (None)
            return

        if resourcetype == 'GPU' and self.gpu.workqueue.is_empty () == False:
            timeout = self.get_timeout_value (rmanager, pmanager, resourcetype)
            self.add_idle_period(resourcetype, env.now)
            self.gpu.workqueue.get_workitem ().submit (pmanager, timeout, thread_exec, env)
            self.gpu.set_busy (True)
            self.gpu.set_last_completion_time (None)
            return

    def schedule_whole (self, rmanager, pmanager):
        if self.workqueue.is_empty () == False:
            timeout = self.get_timeout_value_whole (rmanager, pmanager)
            self.workqueue.get_workitem ().submit (pmanager, timeout)
            self.set_busy (True)
            return

    def get_timeout_value_whole (self, rmanager, pmanager):
        return 20000

    def set_busy (self, busy):
        self.busy = busy

    def get_timeout_value (self, rmanager, pmanager, resourcetype):
        #print ('get_timeout_value ():', self.id)

        if resourcetype == 'CPU' and self.cpu != None and self.cpu.workqueue.is_empty () == False:
            workitem = self.cpu.workqueue.get_workitem ()
            workitem_pipelinestages = workitem.get_pipelinestages ()

            max_exectime = self.get_max_exectime (workitem_pipelinestages) * 2

            if max_exectime == 0:#first time execution
                max_exectime = rmanager.get_max_exectime (workitem_pipelinestages, self.id) * 2

            if max_exectime == 0:#no one has completed their execution
                max_exectime = 20 * 60

            return max_exectime

        if resourcetype == 'GPU' and self.gpu != None and self.gpu.workqueue.is_empty () == False:
            workitem = self.gpu.workqueue.get_workitem ()
            workitem_pipelinestages = workitem.get_pipelinestages ()

            max_exectime = self.get_max_exectime (workitem_pipelinestages) * 2

            if max_exectime == 0:#first time execution
                max_exectime = rmanager.get_max_exectime (workitem_pipelinestages, self.id) * 2

            if max_exectime == 0:#no one has completed their execution
                max_exectime = 15 * 60

            return max_exectime


    def get_status_whole (self, pmanager):
        #print ('get_status ():', self.id)

        if self.workqueue.is_empty () == False:
            workitem = self.workqueue.get_workitem ()
            ret, start_time, end_time, status, r_timetaken = workitem.probe_status ()
            if ret == True:
                self.set_busy (False)

    def get_status (self, rmanager, pmanager, threads, outputfile):
        #print ('get_status ():', self.id)
        #first cpu
        if self.cpu != None and self.cpu.workqueue.is_empty () == False:
            workitem = self.cpu.workqueue.get_workitem ()
            ret, start_time, end_time, status, r_timetaken = workitem.probe_status (threads[0], outputfile)
            if ret == True:
                if status == 'SUCCESS':
                    #print ('cpu workitem complete')
                    self.cpu.set_busy (False)
                    self.cpu.set_last_completion_time (end_time)
                    self.cpu.set_idle_start_time (end_time)
                    self.add_count (workitem.get_pipelinestages ())
                    rmanager.add_exectime (self.cpu.name, workitem.get_pipelinestages (), start_time, end_time)
                    self.add_exectime(workitem.get_pipelinestages(), start_time,
                                      end_time)
                elif status == 'FAILED':
                    #print ('cpu workitem failed')
                    self.cpu.set_busy (False)
                    self.cpu.set_last_completion_time (end_time)
                elif status == 'CANCELLED':
                    #print ('cpu workitem cancelled')
                    self.cpu.set_busy (False)
                    self.cpu.set_last_completion_time (end_time)

                #self.add_transfer_time (pmanager.encode_pipeline_stages(workitem.get_pipelinestages ()), r_timetaken)

        #now gpu
        if self.gpu != None and self.gpu.workqueue.is_empty () == False:
            workitem = self.gpu.workqueue.get_workitem ()
            ret, start_time, end_time, status, r_timetaken = workitem.probe_status (threads[1], outputfile)
            if ret == True:
                if status == 'SUCCESS':
                    #print ('gpu workitem complete')
                    self.gpu.set_busy (False)
                    self.gpu.set_last_completion_time (end_time)
                    self.gpu.set_idle_start_time(end_time)
                    self.add_count (workitem.get_pipelinestages ())
                    rmanager.add_exectime (self.gpu.name, workitem.get_pipelinestages (), start_time, end_time)
                    self.add_exectime(workitem.get_pipelinestages (), start_time,
                                      end_time)
                elif status == 'FAILED':
                    #print ('gpu workitem failed')
                    self.gpu.set_busy (False)
                    self.gpu.set_last_completion_time (end_time)
                elif status == 'CANCELLED':
                    #print ('gpu workitem cancelled')
                    self.gpu.set_busy (False)
                    self.gpu.set_last_completion_time (end_time)

                #self.add_transfer_time (pmanager.encode_pipeline_stages(workitem.get_pipelinestages ()), r_timetaken)

    def get_last_completion_time (self, resourcetype):
        if resourcetype == 'CPU' and self.cpu != None:
            return self.cpu.get_last_completion_time ()

        if resourcetype == 'GPU' and self.gpu != None:
            return self.gpu.get_last_completion_time ()

    def add_workitem_full (self, workitem):
        #print ('add_workitem ():', self.id, workitem.id, workitem.version)
        self.workqueue.add_workitem (workitem)

    def add_workitem (self, workitem, resourcetype):
        #print ('add_workitem ():', self.id, workitem.id, workitem.version)
        if resourcetype == 'CPU':
            if self.cpu == None:
                #print (self.id, 'CPU not available')
                return
            self.cpu.workqueue.add_workitem (workitem)
            self.executionqueues[0] = workitem.version

        if resourcetype == 'GPU':
            if self.gpu == None:
                #print (self.id, 'GPU not available')
                return
            self.gpu.workqueue.add_workitem (workitem)
            self.executionqueues[1] = workitem.version

    def pop_if_complete_whole (self):
        if self.workqueue.is_empty () == False:
            if self.workqueue.get_workitem ().is_complete () == True:
                #print (self.id, 'workitem complete')
                workitem = self.workqueue.pop_workitem ()
                return workitem

        return None

    def pop_if_complete (self, resourcetype):
        if resourcetype == 'CPU' and self.cpu == None:
            #print (self.id, 'CPU not available')
            return None

        if resourcetype == 'GPU' and self.gpu == None:
            #print (self.id, 'GPU not available')
            return None

        if resourcetype == 'CPU' and self.cpu.workqueue.is_empty () == False:
            if self.cpu.workqueue.get_workitem ().is_complete () == True:
                #print (self.id, 'CPU workitem complete')
                workitem = self.cpu.workqueue.pop_workitem ()
                self.executionqueues[0] = -1
                return workitem

        if resourcetype == 'GPU' and self.gpu.workqueue.is_empty () == False:
            if self.gpu.workqueue.get_workitem ().is_complete () == True:
                #print (self.id, 'GPU workitem complete')
                workitem = self.gpu.workqueue.pop_workitem ()
                self.executionqueues[1] = -1
                return workitem

        return None

    def get_workitem (self, resourcetype):
        if resourcetype == 'CPU' and self.cpu == None:
            return None

        if resourcetype == 'GPU' and self.gpu == None:
            return None

        if resourcetype == 'CPU' and self.cpu.workqueue.is_empty () == False:
            workitem = self.cpu.workqueue.get_workitem()
            return workitem

        if resourcetype == 'GPU' and self.gpu.workqueue.is_empty () == False:
            workitem = self.gpu.workqueue.get_workitem()
            return workitem

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
        seconds = endtime - starttime

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

    def add_transfer_time (self, pipelinestages, timetaken):
        if timetaken == 0:
            return

        if pipelinestages not in self.transfertimes:
            self.transfertimes[pipelinestages] = [timetaken, 1]
        else:
            avg_time = self.transfertimes[pipelinestages][0]
            count = self.transfertimes[pipelinestages][1]
            new_transfer_time = ((avg_time * count) + timetaken) / (count + 1)
            self.transfertimes[pipelinestages] = [new_transfer_time, count + 1]

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

    def get_exectime_pipelinestages (self, pmanager, pipelinestages, resourcetype):
        if resourcetype == 'CPU' and self.cpu != None:
            encoded_pipelinestages = pmanager.encode_pipeline_stages (pipelinestages)
            exectime = self.get_exectime(encoded_pipelinestages)
            if exectime == 0:
                return None
            else:
                return exectime
        else:
            return None

        if resourcetype == 'GPU' and self.gpu != None:
            encoded_pipelinestages = pmanager.encode_pipeline_stages (pipelinestages)
            exectime = self.get_exectime(encoded_pipelinestages)
            if exectime == 0:
                return None
            else:
                return exectime
        else:
            return None

    def get_exectime_current (self, pmanager, resourcetype):

        if resourcetype == 'CPU' and self.cpu != None:
            if self.cpu.workqueue.is_empty () == False:
                workitem = self.cpu.workqueue.get_workitem ()
                workitem_pipelinestages = workitem.get_pipelinestages ()
                encoded_workitem_pipelinestages = pmanager.encode_pipeline_stages (workitem_pipelinestages)
                exectime = self.get_exectime(encoded_workitem_pipelinestages)
                if exectime == 0:
                    return None
                else:
                    return exectime
            else:
                return None

        if resourcetype == 'GPU' and self.gpu != None:
            if self.gpu.workqueue.is_empty () == False:
                workitem = self.gpu.workqueue.get_workitem ()
                workitem_pipelinestages = workitem.get_pipelinestages ()
                encoded_workitem_pipelinestages = pmanager.encode_pipeline_stages (workitem_pipelinestages)
                exectime = self.get_exectime(encoded_workitem_pipelinestages)
                if exectime == 0:
                    return None
                else:
                    return exectime
            else:
                return None

    def get_transfertime (self, pipelinestages):
        if pipelinestages not in self.exectimes:
            return 0
        else:
            return self.transfertimes[pipelinestages][0]

    def get_work_left (self, resourcetype, current_time):
        if resourcetype == 'CPU' and self.cpu != None:
            if self.cpu.workqueue.is_empty () == False:
                workitem = self.cpu.workqueue.get_workitem ()
                workitem_pipelinestages = workitem.get_pipelinestages ()

                exectime = self.get_exectime (workitem_pipelinestages)

                if exectime == 0:
                    return None
                else:
                    work_remaining = (exectime - (current_time - workitem.scheduletime))/exectime
                    if work_remaining < 0:
                        work_remaining = 0
                return work_remaining

        if resourcetype == 'GPU' and self.gpu != None:
            if self.gpu.workqueue.is_empty () == False:
                workitem = self.gpu.workqueue.get_workitem ()
                workitem_pipelinestages = workitem.get_pipelinestages ()

                exectime = self.get_exectime (workitem_pipelinestages)

                if exectime == 0:
                    return None
                else:
                    work_remaining = (exectime - (current_time - workitem.scheduletime))/exectime
                    if work_remaining < 0:
                        work_remaining = 0

                return work_remaining

            return None

class ResourceManager:
    def __init__ (self, resourcefile, availablefile):
        self.resourcefile = resourcefile
        self.availablefile = availablefile
        self.nodes = []
        self.reservednodes = []
        self.nodesdict = {}
        self.reservednodesdict = {}
        self.exectimes = {}
        self.max_exectimes = {}
        self.new_resource_id = 150
        self.cpunodes = []
        self.gpunodes = []
        self.cpuid_counter = 0
        self.gpuid_counter = 0

    def get_new_resource (self):
        resource = Resource (self.new_resource_id_start)
        self.new_resource_id += 1
        return resource

    def add_resource (self, new_resource):
        self.nodes.append (new_resource)

    def parse_resources (self):
        yaml_resourcefile = open (self.resourcefile)
        resources = yaml.load (yaml_resourcefile, Loader=yaml.FullLoader)

        for cputype in resources['available']['CPU']:
            print (cputype)
            count = cputype['count']
            for i in range (0, count):
                new_resource = Resource ('c' + str(self.cpuid_counter))
                new_resource.add_cpu(cputype['id'])
                self.nodesdict[new_resource.id] = new_resource
                self.cpunodes.append(copy.deepcopy(self.nodesdict[new_resource.id]))
                self.cpuid_counter += 1

        for gputype in resources['available']['GPU']:
            count = gputype['count']
            for i in range (0, count):
                new_resource = Resource ('g' + str(self.gpuid_counter))
                new_resource.add_gpu(gputype['id'])
                self.nodesdict[new_resource.id] = new_resource
                self.gpunodes.append(copy.deepcopy(self.nodesdict[new_resource.id]))
                self.gpuid_counter += 1

        self.availablenodesdict = self.nodesdict

        self.nodes = copy.deepcopy(list(self.availablenodesdict.values()))

        print (self.nodes)

    def parse_resources_old (self):
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
                    new_resource.add_cpu (node['name'])
                    self.nodesdict[str(i)] = new_resource

        #parse gpus
        for gpu in arc_resources['gpus']:
            for gpurange in gpu['range']:
                for i in range (gpurange[0], gpurange[1] + 1):
                    self.nodesdict[str(i)].add_gpu (gpu['name'])

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

                if node['cpu'] == 0:
                    resources[str(nodeid)].remove_cpu ()
                else:
                    self.cpunodes.append (copy.deepcopy (resources[str(nodeid)]))
                if node['gpu'] == 0:
                    resources[str(nodeid)].remove_gpu ()
                else:
                    self.gpunodes.append (copy.deepcopy (resources[str(nodeid)]))

        if len (availableresources['reserved']) > 0:
            for i in availableresources['reserved']:
                reservedresources[str(i)] = copy.deepcopy (self.nodesdict[str(i)])

        self.availablenodesdict = resources
        self.reservednodesdict = reservedresources

        self.nodes = copy.deepcopy (list (self.availablenodesdict.values ()))
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

    def get_cpu_resources_count (self):
        return len (self.cpunodes)

    def get_gpu_resources_count (self):
        return len (self.gpunodes)

    def add_exectime (self, resourcename, pipelinestages, starttime, endtime):
        seconds = endtime - starttime

        if resourcename not in self.exectimes:
            self.exectimes[resourcename] = {}
            self.max_exectimes[resourcename] = {}
            if pipelinestages not in self.exectimes:
                self.exectimes[resourcename][pipelinestages] = [seconds, 1]
                self.max_exectimes[resourcename][pipelinestages] = seconds
        else:
            if pipelinestages not in self.exectimes:
                self.exectimes[resourcename][pipelinestages] = [seconds, 1]
                self.max_exectimes[resourcename][pipelinestages] = seconds
            else:
                if seconds > self.max_exectimes[resourcename][pipelinestages]:
                    self.max_exectimes[resourcename][pipelinestages] = seconds
                avg_time = self.exectimes[resourcename][pipelinestages][0]
                count = self.exectimes[resourcename][pipelinestages][1]
                new_avg_time = ((avg_time * count) + seconds) / (count + 1)
                self.exectimes[resourcename][pipelinestages] = [new_avg_time, count + 1]

        #print (self, self.exectimes)

    def get_resource (self, resource_id):
        for node in self.nodes:
            if node.id == resource_id:
                return node
        return None

    def get_resource_exist (self, resource_id):
        resource_id = resource_id[1:]
        if resource_id in self.nodesdict.keys():
            return self.nodesdict[resource_id]

        return None

    def get_resources (self):
        return self.nodes

    def get_resources_type (self, resourcetype):
        resources = []
        for resource in self.nodes:
            if resourcetype == 'CPU' and resource.cpu != None:
                resources.append(resource)
            elif resourcetype == 'GPU' and resource.gpu != None:
                resources.append(resource)
        return resources

    def request_reserved_resource (self):
        if len (self.reservednodesdict.keys()) > 0:
            new_key = self.reservednodesdict.keys()[0]
            new_resource = self.reservednodesdict.pop (new_key)
            self.nodesdict[new_key] = new_resource
            self.nodes.append (new_resource)
            return new_resource

        return None
